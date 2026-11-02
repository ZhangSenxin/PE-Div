#!/usr/bin/env python3
"""从标签、氨基酸序列或DNA翻译结果开始的多规则联合富集分析（Python 3.9+）。"""
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET
from xlsx_stream import write_workbook

# 标准遗传密码表；不搜索起始密码子，不截断，不自动反向互补。
CODONS = {a+b+c: aa for (a, b, c), aa in zip(
    ((a, b, c) for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG')}


def translate_dna(seq, frame=0, stop_policy='exclude'):
    """按给定阅读框完整翻译；返回 (氨基酸序列, 排除原因)。"""
    if frame not in (0, 1, 2):
        raise ValueError('reading frame 必须为0、1或2')
    if stop_policy not in ('exclude','keep'):
        raise ValueError('stop policy 必须为exclude或keep')
    seq = seq.upper()[frame:]
    if not seq or len(seq) % 3:
        return None, 'invalid_length'
    if re.fullmatch('[ACGT]+', seq) is None:
        return None, 'invalid_bases'
    aa = ''.join(CODONS[seq[i:i+3]] for i in range(0, len(seq), 3))
    if '*' in aa and stop_policy == 'exclude':
        return None, 'stop_codon'
    return aa, None


def read_meta(path):
    """只读 xlsx，兼容合并表头旁的样品名称列；也接受 CSV/TSV。"""
    if path.suffix.lower() == '.xlsx':
        ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        with zipfile.ZipFile(path) as z:
            strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                strings = [''.join(x.itertext()) for x in ET.fromstring(
                    z.read('xl/sharedStrings.xml')).findall('m:si', ns)]
            wb = ET.fromstring(z.read('xl/workbook.xml'))
            sheet = wb.find('m:sheets/m:sheet', ns)
            rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
            target = next(x.attrib['Target'] for x in rels if x.attrib['Id'] == rid)
            target = target.lstrip('/') if target.startswith('/') else 'xl/' + target
            rows = []
            for row in ET.fromstring(z.read(target)).findall('m:sheetData/m:row', ns):
                vals = {}
                for c in row:
                    letters = re.match(r'[A-Z]+', c.attrib['r'])[0]
                    col = 0
                    for char in letters:
                        col = col * 26 + ord(char) - 64
                    v = c.find('m:v', ns)
                    value = v.text if v is not None else ''
                    if c.attrib.get('t') == 's':
                        value = strings[int(value)]
                    elif c.attrib.get('t') == 'inlineStr':
                        value = ''.join(c.find('m:is', ns).itertext())
                    vals[col - 1] = value or ''
                rows.append([vals.get(i, '') for i in range(max(vals, default=0) + 1)])
    else:
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.reader(f, delimiter='\t' if path.suffix.lower() in {'.tsv', '.txt'} else ','))
    if not rows:
        raise ValueError('meta 为空')
    headers = [x.strip() for x in rows[0]]
    named = [x for x in headers[1:] if x]
    if len(named) != len(set(named)):
        raise ValueError('meta 存在重复列名')
    result = {}
    for values in rows[1:]:
        if not any(values):
            continue
        batch = values[0].strip()
        if not batch or batch in result:
            raise ValueError('meta 的 batch ID 为空或重复: ' + batch)
        safe_name(batch)
        result[batch] = {h: values[i].strip() if i < len(values) else ''
                         for i, h in enumerate(headers) if h and i > 0}
    return result


def safe_name(value):
    if not re.fullmatch(r'[\w.-]+', value) or value in {'.', '..'}:
        raise ValueError('输出名称只允许字母、数字、下划线、连字符和点: ' + value)
    return value


def group(expr):
    expr = re.sub(r'\s+', ' ', expr.strip().lower())
    if expr == 'pre pool':
        return expr
    expr = re.sub(r'^pos\s+', '', expr)
    if re.fullmatch(r'lib\d+', expr):
        return 'pos lib' + str(int(expr[3:]))
    if re.fullmatch(r'lib\(\s*\d+(\s*\+\s*\d+)*\s*\)', expr):
        nums = [int(n) for n in re.findall(r'\d+', expr)]
        if len(nums) != len(set(nums)):
            raise ValueError('合并组中 lib 重复: ' + expr)
        return 'pos lib(' + '+'.join(map(str, sorted(nums))) + ')'
    raise ValueError('不支持的组表达式: ' + expr)


def parse_rules(path):
    result = []
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        line = re.sub(r'^\d+\s*[.、]\s*', '', line)
        parts = re.split(r'\s+vs\s+', line, flags=re.I)
        if len(parts) != 2:
            raise ValueError(f'{path}:{line_no}: 规则必须为 post vs pre')
        pair = tuple(group(x) for x in parts)
        if pair in result or pair[0] == pair[1]:
            raise ValueError(f'{path}:{line_no}: 重复比较或自身比较')
        result.append(pair)
    if not result:
        raise ValueError('rules 文件没有有效比较: ' + str(path))
    return result


def members(expr):
    return ['pre pool'] if expr == 'pre pool' else [f'pos lib{n}' for n in re.findall(r'\d+', expr)]


def read_counts(path, excluded, drop_star, input_type='label', frame=0, aa_output=None, audit=None, stop_policy='exclude'):
    """识别单列/双列输入，可先翻译DNA，再合并同义编码的reads。"""
    counts = Counter()
    skipped = 0
    two_columns = None
    reasons = Counter()
    translation_cache = {}
    input_reads = 0
    out = None
    if aa_output is not None:
        aa_output = Path(aa_output)
        if aa_output.resolve() == path.resolve():
            raise ValueError('氨基酸输出不能覆盖原始DNA文件')
        aa_output.parent.mkdir(parents=True, exist_ok=True)
        out = aa_output.open('x', encoding='utf-8', newline='')
        out.write('ID\tSeq\n')
    try:
      with path.open(encoding='utf-8-sig') as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                skipped += 1
                reasons['blank_lines'] += 1
                continue
            fields = [value.strip() for value in line.rstrip('\r\n').split('\t')]
            if two_columns is None:
                two_columns = fields == ['ID', 'Seq']
                if two_columns:
                    continue
            if two_columns:
                if len(fields) != 2 or not fields[0] or not fields[1]:
                    raise ValueError(f'{path}:{n}: ID/Seq 数据行必须恰好两列且均非空')
                key = fields[1]
            else:
                if len(fields) != 1:
                    raise ValueError(f'{path}:{n}: 双列文件必须使用制表符分隔并以 ID/Seq 为表头')
                key = fields[0]
            input_reads += 1
            if not key or key in excluded or (drop_star and '*' in key):
                skipped += 1
                reasons['excluded_label'] += 1
                continue
            if any(c.isspace() for c in key):
                raise ValueError(f'{path}:{n}: 标签或序列中不能包含空白')
            if input_type == 'dna':
                dna = key.upper()
                if dna not in translation_cache:
                    translation_cache[dna] = translate_dna(dna, frame, stop_policy)
                key, reason = translation_cache[dna]
                if reason:
                    skipped += 1
                    reasons[reason] += 1
                    continue
            elif input_type == 'aa':
                key = key.upper()
                reason = 'stop_codon' if '*' in key and stop_policy == 'exclude' else ('invalid_aa' if re.fullmatch('[ACDEFGHIKLMNPQRSTVWY*]+', key) is None else None)
                if reason:
                    skipped += 1
                    reasons[reason] += 1
                    continue
            counts[key] += 1
            if out is not None:
                read_id = fields[0] if two_columns else str(input_reads-1)
                out.write(read_id + '\t' + key + '\n')
    finally:
        if out is not None:
            out.close()
    if not counts:
        raise ValueError('输入文件没有有效记录: ' + str(path))
    if audit is not None:
        audit.update(input_type=input_type, reading_frame=frame if input_type == 'dna' else None,
                     stop_policy=stop_policy,
                     input_reads=input_reads, retained_reads=sum(counts.values()),
                     rejected_reads=input_reads-sum(counts.values()), rejection_counts=dict(reasons),
                     unique=len(counts))
        audit.update(stop_containing_retained_reads=sum(v for k,v in counts.items() if '*' in k),
                     stop_containing_retained_unique=sum('*' in k for k in counts))
        if input_type == 'dna':
            valid_dna_unique = sum(reason is None for _, reason in translation_cache.values())
            audit.update(dna_unique=len(translation_cache), valid_dna_unique=valid_dna_unique,
                         synonymous_unique_collapsed=valid_dna_unique-len(counts),
                         aa_output=str(aa_output) if aa_output is not None else None)
    return counts, skipped


def shrink(scores, variances):
    tau = max(0.0, statistics.pvariance(scores) - statistics.mean(variances))
    return tau, [tau / (v + tau) if v > 0 else 0.0 for v in variances]


def single_round(pre, post):
    labels = sorted(pre.keys() | post.keys())
    k = len(labels)
    npre, npost = sum(pre.values()), sum(post.values())
    rows = {}
    for key in labels:
        cp, cq = pre[key], post[key]
        fp, fq = (cp + 1) / (npre + k), (cq + 1) / (npost + k)
        score = math.log2(fq / fp)
        var = (1 / (cq + 1) + 1 / (cp + 1)) / math.log(2)**2
        rows[key] = dict(label=key, c_pre=cp, c_post=cq, f_pre=fp, f_post=fq,
                         S_i=score, sigma_i=math.sqrt(var), Z_i=score / math.sqrt(var))
    tau, weights = shrink([r['S_i'] for r in rows.values()], [r['sigma_i']**2 for r in rows.values()])
    for r, w in zip(rows.values(), weights):
        r.update(shrinkage_w=w, theta_i=w * r['S_i'])
    return rows, dict(K=k, N_pre=npre, N_post=npost, tau2=tau)


def isc(values, k):
    current = list(values)
    mu, sd = statistics.mean(current), statistics.pstdev(current)
    for _ in range(100):
        threshold = mu + k * sd
        truncated = [v for v in current if v <= threshold]
        new_mu, new_sd = statistics.mean(truncated), statistics.pstdev(truncated)
        if abs(new_mu - mu) < 1e-4 and abs(new_sd - sd) < 1e-4:
            return new_mu, new_sd, threshold
        current, mu, sd = truncated, new_mu, new_sd
    return mu, sd, threshold


def combine(rounds, infos, mode, isc_k):
    labels = sorted(set().union(*(r.keys() for r in rounds)))
    scores = [sum(r[key]['S_i'] for r in rounds if key in r) for key in labels]
    variances = [sum(r[key]['sigma_i']**2 for r in rounds if key in r) for key in labels]
    tau, weights = shrink(scores, variances)
    output = []
    ghost_count = 0
    for key, standard_s, standard_v, weight in zip(labels, scores, variances, weights):
        first = next((i for i, r in enumerate(rounds) if key in r and r[key]['c_post'] > 0), None)
        first_count = rounds[first][key]['c_post'] if first is not None else 0
        info = infos[first if first is not None else 0]
        ghost_freq = (first_count + 1) / (info['N_post'] + info['K'])
        total_s, total_v = standard_s, standard_v
        row = dict(label=key, S_total_standard=standard_s, sigma_total_standard=math.sqrt(standard_v),
                   shrinkage_weight=weight)
        for i, (r, info) in enumerate(zip(rounds, infos), 1):
            present = key in r
            s = r[key]['S_i'] if present else 0.0
            v = r[key]['sigma_i']**2 if present else 0.0
            if not present and mode != 'none':
                s = math.log2(1 / (info['N_post'] + info['K']) / ghost_freq)
                total_s += s
                if mode == 'legacy':
                    v = (1 + 1 / (first_count + 1)) / math.log(2)**2
                    total_v += v
                ghost_count += 1
            row.update({f'S_round{i}': s, f'sigma_round{i}': math.sqrt(v),
                        f'Z_round{i}': r[key]['Z_i'] if present else 0.0,
                        f'theta_round{i}': r[key]['theta_i'] if present else 0.0,
                        f'dual_absent_round{i}': not present})
        z = total_s / math.sqrt(total_v)
        row.update(S_total=total_s, sigma_total=math.sqrt(total_v), Z_total=z,
                   theta_total=weight * total_s,
                   significance='***' if z > 3.291 else '**' if z > 2.576 else 'depleted' if z < -2.576 else 'ns')
        output.append(row)
    mu, sd, threshold = isc([r['Z_total'] for r in output], isc_k)
    for row in output:
        row['ISC_enriched'] = row['Z_total'] > threshold
    output.sort(key=lambda r: (-r['Z_total'], r['label']))
    return output, dict(tau2_total=tau, ISC_mu=mu, ISC_sigma=sd, ISC_threshold=threshold,
                        total_seqs=len(labels), n_rounds=len(rounds), dual_absent_pairs=ghost_count,
                        significant_enriched=sum(r['ISC_enriched'] for r in output))


def select_by_z(rows, minimum_count):
    """正向富集筛选：与theta和tau²无关，不等同于ISC显著性检验。"""
    return {r['label'] for r in rows if r['Z_total']>0 and r['count_final_post']>=minimum_count}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--meta', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    rg = p.add_mutually_exclusive_group(required=True)
    rg.add_argument('--rules', type=Path, nargs='+')
    rg.add_argument('--rules-dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batch', default='all', help='all 或逗号分隔的 batch ID')
    p.add_argument('--file-template', default='{sample}_45.txt')
    p.add_argument('--input-type', choices=['label', 'dna', 'aa'], default='label',
                   help='label 原样计数；dna 先翻译；aa 校验后按氨基酸计数')
    p.add_argument('--reading-frame', type=int, choices=[0, 1, 2], default=0,
                   help='DNA翻译起点偏移，默认0即从第一个碱基开始')
    p.add_argument('--stop-policy', choices=['exclude','keep'],default='exclude',
                   help='exclude剔除含终止密码子的read；keep保留完整翻译并以*表示终止')
    p.add_argument('--aa-data-dir', type=Path, help='dna模式可将有效reads另存为ID/Seq氨基酸文件')
    p.add_argument('--translate-all-data', action='store_true',
                   help='dna模式翻译data根目录全部txt文件，包括meta未引用文件')
    p.add_argument('--dual-absent', choices=['legacy', 'score-only', 'none'], default='legacy')
    p.add_argument('--isc-k', type=float, default=2.0)
    p.add_argument('--thresholds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--exclude-label', action='append', default=[], help='额外排除的标签，可重复指定')
    p.add_argument('--keep-none', action='store_true', help='保留字面值 None')
    p.add_argument('--drop-star', action='store_true', help='排除含 * 的记录；默认原样计数')
    p.add_argument('--xlsx', action='store_true', help='兼容旧参数；现在始终只导出xlsx结果')
    p.add_argument('--check-only', action='store_true', help='检查 meta/rules/文件路径，不要求服务器数据已存在')
    args = p.parse_args(argv)
    if (args.aa_data_dir or args.translate_all_data) and args.input_type != 'dna':
        p.error('aa-data-dir 和 translate-all-data 仅用于 input-type=dna')
    if args.stop_policy == 'keep' and args.drop_star:
        p.error('stop-policy=keep不能同时使用drop-star')
    if not math.isfinite(args.isc_k) or args.isc_k <= 0 or min(args.thresholds) < 1:
        p.error('isc-k 必须为正数，thresholds 必须为正整数')
    meta = read_meta(args.meta)
    batches = list(meta) if args.batch == 'all' else args.batch.split(',')
    if not batches or len(batches) != len(set(batches)) or any(b not in meta for b in batches):
        raise ValueError('batch 不存在或重复')
    rule_paths = args.rules or sorted(args.rules_dir.glob('*.txt'))
    if not rule_paths or len({p.stem for p in rule_paths}) != len(rule_paths):
        raise ValueError('rules 为空或文件名重复')
    rules = {safe_name(path.stem): parse_rules(path) for path in rule_paths}
    excluded = set(args.exclude_label) | (set() if args.keep_none else {'None'})
    plan = []
    resolved = {}
    for batch in batches:
        for name, pairs in rules.items():
            for g in dict.fromkeys(x for pair in pairs for x in pair):
                paths = []
                for col in members(g):
                    sample = meta[batch].get(col)
                    if not sample:
                        raise ValueError(f'{batch}: meta 缺少 {col} 的样品 ID')
                    filename = args.file_template.format(sample=sample, batch=batch)
                    path = args.data / filename
                    if not path.resolve().is_relative_to(args.data.resolve()):
                        raise ValueError('样品路径必须位于 data 目录内')
                    paths.append(path)
                if len(paths) != len(set(paths)):
                    raise ValueError(f'{batch}/{g}: 合并组包含重复样品')
                resolved[batch, g] = paths
                plan.append(dict(batch=batch, rules=name, group=g, files=[str(p) for p in paths],
                                 exists=all(p.is_file() for p in paths)))
    if args.check_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    missing = sorted({str(p) for paths in resolved.values() for p in paths if not p.is_file()})
    if missing:
        raise ValueError('缺少数据文件（未启动分析）:\n' + '\n'.join(missing))
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('输出目录非空，请使用新目录以避免覆盖/混入旧结果')
    if args.aa_data_dir and args.aa_data_dir.exists() and any(args.aa_data_dir.iterdir()):
        raise ValueError('氨基酸输出目录非空，请使用新目录')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', arguments={k: str(v) for k, v in vars(args).items()},
                    threshold_filter=dict(metric='Z_total',operator='>',value=0,minimum_count_group='last_post'),
                    plan=plan, input_audit=[], translation_audit=[], rules=rules, source_method='signal_alignment_2 / enrichment_further.py + multi_round_enrichment_general_pipeline.py',
                    translation=dict(input_type=args.input_type, genetic_code=1, reading_frame=args.reading_frame,
                                     incomplete_codon='exclude', ambiguous_bases='exclude', stop_codon=args.stop_policy) if args.input_type == 'dna' else None,
                    rules_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in rule_paths},
                    meta_sha256=hashlib.sha256(args.meta.read_bytes()).hexdigest())
    manifest_path = args.output / 'manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    save_manifest()
    try:
        receipts=[]
        stage_stats=[]
        combined_stats=[]
        threshold_stats=[]
        cache = {}
        def load_sample(path):
            if path not in cache:
                print(f'读取样品: {path} (input-type={args.input_type})', flush=True)
                audit = {}
                aa_path = args.aa_data_dir / path.relative_to(args.data) if args.aa_data_dir else None
                c, skipped = read_counts(path, excluded, args.drop_star, args.input_type,
                                         args.reading_frame, aa_path, audit, args.stop_policy)
                cache[path] = (c, skipped, audit)
                if args.input_type == 'dna':
                    manifest['translation_audit'].append(dict(file=str(path), **audit))
                save_manifest()
                print(f'  保留 {sum(c.values()):,} reads, {len(c):,} unique; 跳过 {skipped:,}', flush=True)
            return cache[path]
        if args.translate_all_data:
            for path in sorted(set(args.data.glob('*.txt')) | {p for paths in resolved.values() for p in paths}):
                load_sample(path)
        for batch in batches:
            groups = {}
            for (b, g), paths in resolved.items():
                if b != batch:
                    continue
                counts = Counter()
                for path in paths:
                    c, skipped, audit = load_sample(path)
                    if not any(r['batch'] == batch and r['file'] == str(path) for r in manifest['input_audit']):
                        manifest['input_audit'].append(dict(batch=batch, file=str(path), reads=sum(c.values()),
                                                            unique=len(c), skipped=skipped, bytes=path.stat().st_size,
                                                            **{k:v for k,v in audit.items() if k != 'unique'}))
                    counts.update(c)
                groups[g] = counts
            batch_dir = args.output / batch
            batch_dir.mkdir()
            results = {}
            for name, pairs in rules.items():
                dest = batch_dir / name
                dest.mkdir()
                rounds, infos = [], []
                workbook_sheets={}
                for i, (post, pre) in enumerate(pairs, 1):
                    r, info = single_round(groups[pre], groups[post])
                    rounds.append(r)
                    infos.append(dict(info, round=i, post=post, pre=pre))
                    workbook_sheets[f'round{i}']=(list(r.values()),list(next(iter(r.values()))))
                    overlap=len(groups[pre].keys() & groups[post].keys())
                    stage_stats.append(dict(batch=batch,rules=name,**infos[-1],
                                            pre_unique=len(groups[pre]),post_unique=len(groups[post]),overlap=overlap,
                                            pre_only=len(groups[pre])-overlap,post_only=len(groups[post])-overlap,
                                            jaccard=overlap/info['K']))
                rows, summary = combine(rounds, infos, args.dual_absent, args.isc_k)
                for row in rows:
                    for g in dict.fromkeys(x for pair in pairs for x in pair):
                        row['count_' + g.replace(' ', '_')] = groups[g][row['label']]
                    row['count_final_post'] = groups[pairs[-1][0]][row['label']]
                summary_rows = [dict(metric=k, value=v) for k, v in summary.items()]
                sheets={'summary':(summary_rows,['metric','value']),'results':(rows,list(rows[0])),
                        'round_summary':(infos,list(infos[0])),**workbook_sheets}
                receipts.extend(write_workbook(dest/'analysis.xlsx',sheets))
                combined_stats.append(dict(batch=batch,rules=name,**summary,
                                           positive_theta=sum(r['theta_total']>0 for r in rows),
                                           stop_containing_unique=sum('*' in r['label'] for r in rows)))
                results[name] = (rows, summary)
                print(f'{batch}/{name}: {len(rows)} labels, tau2={summary["tau2_total"]:.6g}', flush=True)
            # 统一按Z_total筛选，避免全局收缩使rule2的theta全零而丢失正向标签。
            indexed = {name: {r['label']: r for r in rows} for name, (rows, _) in results.items()}
            sheets = {}
            fields = ['label'] + [f'{metric}_{name}' for name in rules for metric in ['theta', 'Z', 'selected', 'count_final_post']]
            for threshold in sorted(set(args.thresholds)):
                selected = {name: select_by_z(rows,threshold) for name, (rows, _) in results.items()}
                output = []
                for key in sorted(set().union(*selected.values())):
                    row = dict(label=key)
                    for name in rules:
                        r = indexed[name].get(key, {})
                        row.update({f'theta_{name}': r.get('theta_total', 0), f'Z_{name}': r.get('Z_total', 0),
                                    f'selected_{name}': key in selected[name], f'count_final_post_{name}': r.get('count_final_post', 0)})
                    output.append(row)
                output.sort(key=lambda r: (-r[f'Z_{next(iter(rules))}'], r['label']))
                sheets[f'threshold_{threshold}'] = (output, fields)
                threshold_stats.append(dict(batch=batch,minimum_final_count=threshold,union_count=len(output),
                                            selection_metric='Z_total',score_min_exclusive=0,
                                            **{f'selected_{name}':len(selected[name]) for name in rules}))
            receipts.extend(write_workbook(batch_dir/'filtered_by_threshold.xlsx',sheets))
        sample_rows=[]
        seen=set()
        for item in manifest['input_audit']:
            if item['file'] in seen:
                continue
            seen.add(item['file'])
            sample_rows.append({k:item[k] for k in ('file','input_reads','reads','unique','rejected_reads',
                                                    'stop_containing_retained_reads','stop_containing_retained_unique')})
        stats_sheets={'samples':(sample_rows,list(sample_rows[0])),
                      'comparisons':(stage_stats,list(stage_stats[0])),
                      'combined':(combined_stats,list(combined_stats[0])),
                      'thresholds':(threshold_stats,list(threshold_stats[0]))}
        receipts.extend(write_workbook(args.output/'statistics.xlsx',stats_sheets))
        (args.output/'xlsx_export_receipt.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2),encoding='utf-8')
        (args.output/'stage_statistics.json').write_text(json.dumps(dict(samples=sample_rows,comparisons=stage_stats,
                                                                        combined=combined_stats,thresholds=threshold_stats),ensure_ascii=False,indent=2),encoding='utf-8')
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=str(exc))
        raise
    finally:
        save_manifest()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
