import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import Counter, defaultdict
SEED=42
CAT_COL="level_1"
LEN_POS="fa_len"
LEN_NEG="trunc_len"
SPLITS=["train","val","test"]

def isc_threshold(data, k=2, max_iter=100, tol=1e-4, tail='right'):
    """ISC（迭代Sigma裁剪）"""
    current_data = data.copy()
    mu_prev, sigma_prev = np.mean(data), np.std(data)
    for _ in range(max_iter):
        if tail == 'right':
            threshold = mu_prev + k * sigma_prev
            truncated = current_data[current_data <= threshold]
        else:
            threshold = mu_prev - k * sigma_prev
            truncated = current_data[current_data >= threshold]
        if len(truncated) == 0:
            break
        mu_new, sigma_new = np.mean(truncated), np.std(truncated)
        if abs(mu_new - mu_prev) < tol and abs(sigma_new - sigma_prev) < tol:
            break
        mu_prev, sigma_prev = mu_new, sigma_new
        current_data = truncated
    return threshold

def normalize_theta_per_exp(theta_series):
    theta = theta_series.copy()
    max_val = theta.max()
    min_val = abs(theta.min())
    if max_val > 0: theta[theta >= 0] = theta[theta >= 0] / max_val
    if min_val > 0: theta[theta < 0] = theta[theta < 0] / min_val
    return theta

def format_p_value(p):
    """格式化p值为字符串，避免科学计数法"""
    if p >= 0.01:
        return f'{p:.2f}'.rstrip('0').rstrip('.')
    else:
        # 对于小数，直接用字符串表示
        s = f'{p:.10f}'.rstrip('0')
        # 去掉末尾的0和小数点
        if s.endswith('.'):
            s = s[:-1]
        return s

def process_scheme_b(df_all, k_values, k2p, exp2pos):
    """方案B：归一化 + ISC + 等量阴性"""
    df_norm = df_all.copy()
    norm_thetas = []
    for exp_id, group in df_all.groupby('exp_id'):
        norm_thetas.append(normalize_theta_per_exp(group['theta_total']))
    df_norm['theta_norm'] = pd.concat(norm_thetas)

    results = {}
    for k in k_values:
        p = k2p[k]
        p_str = format_p_value(p)
        scheme_name = f'norm_exp_p{p_str}'
        pos_thresh = isc_threshold(df_norm['theta_norm'].values, k=k, tail='right')
        positives = df_norm[df_norm['theta_norm'] >= pos_thresh]
        n_pos = len(positives)
        if n_pos == 0: continue
        negatives = df_norm.nsmallest(n_pos, 'theta_norm')
        labeled = pd.concat([positives, negatives])
        labeled['class'] = 0
        labeled.loc[labeled['theta_norm'] >= pos_thresh, 'class'] = 1
        labeled['start_pos'] = labeled['exp_id'].map(exp2pos)
        results[scheme_name] = labeled
    return results
def largest_remainder(total, weights):
    weights = np.asarray(weights, dtype=float)
    s = weights.sum()
    if total <= 0 or s <= 0:
        return [0] * len(weights)
    raw = weights / s * total
    base = np.floor(raw).astype(int)
    rem = int(total - base.sum())
    if rem > 0:
        order = np.argsort(-(raw - base), kind='stable')
        for i in range(rem):
            base[order[i % len(base)]] += 1
    return [int(x) for x in base]

def hist_counter(values):
    return Counter(int(v) for v in values)

def tvd_hist(h1, h2, n1, n2):
    keys = set(h1) | set(h2)
    s = 0.0
    for k in keys:
        s += abs(h1.get(k, 0) / max(n1, 1) - h2.get(k, 0) / max(n2, 1))
    return s / 2.0

def max_feasible_alpha(pos_hist, avail_hist):
    alphas = [avail_hist.get(L, 0) / c for L, c in pos_hist.items() if c > 0]
    return float(min(alphas)) if alphas else 0.0

def quotas_from_shape(pos_hist, n_total, avail_hist):
    lengths = sorted(L for L, c in pos_hist.items() if c > 0)
    if not lengths or n_total <= 0:
        return {}
    alloc = largest_remainder(n_total, [pos_hist[L] for L in lengths])
    quotas = {L: min(a, avail_hist.get(L, 0)) for L, a in zip(lengths, alloc)}
    total = sum(quotas.values())
    if total < n_total:
        changed = True
        while total < n_total and changed:
            changed = False
            cands = [L for L in lengths if avail_hist.get(L, 0) > quotas.get(L, 0)]
            cands.sort(key=lambda L: (-(avail_hist.get(L, 0) - quotas.get(L, 0)), L))
            for L in cands:
                if total >= n_total:
                    break
                quotas[L] = quotas.get(L, 0) + 1
                total += 1
                changed = True
    elif total > n_total:
        for L in sorted(lengths, key=lambda x: (-quotas[x], x)):
            while total > n_total and quotas[L] > 0:
                quotas[L] -= 1
                total -= 1
    return quotas

def quotas_exact_pos(pos_hist, avail_hist, n_target):
    quotas = {L: min(int(c), int(avail_hist.get(L, 0))) for L, c in pos_hist.items()}
    total = sum(quotas.values())
    if total < n_target:
        prefer = sorted(pos_hist, key=lambda L: (-(avail_hist.get(L, 0) - quotas.get(L, 0)), L))
        others = sorted(
            [L for L in avail_hist if L not in pos_hist],
            key=lambda L: (-(avail_hist[L] - quotas.get(L, 0)), L),
        )
        for L in prefer + others:
            if total >= n_target:
                break
            room = avail_hist.get(L, 0) - quotas.get(L, 0)
            if room <= 0:
                continue
            take = min(room, n_target - total)
            quotas[L] = quotas.get(L, 0) + take
            total += take
    elif total > n_target:
        for L in sorted(quotas, key=lambda x: (pos_hist.get(x, 0), -quotas[x], x)):
            while total > n_target and quotas[L] > 0:
                quotas[L] -= 1
                total -= 1
    return quotas

def draw_rows_by_len_cat(df, quotas, rng, len_col, cat_col=CAT_COL):
    """Draw rows for length quotas; within length stratify by category (row-level OK here)."""
    df = df.reset_index(drop=True)
    by_len = defaultdict(list)
    for i, L in enumerate(df[len_col].astype(int).tolist()):
        by_len[L].append(i)

    chosen = []
    stats = {}
    for L in sorted(int(k) for k in quotas):
        q = int(quotas.get(L, 0))
        pool = list(by_len.get(L, []))
        if q <= 0 or not pool:
            stats[L] = {'quota': int(quotas.get(L, 0)), 'avail': len(pool), 'drawn': 0}
            continue
        q = min(q, len(pool))
        by_cat = defaultdict(list)
        for idx in pool:
            by_cat[df.at[idx, cat_col]].append(idx)
        cats = sorted(by_cat)
        alloc = largest_remainder(q, [len(by_cat[c]) for c in cats])
        drawn = []
        for c, k in zip(cats, alloc):
            mem = list(by_cat[c])
            rng.shuffle(mem)
            drawn.extend(mem[:k])
        drawn = list(dict.fromkeys(drawn))
        if len(drawn) < q:
            rest = [i for i in pool if i not in set(drawn)]
            rng.shuffle(rest)
            drawn.extend(rest[: q - len(drawn)])
        elif len(drawn) > q:
            rng.shuffle(drawn)
            drawn = drawn[:q]
        chosen.extend(drawn)
        stats[L] = {'quota': q, 'avail': len(pool), 'drawn': len(drawn)}
    return chosen, stats

def stage1_global(neg, pos, rng):
    pos_h = hist_counter(pos[LEN_POS])
    avail_h = hist_counter(neg[LEN_NEG])
    n_pos, n_neg = len(pos), len(neg)
    alpha = max_feasible_alpha(pos_h, avail_h)
    n1 = int(min(n_neg, np.floor(alpha * n_pos + 1e-9)))
    n1 = max(n1, min(n_neg, n_pos))
    n1 = min(n1, n_neg)
    quotas = quotas_from_shape(pos_h, n1, avail_h)
    chosen, stats = draw_rows_by_len_cat(neg, quotas, rng, LEN_NEG)
    out = neg.iloc[sorted(chosen)].copy()
    oh = hist_counter(out[LEN_NEG])
    return out, {
        'n_in': n_neg,
        'n_out': int(len(out)),
        'alpha_max': round(alpha, 4),
        'n1_target': int(n1),
        'quota_by_length': {str(k): int(v) for k, v in sorted(quotas.items()) if v},
        'out_hist': {str(k): int(v) for k, v in sorted(oh.items())},
        'pos_hist': {str(k): int(v) for k, v in sorted(pos_h.items())},
        'length_TVD': round(float(tvd_hist(pos_h, oh, n_pos, len(out))), 4),
        'length_stats': {str(k): v for k, v in sorted(stats.items()) if v.get('quota')},
    }

def stage2_global(pool, pos, rng):
    n_pos, n_pool = len(pos), len(pool)
    pos_h = hist_counter(pos[LEN_POS])
    if n_pool <= n_pos:
        oh = hist_counter(pool[LEN_NEG])
        return pool.copy(), {
            'n_in': n_pool, 'n_out': n_pool, 'mode': 'no_shrink',
            'length_TVD': round(float(tvd_hist(pos_h, oh, n_pos, max(n_pool, 1))), 4),
        }
    avail_h = hist_counter(pool[LEN_NEG])
    quotas = quotas_exact_pos(pos_h, avail_h, n_pos)
    chosen, stats = draw_rows_by_len_cat(pool, quotas, rng, LEN_NEG)
    if len(chosen) < n_pos:
        rest = [i for i in range(len(pool)) if i not in set(chosen)]
        rng.shuffle(rest)
        chosen.extend(rest[: n_pos - len(chosen)])
    elif len(chosen) > n_pos:
        rng.shuffle(chosen)
        chosen = chosen[:n_pos]
    out = pool.iloc[sorted(chosen)].reset_index(drop=True)
    oh = hist_counter(out[LEN_NEG])
    cat_in = pool[CAT_COL].value_counts()
    cat_out = out[CAT_COL].value_counts()
    return out, {
        'n_in': n_pool,
        'n_out': int(len(out)),
        'mode': 'exact_pos_length_quotas',
        'quota_by_length': {str(k): int(v) for k, v in sorted(quotas.items()) if v},
        'out_hist': {str(k): int(v) for k, v in sorted(oh.items())},
        'length_TVD': round(float(tvd_hist(pos_h, oh, n_pos, len(out))), 4),
        'by_cat_in': {str(k): int(v) for k, v in cat_in.items()},
        'by_cat': {str(k): int(v) for k, v in cat_out.items()},
        'retention_by_cat': {
            str(c): float(cat_out.get(c, 0)) / float(cat_in[c]) if cat_in[c] else 0.0
            for c in cat_in.index
        },
        'length_stats': {str(k): v for k, v in sorted(stats.items()) if v.get('quota')},
    }
def assign_entities_to_row_targets(df, target_rows, seed):
    """Assign entities (groups) to splits so each split's ROW count hits target_rows.

    target_rows: {split: n_rows}. Sum(target) must equal len(df).
    Entity sizes may prevent exact hits; we greedily fill and never split entities.
    """
    rng = np.random.RandomState(seed)
    sizes = df.groupby('entity_key').size().to_dict()
    ents = list(sizes.keys())
    rng.shuffle(ents)

    # targets in order: fill largest deficits first
    got = {sp: 0 for sp in target_rows}
    assign = {}
    # sort splits by target size desc for initial fill
    # multi-pass: assign entity to split with largest remaining need if it fits,
    # else largest remaining need (may overshoot slightly) on last resort
    remaining_need = {sp: target_rows[sp] - got[sp] for sp in target_rows}

    def refresh():
        for sp in target_rows:
            remaining_need[sp] = target_rows[sp] - got[sp]

    # Pass 1: entities that fit
    for e in sorted(ents, key=lambda x: -sizes[x]):
        sz = sizes[e]
        refresh()
        fits = [(remaining_need[sp], sp) for sp in target_rows if remaining_need[sp] >= sz]
        if not fits:
            continue
        fits.sort(reverse=True)
        sp = fits[0][1]
        assign[e] = sp
        got[sp] += sz

    # Pass 2: remaining entities → split with largest remaining need
    for e in ents:
        if e in assign:
            continue
        sz = sizes[e]
        refresh()
        order = sorted(target_rows, key=lambda sp: (-remaining_need[sp], sp))
        sp = order[0]
        assign[e] = sp
        got[sp] += sz

    # Pass 3: try to improve exactness by swapping single-row entities if any mismatch
    # (optional light fix: move size-1 entities between splits toward targets)
    for _ in range(50):
        refresh()
        mismatches = [sp for sp in target_rows if remaining_need[sp] != 0]
        if not mismatches:
            break
        # find donor (excess) and receiver (deficit)
        receivers = [sp for sp in target_rows if remaining_need[sp] > 0]
        donors = [sp for sp in target_rows if remaining_need[sp] < 0]
        if not receivers or not donors:
            break
        moved = False
        for recv in receivers:
            for don in donors:
                # find size-1 entity in donor that helps receiver
                cands = [e for e, sp in assign.items()
                         if sp == don and sizes[e] == 1 and
                         (recv in target_rows)]
                # prefer if recv needs exactly 1
                if remaining_need[recv] >= 1 and remaining_need[don] <= -1:
                    if cands:
                        e = cands[0]
                        assign[e] = recv
                        got[don] -= 1
                        got[recv] += 1
                        moved = True
                        break
            if moved:
                break
        if not moved:
            break

    refresh()
    return assign, got
