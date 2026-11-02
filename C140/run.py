#!/usr/bin/env python3
"""C140: score target reads, or analyze the complete joint ESSI results."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ESSI = HERE.parent / 'ESSI'
sys.path.insert(0, str(ESSI))
from xlsx_stream import write_workbook


def read_config(path):
    return json.loads(path.read_text(encoding='utf-8'))


def score(args):
    cfg = read_config(args.config)
    base = args.config.resolve().parent
    command = [sys.executable, str(ESSI / 'multi_round_pipeline.py'),
               '--meta', str(base / cfg['meta']), '--data', str(args.data.resolve()),
               '--rules', *(str(base / rule) for rule in cfg['rules']),
               '--batch', ','.join(cfg['batches']), '--file-template', cfg['file_template'],
               '--input-type', cfg['input_type'], '--reading-frame', str(cfg['reading_frame']),
               '--stop-policy', cfg['stop_policy'], '--dual-absent', cfg['dual_absent'],
               '--output', str(args.output.resolve())]
    if args.check_only:
        command.append('--check-only')
    subprocess.run(command, check=True)


def analyze(args):
    from analysis import result_columns, family_data, select_examples
    from isc import right_isc
    from figures import plot_asop, plot_examples
    cfg = read_config(args.config)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty; existing results are never overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(status='running', config=cfg, inputs={'exp1': str(args.exp1.resolve()),
                                                        'exp2': str(args.exp2.resolve())})
    manifest_path = args.output / 'run.json'
    try:
        selected, thresholds, histories, receipts = {}, [], [], []
        selection = cfg['selection']
        for batch, path in (('exp1', args.exp1), ('exp2', args.exp2)):
            print(f'Reading complete results: {batch}: {path}', flush=True)
            sequences, values = result_columns(path, selection['sheet'], selection['sequence_column'], selection['score_column'])
            result, history = right_isc(values, selection['alpha'], selection['tolerance'], selection['max_iterations'])
            selected[batch] = {s: float(z) for s, z in zip(sequences, values) if z > result['threshold']}
            thresholds.append(dict(batch=batch, complete_sequences=len(values), positive_sequences=len(selected[batch]), **result))
            histories.extend(dict(batch=batch, **h) for h in history)
            rows = [dict(label=s, Z_total=z) for s, z in selected[batch].items()]
            receipts.extend(write_workbook(args.output / f'{batch}_positive.xlsx', {'results': (rows, ['label', 'Z_total'])}))
            print(f'{batch}: threshold={result["threshold"]:.15g}, positive={len(rows)}, converged={result["converged"]}', flush=True)
            del sequences, values
        data = family_data(selected, cfg['families'])
        data['selected_examples'] = select_examples(data['families'], cfg['plot']['example_targets'])
        layout = plot_asop(data, cfg['plot'], args.output)
        coordinates, logo_sites = plot_examples(data, args.output)
        node_to_family = {row['sequence_id']: row['cluster_id'] for row in data['memberships']}
        edges = [dict(source_id=data['union'][u]['sequence_id'], target_id=data['union'][v]['sequence_id'],
                      hamming_aa=w, cluster_id=node_to_family[data['union'][u]['sequence_id']])
                 for u, v, w in data['global_mst_edges'] if w <= 1]
        families = [{k: v for k, v in family.items() if k != 'indices'} for family in data['families']]
        examples = [{k: v for k, v in family.items() if k != 'indices'} for family in data['selected_examples']]
        summary = dict(positive_union=len(data['union']), all_1aa_families=len(families),
                       families_ge_minimum=sum(f['included'] for f in families),
                       plotted_families=sum(f['plotted'] for f in families),
                       sequences_in_included_families=sum(f['total_nodes'] for f in families if f['included']))
        for category in ('exp1_only', 'exp2_only', 'overlap'):
            summary[category] = sum(row['category'] == category for row in data['union'])
        sheets = {
            'summary': ([dict(metric=k, value=v) for k, v in summary.items()], ['metric', 'value']),
            'thresholds': (thresholds, list(thresholds[0])),
            'ISC_iterations': (histories, list(histories[0])),
            'positive_union': (data['union'], list(data['union'][0])),
            'families': (families, list(families[0])),
            'memberships': (data['memberships'], list(data['memberships'][0])),
            'display_positions': (layout, ['cluster_id', 'true_x', 'true_y', 'display_x', 'display_y',
                                           'offset_x_points', 'offset_y_points', 'expanded']),
            'selected_examples': (examples, ['target_x'] + list(families[0])),
            'tree_edges': (edges, ['cluster_id', 'source_id', 'target_id', 'hamming_aa']),
            'example_tree_nodes': (coordinates, ['cluster_id', 'sequence_id', 'radius_aa', 'theta_radians',
                                                 'x', 'y', 'category', 'is_root']),
            'example_logo_sites': (logo_sites, ['cluster_id', 'position', 'amino_acid', 'frequency',
                                                'entropy_bits', 'information_bits', 'letter_height_bits'])}
        receipts.extend(write_workbook(args.output / 'c140_summary.xlsx', sheets))
        manifest.update(status='complete', summary=summary, thresholds=thresholds, xlsx_receipts=receipts)
        print(json.dumps(summary), flush=True)
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    for name in ('score', 'analyze'):
        sub = subparsers.add_parser(name)
        sub.add_argument('--config', type=Path, default=HERE / 'config/c140.json')
        sub.add_argument('--output', type=Path, required=True)
        if name == 'score':
            sub.add_argument('--data', type=Path, required=True)
            sub.add_argument('--check-only', action='store_true')
        else:
            sub.add_argument('--exp1', type=Path, required=True, help='Complete exp1 rule1 XLSX, results sheet')
            sub.add_argument('--exp2', type=Path, required=True, help='Complete exp2 rule1 XLSX, results sheet')
    args = parser.parse_args()
    (score if args.command == 'score' else analyze)(args)


if __name__ == '__main__':
    main()
