"""Portable entry point. Run --help; importing modules does not run analyses."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from capture_model import CaptureModel
from simulation import Simulator
from richness import analyze_samples
from inference import estimate_samples

BASE=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=BASE/'config.json')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('richness',help='Count/merge exact sequences and run nested read subsampling')
    p.add_argument('--meta',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p=sub.add_parser('curves',help='Analytical Dirichlet mean curves')
    p.add_argument('--points',type=int,default=501)
    p=sub.add_parser('simulate',help='Dirichlet prediction grid or explicit local nodes')
    p.add_argument('--nodes',type=Path,help='CSV with D_cells,target_gini; use for local refinement')
    p.add_argument('--keep-replicates',action='store_true')
    p=sub.add_parser('sensitivity',help='Raw-Gini-constrained lognormal mean control only')
    p.add_argument('--keep-replicates',action='store_true')
    p=sub.add_parser('infer',help='Infer D_cells from full-depth jack1 library estimates')
    p.add_argument('--library-table',type=Path,required=True)
    p.add_argument('--predictions',type=Path,nargs='*',default=[],help='Base grid then local refinement CSVs; later duplicate nodes win')
    for p in sub.choices.values():p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text(encoding='utf8'))
    budget=CaptureModel(cfg);s=cfg['simulation'];out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    # Record settings and inputs, not copies of data.
    manifest=dict(command=args.command,status='running',config=cfg,arguments={k:str(v) for k,v in vars(args).items()},
                  versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__))
    (out/'run_config.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    if args.command=='richness':analyze_samples(args.meta,args.data_dir,cfg,out)
    elif args.command=='curves':
        if args.points<2:parser.error('--points must be >=2')
        rows=[]
        for G in s['ginis']:
            lo=budget.min_D(G)
            if lo>budget.Ns:continue
            grid=np.unique(np.r_[np.geomspace(lo,min(1000,budget.Ns),80),np.linspace(min(1000,budget.Ns),budget.Ns,args.points)])
            for D in grid:rows.append(dict(target_gini=G,D_cells=D,expected_D_library=budget.expected(D,G),M=budget.M,beta=budget.beta,p=budget.p,eta=budget.eta))
        pd.DataFrame(rows).to_csv(out/'mean_curves.csv',index=False)
    elif args.command in ('simulate','sensitivity'):
        sim=Simulator(budget,cfg,out,args.keep_replicates);rows=[]
        if args.command=='simulate':
            if args.nodes:nodes=pd.read_csv(args.nodes)[['D_cells','target_gini']].to_records(index=False)
            else:
                grid=sorted(set(np.rint(np.linspace(s['grid_start'],budget.Ns,s['grid_points'])).astype(int))|set(s['key_positions']))
                nodes=[(D,G) for G in s['ginis'] for D in grid if budget.min_D(G)<=D<=budget.Ns]
            for D,G in nodes:
                R=s['key_replicates'] if D in s['key_positions'] else s['replicates']
                result=sim.draws(D,G,s['bins'],R,chunk=s['chunk'])
                rows.append(sim.summarize(D,G,s['bins'],R,result))
                pd.DataFrame(rows).to_csv(out/'prediction_intervals.csv',index=False)
                print(f'Dirichlet G={G:g}, D={D:g}, R={R}',flush=True)
        else:
            for G in s['lognormal_ginis']:
                for D in s['lognormal_positions']:
                    if D>budget.Ns:continue
                    sigma,initial,calibrated=sim.calibrate_lognormal(D,G,s['bins'])
                    result=sim.draws(D,G,s['bins'],s['replicates'],family='lognormal',sigma=sigma,chunk=s['chunk'])
                    rows.append(dict(D_cells=D,target_gini=G,sigma=sigma,calibration_gini_mean=calibrated,
                        lognormal_expected_D_library=float(result['conditional_mean'].mean()),
                        dirichlet_expected_D_library=budget.expected(D,G),replicates=s['replicates'],M=budget.M,beta=budget.beta))
                    pd.DataFrame(rows).to_csv(out/'lognormal_mean_sensitivity.csv',index=False)
    else:
        pred=pd.concat([pd.read_csv(p) for p in args.predictions],ignore_index=True) if args.predictions else None
        detail,summary=estimate_samples(pd.read_csv(args.library_table),budget,s['ginis'],pred)
        detail.to_csv(out/'sample_D_cells_by_gini_with_intervals.csv',index=False)
        summary.to_csv(out/'D_cells_ranges.csv',index=False)
    manifest['status']='completed'
    (out/'run_config.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')

if __name__=='__main__':main()
