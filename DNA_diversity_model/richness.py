"""Exact nucleotide identity, jackknife richness and nested read subsampling."""
from pathlib import Path
from collections import Counter
import csv,gzip
import numpy as np
import pandas as pd

def read_counts(path):
    counts=Counter();empty=0
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',encoding='utf-8-sig') as f:
        for line_no,line in enumerate(f,1):
            seq=line.rstrip('\r\n').split('\t')[-1]
            if seq=='Seq':continue
            if not seq:empty+=1;continue
            if any(c not in 'ACGTNacgtn' for c in seq):
                raise ValueError(f'{path}:{line_no}: invalid nucleotide sequence')
            counts[seq]+=1
    if not counts:raise ValueError(f'No sequence records in {path}')
    return counts,empty

def read_metadata(path,data_dir):
    path=Path(path);samples=[];seen=set();used=set()
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            sid=int(row['id'])
            names=[row[k].strip() for k in ('file','file1','file2','file3') if row.get(k,'').strip()]
            if sid in seen or not names:raise ValueError('Duplicate id or missing filenames')
            seen.add(sid);files=[]
            for name in names:
                p=Path(name);p=p if p.is_absolute() else Path(data_dir)/p
                p=p.resolve()
                if p in used or not p.is_file():raise ValueError(f'Duplicate/missing file: {p}')
                used.add(p);files.append(p)
            samples.append((sid,files))
    if not samples:raise ValueError('Empty metadata')
    return samples

def jackknife(counts):
    counts=np.asarray(counts,dtype=np.int64)
    counts=counts[counts>0];R=int(counts.sum());S=len(counts);f1=int(np.sum(counts==1))
    if R==0:raise ValueError('Empty read pool')
    return dict(reads=R,S_obs=S,f1=f1,D_library=S+f1*(R-1)/R)
def nested_draws(counts,fractions,rng):
    """Return nested integer count vectors indexed by requested fraction."""
    total=int(counts.sum());current=counts.copy();draws={1.:current.copy()}
    for fraction in sorted((f for f in fractions if f<1),reverse=True):
        target=int(np.floor(total*fraction))
        current=rng.multivariate_hypergeometric(current,target)
        assert int(current.sum())==target
        draws[fraction]=current.copy()
    return draws

def analyze_samples(meta,data_dir,config,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    settings=config['read_subsampling'];rows=[];full=[]
    for sid,files in read_metadata(meta,data_dir):
        pooled=Counter();empty=0
        for path in files:
            counts,n=read_counts(path);pooled.update(counts);empty+=n
        counts=np.fromiter(pooled.values(),dtype=np.int64)
        full.append(dict(sample_id=sid,libraries=len(files),estimator='jack1',empty_rows=empty,**jackknife(counts)))
        seed=settings.get('sample_seed_overrides',{}).get(str(sid),settings['seed'])
        for rep in range(settings['replicates']):
            rng=np.random.default_rng(np.random.SeedSequence([seed,sid,rep]))
            for fraction,draw in nested_draws(counts,settings['fractions'],rng).items():
                rows.append(dict(sample_id=sid,libraries=len(files),estimator='jack1',fraction=fraction,replicate=rep,**jackknife(draw)))
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['sample_id','libraries','estimator','fraction'],as_index=False).agg(
        D_library_mean=('D_library','mean'),technical_SD=('D_library','std'),
        technical_q025=('D_library',lambda x:x.quantile(.025)),technical_q975=('D_library',lambda x:x.quantile(.975)),
        repetitions=('D_library','count'),reads=('reads','first'),observed_kinds_mean=('S_obs','mean'))
    frame.to_csv(output/'D_library_replicates.csv',index=False)
    summary.to_csv(output/'D_library_summary.csv',index=False)
    pd.DataFrame(full).to_csv(output/'full_depth_estimators.csv',index=False)
    return pd.DataFrame(full)
