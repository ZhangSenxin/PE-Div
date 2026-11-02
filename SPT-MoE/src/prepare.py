"""Small, explicit data preparation functions for the Methods protocol."""
import argparse, json, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd


from data_methods import stage1_global,stage2_global

def truncate_sequence(seq,length_range=(8,30),rng=None):
    rng=rng or random
    lo,hi=length_range
    if len(seq)<lo:return None
    return seq[:rng.randint(lo,min(hi,len(seq)))]

def truncate_fasta(root,output):
    """User-supplied historical algorithm; explicit sorted files for portability."""
    from Bio import SeqIO
    rng=random.Random(42);rows=[];skipped=0
    for folder in ['T1','T2']:
        for file in sorted((Path(root)/folder).iterdir()):
            if not file.is_file() or file.suffix.lower() not in ['.fa','.fasta','.faa','.fas']:continue
            records=list(SeqIO.parse(file,'fasta'))
            category=file.name.split('uniprotkb_')[-1].split('keyword_')[-1].split('_revi')[0]
            for record in records:
                seq=str(record.seq);cut=truncate_sequence(seq,rng=rng)
                if cut is None:skipped+=1;continue
                rows.append(dict(sample_id=f'NEG|{folder}|{category}|{len(records)}|{record.id}',level_0=folder,level_1=category,sequence=seq,trunc_seq=cut,trunc_len=len(cut)))
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);pd.DataFrame(rows).to_csv(output/'negative_fragments.csv',index=False)
    (output/'negative_fragments.fasta').write_text(''.join(f'>{r["sample_id"]}\n{r["trunc_seq"]}\n' for r in rows))
    (output/'audit.json').write_text(json.dumps(dict(seed=42,kept=len(rows),short_sequences_removed=skipped,file_order='sorted; record order unchanged'),indent=2))

def select_husp_positives(annotations,library,excluded_ids=()):
    """Inputs: reviewed human SIGNAL peptides; coding library ID and Seq columns."""
    from Bio.Seq import Seq
    lib=library.copy();lib['translated']=[str(Seq(str(x).upper().replace('U','T')).translate()).removesuffix('*') for x in lib.Seq]
    p=annotations.drop_duplicates('ID').merge(lib[['ID','Seq','translated']],on='ID')
    p=p[(p.sp_seq==p.translated)&p.Seq.str.upper().str.startswith('ATG')&(p.Seq.str.len()%3==0)].copy()
    p=p.sort_values(['sp_seq','ID']).drop_duplicates('sp_seq');p=p[~p.ID.isin(excluded_ids)].copy()
    p['protein']=p.sp_seq;p['fa_len']=p.sp_seq.str.len();p['acc']=p.ID;p['entity_key']='P:'+p.sp_seq
    return p

def match_husp_negatives(positive,negative):
    negative=negative[~negative.acc.isin(positive.acc)&~negative.trunc_seq.isin(positive.protein)].copy()
    rng=np.random.RandomState(42);s1,a=stage1_global(negative,positive,rng);s2,b=stage2_global(s1,positive,rng)
    return s2,dict(stage1=a,stage2=b)

def feature_arrays(sequences,token_features):
    """ESM token arrays exclude BOS/EOS; only mean/max channels are consumed."""
    tokens=np.zeros((len(sequences),34,8),np.float32);mask=np.zeros((len(sequences),34),bool);main=[]
    for i,(seq,z) in enumerate(zip(sequences,token_features)):
        z=np.asarray(z,np.float32)
        if z.shape!=(len(seq),2560) or not 0<len(seq)<=34:raise ValueError('Sequence/ESM length mismatch')
        main.append(z.mean(0));tokens[i,:len(seq),0]=z.mean(1);tokens[i,:len(seq),1]=z.max(1);mask[i,:len(seq)]=True
    return dict(esm=np.asarray(main),tokens=tokens,mask=mask)

def pack(manifest,feature_dir,output):
    """Manifest supplies published member order and groups; never resplit here."""
    f=pd.read_csv(manifest);seq=f.input_sequence.astype(str).tolist()
    raw=feature_arrays(seq,[np.load(Path(feature_dir)/(hashlib.sha256(str(i).encode()).hexdigest()+'.npy')) for i in f.sample_id]);raw['ids']=f.sample_id.astype(str).to_numpy()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for split,g in f.groupby('split',sort=False):
        ix=g.index.to_numpy();g.to_csv(output/(split+'.csv'),index=False);np.savez(output/(split+'.npz'),**{k:v[ix] for k,v in raw.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    t=s.add_parser('truncate');t.add_argument('--root',required=True);t.add_argument('--output',required=True)
    t=s.add_parser('pack');t.add_argument('--manifest',required=True);t.add_argument('--features',required=True);t.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='truncate':truncate_fasta(a.root,a.output)
    else:pack(a.manifest,a.features,a.output)


