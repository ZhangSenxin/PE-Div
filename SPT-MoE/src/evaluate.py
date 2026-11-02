"""Frozen prediction and external evaluation; no parameter fitting to NLUC."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
from model_configs import Phase6TaskMoE,prepare_deterministic_inference
from inference import TaskMC,dl_inputs,pooled_fusion

def correlations(frame,target='activity',score='score'):
    return dict(n=len(frame),pearson=float(pearsonr(frame[target],frame[score]).statistic),spearman=float(spearmanr(frame[target],frame[score]).statistic))

def merge_external(frame,alignment_factor):
    """Input excludes positive display controls and specified >0.9 REF members."""
    f=frame.copy()
    # REF is the reference scale; factor multiplies Recom activity.
    f['aligned_activity']=np.where(f.batch=='Recom',f.activity*alignment_factor,f.activity)
    numeric=['activity','aligned_activity','score','signalp_sp']
    # Only Recom duplicate sequences are merged; never merge across batches.
    f['unit']=np.where(f.batch=='Recom',f.input_sequence,f.sample_id)
    return f.groupby(['batch','unit'],sort=False,as_index=False)[numeric].mean()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--sampling-group',choices=['primary','independent'],default='primary');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    prepare_deterministic_inference();c=json.loads((a.model/'config.json').read_text());m=Phase6TaskMoE(c['model'],'T1').to(a.device);m.load_state_dict(torch.load(a.model/'T1.pth',map_location=a.device,weights_only=True))
    with (a.model/'scaler.pkl').open('rb') as f:scaler=pickle.load(f)
    raw=dict(np.load(a.features));f=pd.read_csv(a.metadata)
    if not np.array_equal(raw['ids'].astype(str),f.sample_id.astype(str)):raise ValueError('ID/order mismatch')
    pred=TaskMC(m,dl_inputs(raw,scaler,c['model']),a.sampling_group,c['mc_draws'],device=a.device).predict(m).mean(0,dtype=np.float64)[:,:2]
    ref=np.load(a.model/'pooled_reference.npy');f['dms_score']=pred[:,0];f['husp_score']=pred[:,1];f['score']=pooled_fusion(ref,np.empty((0,2)),pred)
    f.to_csv(a.output,index=False)
if __name__=='__main__':main()
