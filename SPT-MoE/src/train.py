"""Final P -> T1 recipe; no model search and no external activity access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, copy, json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from model_configs import Phase6TaskMoE, set_seed, prepare_deterministic_inference
from inference import dl_inputs, TaskMC

def load(root, domain):
    root=Path(root)/domain
    frame=pd.read_csv(root/'train.csv')
    raw=dict(np.load(root/'train.npz'))
    if len(frame)!=len(raw['esm']): raise ValueError('Row count mismatch')
    if 'ids' not in raw or not np.array_equal(frame.sample_id.astype(str),raw['ids'].astype(str)):
        raise ValueError('NPZ ids must match CSV sample_id in exact row order')
    if raw['esm'].shape[1]!=2560 or raw['tokens'].shape[1:]!=(34,8):
        raise ValueError('Expected esm (N,2560), tokens (N,34,8)')
    if raw['mask'].shape!=raw['tokens'].shape[:2]:raise ValueError('Mask mismatch')
    if not all(np.isfinite(raw[k]).all() for k in ['esm','tokens']):raise ValueError('Nonfinite features')
    return frame,raw

def weights(frame):
    return (1/frame.groupby('group_id').group_id.transform('size')).to_numpy()

def fit_scaler(frames,raws):
    ws=[weights(frames[d]) for d in frames]
    return StandardScaler().fit(np.concatenate([raws[d]['esm'] for d in frames]),sample_weight=np.concatenate([w/w.sum() for w in ws]))

def train_stage(cfg, settings, role, frames, inputs, device, source=None, validation=None):
    """Optional training-only fold validation; historical val/test must not be passed."""
    set_seed(settings['seed'])
    model=Phase6TaskMoE(cfg,role).to(device)
    if source is not None:
        model.load_state_dict(source)
        model.target_head.load_state_dict(model.fc_out.state_dict())
    partial=('gating.','ffn.',f"experts.{cfg['num_experts']-1}.")
    for name,p in model.named_parameters():
        p.requires_grad_(not name.startswith('target_head.') if role=='P' else name.startswith(('target_head.',)+partial))
    frozen=copy.deepcopy(model.fc_out.state_dict())
    # The original optimizer contains a second, frozen trunk group at 0.02x LR.
    if role=='P':opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    else:
        groups=[[],[]]
        for name,p in model.named_parameters():
            if not name.startswith('fc_out.'):
                groups[0 if name.startswith(('target_head.',)+partial) else 1].append(p)
        opt=torch.optim.AdamW([{'params':groups[0],'lr':cfg['lr']*settings['lr_multiplier']},{'params':groups[1],'lr':cfg['lr']*.02}],weight_decay=cfg['weight_decay'])
    set_seed(settings['seed']);rng=np.random.default_rng(settings['seed']);gen=torch.Generator().manual_seed(settings['seed'])
    domain='dpf' if role=='P' else 'dms';col=1 if role=='P' else 0
    labels={d:torch.tensor(f['class'].to_numpy(),dtype=torch.float32,device=device) for d,f in frames.items()}
    ws={d:weights(f) for d,f in frames.items()};wt=torch.tensor(ws[domain]/ws[domain].mean(),dtype=torch.float32,device=device)
    cap=settings['source_schedule_epochs' if role=='P' else 'transfer_schedule_epochs']
    epochs=cap if validation is not None else settings['source_epochs' if role=='P' else 'transfer_epochs']
    caches={d:TaskMC(model,v,'primary',settings['mc_draws'],device=device) for d,(f,v) in (validation or {}).items()}
    initial=[p['lr'] for p in opt.param_groups];history=[];best=None;best_state=None;wait=0;effective=epochs
    def forward(d,ix):
        x,s,m,_=inputs[d]
        return model.forward_task_margins(*[torch.from_numpy(a[ix]).to(device) for a in [x,s,m]])
    for epoch in range(1,epochs+1):
        model.train()
        if role=='T1':
            for name,module in model.named_modules():
                if isinstance(module,torch.nn.Dropout) and not name.startswith(partial):module.eval()
        order=torch.randperm(len(frames[domain]),generator=gen).numpy();total=0.
        for first in range(0,len(order),cfg['batch_size']):
            ix=order[first:first+cfg['batch_size']];opt.zero_grad(set_to_none=True)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(forward(domain,ix)[:,col],labels[domain][ix],reduction='none')*wt[ix]).mean()
            if role=='T1':
                pi=rng.choice(len(frames['dpf']),len(ix),replace=True,p=ws['dpf']/ws['dpf'].sum())
                loss=loss+settings['replay_weight']*torch.nn.functional.binary_cross_entropy_with_logits(forward('dpf',pi)[:,1],labels['dpf'][pi])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),settings['grad_clip'],error_if_nonfinite=True);opt.step();total+=loss.item()*len(ix)
        factor=settings['cosine_floor']+(1-settings['cosine_floor'])*(1+np.cos(np.pi*epoch/cap))/2
        for pg,lr in zip(opt.param_groups,initial):pg['lr']=lr*factor
        row={'epoch':epoch,'loss':total/len(order)}
        if validation is not None and epoch>=(1 if role=='P' else 16):
            auc={d:roc_auc_score(validation[d][0]['class'].groupby(validation[d][0].group_id).first(),pd.Series(mc.predict(model).mean(0,dtype=np.float64)[:,1 if d=='dpf' else 0],index=validation[d][0].index).groupby(validation[d][0].group_id).mean()) for d,mc in caches.items()}
            rank=(auc['dpf'],) if role=='P' else ((1,auc['dms'],auc['dpf']) if auc['dpf']>=.95 else (0,auc['dpf'],auc['dms']))
            row['auc']=auc
            if best is None or rank>best:best=rank;best_state=copy.deepcopy(model.state_dict());effective=epoch;wait=0
            else:wait+=1
        history.append(row)
        if validation is not None and wait>=(10 if role=='P' else 8):break
    if validation is not None:model.load_state_dict(best_state)
    if role=='T1':assert all(torch.equal(v,model.fc_out.state_dict()[k]) for k,v in frozen.items())
    return model,history,effective

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path(__file__).parents[1]/'config.json');p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    settings=json.loads(a.config.read_text());cfg=settings['model'];prepare_deterministic_inference();torch.set_num_threads(4)
    frames={};raws={}
    for d in ['dms','dpf']:frames[d],raws[d]=load(a.data,d)
    scaler=fit_scaler(frames,raws);inputs={d:dl_inputs(raws[d],scaler,cfg) for d in frames}
    a.output.mkdir(parents=True)
    source=None
    for role in ['P','T1']:
        model,h,n=train_stage(cfg,settings,role,frames,inputs,a.device,source)
        source=copy.deepcopy(model.state_dict());torch.save(source,a.output/(role+'.pth'))
        (a.output/(role+'_history.json')).write_text(json.dumps(h,indent=2))
    with (a.output/'scaler.pkl').open('wb') as f:pickle.dump(scaler,f)
    refs=[]
    for d in frames:
        prediction=TaskMC(model,inputs[d],'primary',settings['mc_draws'],device=a.device).predict(model).mean(0,dtype=np.float64)[:,:2]
        ref=pd.DataFrame(prediction,columns=['dms_score','dpf_score']);ref['group_id']=frames[d].group_id.to_numpy();ref=ref.groupby('group_id',sort=True)[['dms_score','dpf_score']].mean();refs.append(ref.to_numpy())
    np.save(a.output/'pooled_reference.npy',np.concatenate(refs));(a.output/'config.json').write_text(json.dumps(settings,indent=2))
if __name__=='__main__':main()
