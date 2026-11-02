import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent/'src'))
from model_configs import Phase6TaskMoE,prepare_deterministic_inference,set_seed
from prepare import truncate_sequence,feature_arrays
from data_methods import isc_threshold
from inference import TaskMC,dl_inputs,percentile,pooled_fusion
from train import fit_scaler
import pandas as pd

def main():
    torch.set_num_threads(1);prepare_deterministic_inference();set_seed(42)
    c=json.loads((Path(__file__).parent/'config.json').read_text());cfg=c['model']
    assert truncate_sequence('AAA') is None
    import random
    assert truncate_sequence('M'*100,rng=random.Random(42))==truncate_sequence('M'*100,rng=random.Random(42))
    assert np.isfinite(isc_threshold(np.array([0.,1.,2.,100.]),k=2))
    rng=np.random.default_rng(42);raw=feature_arrays(['M'*20],[rng.normal(size=(20,2560)).astype(np.float32)])
    f=pd.DataFrame({'group_id':['a']});sc=fit_scaler({'dms':f,'dpf':f},{'dms':raw,'dpf':raw})
    m=Phase6TaskMoE(cfg,'T1');inputs=dl_inputs(raw,sc,cfg)
    mc=TaskMC(m,inputs,'primary',2,device='cpu');a=mc.predict(m);b=mc.predict(m);np.testing.assert_array_equal(a,b)
    m.train();x=torch.from_numpy(inputs[0]);s=torch.from_numpy(inputs[1]);mask=torch.from_numpy(inputs[2]);m.forward_task_margins(x,s,mask)[:,0].sum().backward()
    assert m.target_head.weight.grad is not None
    np.testing.assert_allclose(percentile(np.array([0.,0.,1.]),np.array([0.])),[.375])
    refs=np.array([[0.,0.],[1.,1.]])
    np.testing.assert_allclose(pooled_fusion(refs,refs,refs),[.3,.7])
    print('PASS: imports, truncation, ISC, features, repeat MC, gradient, percentile ties and fusion')
if __name__=='__main__':main()
