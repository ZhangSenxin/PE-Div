"""Cache fixed validation uniforms/tensors without changing MC computations."""
import numpy as np
import torch
from bernoulli_validation import uniforms, mc_context
from model_configs import BHA, BERNOULLI_VALIDATION_CONFIG as C, BERNOULLI_EXECUTION_CONFIG as EC, MAX_L_EVAL


class CachedValidationMC:
    def __init__(self,model,inputs,bank,draws,device='cuda'):
        main,supp,mask,keys=inputs
        self.draws=draws;self.n=len(keys);self.batches=[]
        for first in range(0,len(keys),C['sequence_batch_size']):
            last=min(first+C['sequence_batch_size'],len(keys));n=last-first
            xb=torch.from_numpy(main[first:last]).to(device)
            sb=torch.from_numpy(supp[first:last]).to(device)
            mb=torch.from_numpy(mask[first:last]).to(device)
            for cache_start in range(0,draws,EC['cache_draw_count']):
                count=min(EC['cache_draw_count'],draws-cache_start);cache={}
                for name,module in model.named_modules():
                    if isinstance(module,BHA):
                        length=MAX_L_EVAL if name.startswith('supp_encoder.') else 1
                        u=np.stack([uniforms(key,bank,name,module.nh,length,cache_start,count) for key in keys[first:last]],axis=1)
                        cache[name]=torch.from_numpy(u).to(device)
                chunks=[]
                for begin in range(0,count,C['draw_batch_size']):
                    end=min(begin+C['draw_batch_size'],count);k=end-begin
                    chunks.append((begin,end,xb.repeat(k,1),sb.repeat(k,1,1),mb.repeat(k,1)))
                self.batches.append((first,last,cache_start,cache,chunks))

    def predict(self,model):
        model.eval();margins=np.empty((self.draws,self.n),dtype=np.float32)
        with torch.inference_mode():
            for first,last,start,cache,chunks in self.batches:
                n=last-first
                for begin,end,xb,sb,mb in chunks:
                    with mc_context(model,cache,begin,end):
                        logits=model(xb,sb,mb) if model.supp_ind else model(xb)
                    margins[start+begin:start+end,first:last]=(logits[:,1]-logits[:,0]).reshape(end-begin,n).cpu().numpy()
        return margins
