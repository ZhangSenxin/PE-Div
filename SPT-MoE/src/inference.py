import numpy as np
import torch
from cached_dms_mc import CachedValidationMC
from bernoulli_validation import fingerprint, mc_context

SUPP_VARIANTS={"meanmax":[0,1]}
COLS=["dms_score","dpf_score","regression_score","unified_score"]
def dl_inputs(raw, scaler, cfg):
    main = scaler.transform(raw[cfg['feature']]).astype(np.float32)
    channels = SUPP_VARIANTS[cfg['variant']] if cfg['variant'] else []
    tokens = raw['tokens'][:, :, channels].copy()
    if channels:
        keys = [fingerprint(x, s[m]) for x, s, m in zip(raw[cfg['feature']], tokens, raw['mask'])]
    else:
        keys = [fingerprint(x, np.empty((0, 0), dtype=np.float32)) for x in raw[cfg['feature']]]
    return main, tokens, raw['mask'], keys
class TaskMC(CachedValidationMC):
    def predict(self, model):
        model.eval()
        out = np.empty((self.draws, self.n, len(COLS)), np.float32)
        with torch.inference_mode():
            for first, last, start, cache, chunks in self.batches:
                for begin, end, xb, sb, mb in chunks:
                    with mc_context(model, cache, begin, end):
                        value = model.forward_task_margins(xb, sb, mb)
                    out[start + begin:start + end, first:last] = value.reshape(end - begin, last - first, len(COLS)).cpu().numpy()
        return out
def percentile(ref,values):
    ref=np.sort(ref)
    return (np.searchsorted(ref,values,'left')+np.searchsorted(ref,values,'right')+1)/(2*(len(ref)+1))

def pooled_fusion(dms_reference, husp_reference, predictions):
    """Each reference is grouped by unique sequence before concatenation."""
    refs=np.concatenate([dms_reference,husp_reference],axis=0)
    if refs.ndim!=2 or refs.shape[1]!=2 or not np.isfinite(refs).all():
        raise ValueError('Reference must contain two finite margin columns')
    return (percentile(refs[:,0],predictions[:,0])+percentile(refs[:,1],predictions[:,1]))/2
