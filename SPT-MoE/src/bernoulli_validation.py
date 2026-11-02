"""Order-independent Monte Carlo attention inference of saved checkpoints.

Philox streams belong to an input fingerprint, bank and named layer, rather
than global RNG state. Training and historical sample behavior are unchanged.
"""
import hashlib
from contextlib import contextmanager

import numpy as np
import torch

from model_configs import (BHA, BERNOULLI_VALIDATION_CONFIG as VC, MAX_L_EVAL,
                           SEED, BERNOULLI_EXECUTION_CONFIG as EC)


def fingerprint(main, tokens):
    h = hashlib.sha256()
    for array in (main, tokens):
        h.update(str(array.shape).encode())
        h.update(np.asarray(array, dtype='<f4').tobytes())
    return h.hexdigest()


def uniforms(key, bank, layer, heads, length, start, count):
    seed = int.from_bytes(hashlib.sha256(f'{SEED}|{key}|{bank}|{layer}'.encode()).digest()[:16], 'little')
    coordinates = heads*length*length
    stride = ((coordinates+7)//8)*8
    # A Philox block yields eight float32 uniforms. Each draw gets a padded
    # whole-block range, so draw chunks and prefix length do not change values.
    generator = np.random.Generator(np.random.Philox(seed=seed, counter=start*(stride//8)))
    return generator.random((count, stride), dtype=np.float32)[:, :coordinates].reshape(count, heads, length, length)


@contextmanager
def mc_context(model, cache, start, stop):
    if model.training:
        raise RuntimeError('MC evaluation requires model.eval(); ordinary dropout must be off.')
    previous = {}
    for name, module in model.named_modules():
        if isinstance(module, BHA):
            previous[name] = module.eval_attention_mode
            module.eval_attention_mode = 'mc'
            u = cache[name][start:stop]
            module._mc_uniform = u.reshape(-1, *u.shape[2:])
    try:
        yield
    finally:
        for name, module in model.named_modules():
            if isinstance(module, BHA):
                module.eval_attention_mode = previous[name]
                del module._mc_uniform


def predict_mc(model, main, supp, mask, keys, bank, draws, device, sequence_batch=None, draw_batch=None):
    model.eval()
    sequence_batch = sequence_batch or VC['sequence_batch_size']
    draw_batch = draw_batch or VC['draw_batch_size']
    margins = np.empty((draws, len(keys)), dtype=np.float32)
    probabilities = np.empty_like(margins)
    for first in range(0, len(keys), sequence_batch):
        last = min(first+sequence_batch, len(keys)); n = last-first
        xb = torch.from_numpy(main[first:last]).to(device)
        sb = torch.from_numpy(supp[first:last]).to(device)
        mb = torch.from_numpy(mask[first:last]).to(device)
        for cache_start in range(0, draws, EC['cache_draw_count']):
            cached_count=min(EC['cache_draw_count'],draws-cache_start)
            cache = {}
            for name, module in model.named_modules():
                if isinstance(module, BHA):
                    length = MAX_L_EVAL if name.startswith('supp_encoder.') else 1
                    u = np.stack([uniforms(key, bank, name, module.nh, length, cache_start, cached_count)
                                  for key in keys[first:last]], axis=1)
                    cache[name] = torch.from_numpy(u).to(device)
            with torch.inference_mode():
                for local_begin in range(0, cached_count, draw_batch):
                    local_end=min(local_begin+draw_batch,cached_count);k=local_end-local_begin
                    begin=cache_start+local_begin;end=cache_start+local_end
                    with mc_context(model, cache, local_begin, local_end):
                        if model.supp_ind:
                            logits=model(xb.repeat(k,1),sb.repeat(k,1,1),mb.repeat(k,1))
                        else:
                            logits=model(xb.repeat(k,1))
                    margins[begin:end,first:last]=(logits[:,1]-logits[:,0]).reshape(k,n).cpu().numpy()
                    probabilities[begin:end,first:last]=torch.softmax(logits,1)[:,1].reshape(k,n).cpu().numpy()
    return margins, probabilities
