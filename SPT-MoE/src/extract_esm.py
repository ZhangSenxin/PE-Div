"""ESM2 3B layer33, one sequence per forward pass; actual residues only."""
import argparse, hashlib
from pathlib import Path
import numpy as np
from Bio import SeqIO

def main():
    p=argparse.ArgumentParser();p.add_argument('--fasta',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    import torch,esm
    from model_configs import prepare_deterministic_inference,set_seed
    prepare_deterministic_inference();set_seed(42)
    if a.output.exists():raise FileExistsError(a.output)
    model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(a.checkpoint));model.eval().to(a.device);convert=alphabet.get_batch_converter()
    if model.embed_dim!=2560:raise ValueError('Expected ESM2 3B, 2560 dimensions')
    a.output.mkdir(parents=True);seen=set()
    with torch.inference_mode():
        for record in SeqIO.parse(a.fasta,'fasta'):
            key=record.id;seq=str(record.seq)
            if key in seen:raise ValueError('Duplicate or unsafe FASTA ID')
            if not 0<len(seq)<=34:raise ValueError('Provide signal-only sequences (DMS first20 aa)')
            seen.add(key);_,_,tok=convert([(key,seq)])
            z=model(tok.to(a.device),repr_layers=[33],return_contacts=False)['representations'][33][0,1:len(seq)+1].float().cpu().numpy()
            np.save(a.output/(hashlib.sha256(key.encode()).hexdigest()+'.npy'),z)
if __name__=='__main__':main()

