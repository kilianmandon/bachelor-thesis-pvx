from pathlib import Path
import torch

def main():
    embs = [torch.load(p, weights_only=False) for p in  Path('data/attention_comparison_no_msa').glob('exp_*/embeddings.pkl')]

    keys = embs[0].keys()
    for key in keys:
        print(f'Key {key}:')
        diffs = [(emb[key]-embs[0][key]).abs().max() for emb in embs]
        print(diffs)

if __name__=='__main__':
    main()
