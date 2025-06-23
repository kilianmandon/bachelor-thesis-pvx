import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import scienceplots

plt.style.use(['science', 'bright'])

def read_fasta(filepath: str) -> List[str]:
    """Parses a FASTA file manually and returns list of sequences."""
    sequences = []
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))

    sequences = [seq.replace('/', '') for seq in sequences]
    return sequences

def identity_to_wildtype(sequences: List[str], consider_residues: Optional[np.ndarray] = None) -> List[float]:
    """Computes identity to the wildtype sequence."""
    wildtype = sequences[0]
    wildtype = np.array(list(wildtype))

    if consider_residues is None:
        consider_residues = np.arange(len(wildtype))
    else:
        consider_residues = np.asarray(consider_residues)

    identities = []
    for seq in sequences[1:]:  # skip wildtype
        seq_arr = np.array(list(seq))
        if len(seq_arr) != len(wildtype):
            raise ValueError("Sequence length does not match wildtype.")
        match = seq_arr[consider_residues] == wildtype[consider_residues]
        identities.append(match.sum() / len(consider_residues))
    return identities

def analyze_files(file_dict: dict, consider_residues: Optional[dict] = None) -> Tuple[dict, dict]:
    """
    file_dict should be like:
    {
        "Protein1": {
            "MethodA": "path/to/file1.fasta",
            "MethodB": "path/to/file2.fasta"
        },
        "Protein2": {
            "MethodA": "path/to/file3.fasta",
            ...
        }
    }
    """
    mean_data = {}
    all_identities = {}

    for protein, methods in file_dict.items():
        mean_data[protein] = {}
        all_identities[protein] = {}

        for method, filepath in methods.items():
            sequences = read_fasta(filepath)
            mask = consider_residues[protein] if consider_residues is not None else None
            if mask is not None:
                assert all(len(seq)%mask.shape[0]==0 for seq in sequences)
                sequences = [seq[:mask.shape[0]] for seq in sequences]
            identities = identity_to_wildtype(sequences, mask)
            mean_data[protein][method] = {
                "mean": np.mean(identities),
                "std": np.std(identities)
            }
            all_identities[protein][method] = identities

    return mean_data, all_identities

def plot_results(mean_data: dict, all_identities: dict, plot_type: str = 'bar'):
    """
    plot_type: 'bar' or 'box'
    """
    proteins = list(mean_data.keys())
    methods = list(next(iter(mean_data.values())).keys())
    num_methods = len(methods)

    x = np.arange(len(proteins))
    width = 0.15

    if plot_type == 'bar':
        plt.figure(figsize=(5.91, 3.8))
        for i, method in enumerate(methods):
            means = [mean_data[protein][method]['mean'] for protein in proteins]
            stds = [mean_data[protein][method]['std'] for protein in proteins]
            plt.bar(x + i * width, means, yerr=stds, width=width, label=method, capsize=5)
        plt.xticks(x + width * (num_methods - 1) / 2, proteins)
        plt.ylabel('Mean Identity to Wildtype')
        # plt.title('Comparison of Models Across Proteins')
        plt.legend()
        plt.tight_layout()
        plt.savefig('graphics_generation/images/pmpnn_comparison.svg')
        plt.show()

    elif plot_type == 'box':
        plt.figure(figsize=(5.91, 3.8))
        positions = []
        all_data = []
        colors = plt.cm.tab10.colors
        for i, protein in enumerate(proteins):
            for j, method in enumerate(methods):
                pos = i * (num_methods + 1) + j
                positions.append(pos)
                all_data.append(all_identities[protein][method])
        labels = [f"{p}\n{m}" for p in proteins for m in methods]
        plt.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"), medianprops=dict(color="black"))
        plt.xticks(positions, labels, rotation=45, ha='right')
        plt.ylabel("Identity to Wildtype")
        plt.title("Boxplot of Sequence Identity Across Models and Proteins")
        plt.tight_layout()
        plt.show()

# Example usage (assumes paths exist):
if __name__ == "__main__":
    file_structure = dict()
    abs_base = '/Users/kilianmandon/Projects/ProteinMPNN-decoded'
    consider_residues = dict()
    for prot_name, prot in zip(['PVX', 'TMV', 'PepMV', 'BMV'], ['pvx', 'tmv', 'PepMV', 'bmv']):
        file_structure[prot_name] = {
            'P-MPNN Monomer': f'{abs_base}/data/helical_tests/out_base/{prot}_1x1.fa',
            'P-MPNN 2x2': f'{abs_base}/data/helical_tests/out_base/{prot}_2x2.fa',
            'P-MPNN 3x3': f'{abs_base}/data/helical_tests/out_base/{prot}_3x3.fa',
            'P-MPNN Sym': f'{abs_base}/data/helical_tests/out/{prot}_sym.fa',

        }

        consider_residues[prot_name] = np.ones_like(np.load(f'{abs_base}/data/helical_tests/out/{prot}_neighbours.npy'))
        # consider_residues[prot_name] = np.load(f'{abs_base}/data/helical_tests/out/{prot}_neighbours.npy')
    # optionally: specify residues to compare (e.g., np.array([10, 20, 30]))
    mean_results, identity_distributions = analyze_files(file_structure, consider_residues)

    # Choose one:
    plot_results(mean_results, identity_distributions, plot_type='bar')
    # plot_results(mean_results, identity_distributions, plot_type='box')