from pathlib import Path
from Bio import PDB
import numpy as np
import itertools
import scienceplots

import tqdm

import tmscoring
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde



def plot_cdf(datasets, method='kde', bins=30, ax=None, labels=None, for_powerpoint=False, colors=None):
    """
    Plot CDFs for multiple datasets.

    Parameters:
    - datasets: list of array-like, each a list of continuous values
    - method: 'kde' or 'hist'
    - bins: number of bins (used only if method='hist')
    - ax: matplotlib axis (optional)
    - labels: list of labels for each dataset

    Returns:
    - ax: the axis with the CDF plots
    """
    if ax is None:
        if for_powerpoint:
            fontsize = 10
            figsize = (7, 5)
            plt.rcParams.update({
                "font.family": "sans-serif",
                "font.sans-serif": ["Arial"],
                "font.size": fontsize,
            })
        else:
            plt.style.use(['science'])
            # fontsize=12
            #plt.rcParams.update({
                #"font.size": fontsize,
            #})
            figsize = (5.91, 3.8)
        # plt.figure(figsize=figsize)
        fig, ax = plt.subplots(figsize=figsize)

    if labels is None:
        labels = [''] * len(datasets)

    for i, (data, label) in enumerate(zip(datasets, labels)):
        if method == 'kde':
            kde = gaussian_kde(data, bw_method=0.1)
            x_vals = np.linspace(min(data), max(data), 1000)
            pdf = kde(x_vals)
            cdf = np.cumsum(pdf)
            cdf /= cdf[-1]
            if colors is not None:
                ax.plot(x_vals, cdf, label=label, color=colors[i])
            else:
                ax.plot(x_vals, cdf, label=label)
        elif method == 'hist':
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=label)
        else:
            raise ValueError("Method must be 'kde' or 'hist'.")

    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=0)
    if for_powerpoint:
        ax.set_xlabel('RMSD (Å)')
    else:
        ax.set_xlabel('RMSD (\AA)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(False)
    ax.legend(loc='lower right')
    return ax


def rmsd_score(pdb_file, pdb_test):
    pdb_parser = PDB.PDBParser(QUIET=True)
    cif_parser = PDB.MMCIFParser(QUIET=True)

    if pdb_file.suffix == '.cif':
        struct1 = cif_parser.get_structure('A', pdb_file)
    elif pdb_file.suffix == '.pdb':
        struct1 = pdb_parser.get_structure('A', pdb_file)

    if pdb_test.suffix == '.cif':
        struct2 = cif_parser.get_structure('A', pdb_test)
    elif pdb_test.suffix == '.pdb':
        struct2 = pdb_parser.get_structure('A', pdb_test)

    chain_ids_1 = [c.get_id() for c in struct1.get_chains()]
    chain_ids_2 = [c.get_id() for c in struct2.get_chains()]
    
    chains_1 = {c.get_id(): c for c in struct1.get_chains()}
    chains_2 = {c.get_id(): c for c in struct2.get_chains()}

    coords1 = np.concatenate([np.stack([r['CA'].get_coord() for r in c.get_residues()], axis=0) for c in chains_1.values()], axis=0)
    all_scores = []

    for perm in itertools.permutations(chain_ids_2):
        coords2 = np.concatenate([np.stack([r['CA'].get_coord() for r in chains_2[c_id].get_residues()], axis=0) for c_id in perm], axis=0)
        R, t = tmscoring.compute_alignment(coords2, coords1)
        coords2_aligned = (coords2 - np.mean(coords2, axis=0)) @ R.T + t + np.mean(coords2, axis=0)
        score = np.linalg.norm(coords2_aligned - coords1, axis=-1).max()
        all_scores.append(score)
    
    return np.min(all_scores)


def data_gen():
    sub_paths = [
        Path('data/attention_comparison'),
        Path('data/attention_comparison_no_msa'),
        Path('data/attention_comparison_no_msa_sym'),
        Path('data/attention_comparison_no_msa_sym_track'),
    ]

    names = ['base', 'no_msa', 'no_msa_sym', 'no_msa_sym_track']

    for name, base_path in zip(names, sub_paths):
        pred_files = list(base_path.glob('exp_*/prediction/wildtype_no_symmetry.cif'))
        # for i in range(1, 6):
            # print(f'------------ Model {i} -----------')
            # pred_files = list(Path('/home/rc113013/bachelor/openfold/kilian/wildtype_batch_no_msa/predictions').glob(f'*_model_{i}_multimer_v3_unrelaxed.pdb'))
        pred_files = sorted(pred_files, key=lambda p: str(p))

        test_files = list(Path('trimer_configurations').glob('*.cif'))
        test_file_names = [tf.stem for tf in test_files]

        all_scores = []
        for pred_file in tqdm.tqdm(pred_files):
            scores = np.array([rmsd_score(pred_file, test_file) for test_file in test_files])
            all_scores.append(scores)
            i = np.argmin(scores)
            # print(f'{pred_file.parent.parent.name}: {test_file_names[i]:>6} | Score {scores[i]:.2f}')

        all_scores = np.stack(all_scores, axis=0)
        np.save(f'{name}_scores.npy', all_scores)

def do_plot(file_name, multimer=True, for_powerpoint=False, subset_names=None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    names = ['base', 'sym', 'sym_track', 'no_msa', 'no_msa_sym', 'no_msa_sym_track', 'af2_msa_init_guess']
    if for_powerpoint:
        labels = [
            r'AlphaFold 3',
            r'AlphaFold 3 + Sym',
            r'AlphaFold 3 + Sym+Ori',
            r'$-\text{MSA}$',
            r'$-\text{MSA} + \text{Sym}$',
            r'$-\text{MSA} + \text{Sym+Ori}$',
            r'AlphaFold 2'
        ]
    else:
        labels = [
            r'$+\text{MSA}$',
            r'$+\text{MSA} + \text{Sym}$',
            r'$+\text{MSA} + \text{Sym+Ori}$',
            r'$-\text{MSA}$',
            r'$-\text{MSA} + \text{Sym}$',
            r'$-\text{MSA} + \text{Sym+Ori}$',
            r'AlphaFold 2'
        ]

    if subset_names is None:
        subset_names = names
    



    test_files = list(Path('graphics_generation/af3/prediction_accuracy/trimer_configurations').glob('*.cif'))
    test_file_names = [tf.stem for tf in test_files]

    all_data = []
    all_labels = [labels[names.index(name)] for name in subset_names]
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [colors[names.index(name)] for name in subset_names]

    for name in subset_names:
        print(f'{name}:')
        if multimer:
            data = np.load(f'graphics_generation/af3/prediction_accuracy/multimer_scores/{name}_scores.npy')
        else:
            data = np.load(f'graphics_generation/af3/prediction_accuracy/monomer_scores/{name}_monomer_scores.npy')
        choice = np.argmin(data, axis=-1)
        best = np.min(data, axis=-1)
        correct = best.reshape(-1, 1) < np.array([10, 8, 6, 5])
        all_data.append(best)
        # ax = plot_cdf(best, label=name, method='hist')
        print(np.mean(correct, axis=0))

    plot_cdf(all_data, labels=all_labels, method='kde', for_powerpoint=for_powerpoint, colors=colors)
    if for_powerpoint:
        plt.savefig(f'images/modeling/{file_name}_colloq.svg') 
    else:
        plt.savefig(f'images/modeling/{file_name}.svg') 
    plt.show()

def main():
    subsets = [
        ['af2_msa_init_guess'],
        ['af2_msa_init_guess', 'base'],
        ['af2_msa_init_guess', 'base', 'sym'],
        ['af2_msa_init_guess', 'base', 'sym', 'sym_track'],
    ]
    subset_names = ['af3_af2_only', 'af3_base', 'af3_base_sym', 'af3_base_sym_track']

    for subset, subset_name in zip(subsets, subset_names):
        do_plot(subset_name, multimer=True, for_powerpoint=True, subset_names=subset)
    # do_plot('af3_rmsd_cdf', multimer=True, for_powerpoint=False)
    # do_plot('af3_rmsd_monomer', multimer=False)



    
if __name__=='__main__':
    main()
