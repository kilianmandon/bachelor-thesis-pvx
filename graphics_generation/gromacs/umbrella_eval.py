import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import scienceplots

# Set Seaborn style
plt.style.use(['science'])


def plot_profile(exp_names, labels, color_inds=None):
    figsize = (5.91, 3.8)
    plt.figure(figsize=figsize)

    if color_inds is None:
        color_inds = list(range(len(exp_names)))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for exp_name, label, color_ind in zip(exp_names, labels, color_inds):
        out_dir = Path(f'graphics_generation/gromacs/data/umbrella_out_xvg/out_{exp_name}')
        data = np.loadtxt(out_dir/'wham_profile.xvg', comments=['@', '#'], unpack=True)
        t = data[0]
        G = data[1]
        plt.plot(t, G, label=label, color=color_cycle[color_ind])

    plt.title('WHAM Profile')
    plt.xlabel('x (nm)')
    plt.ylabel('$\Delta$ G (kcal)')
    plt.legend()

    plt.savefig(f'graphics_generation/gromacs/data/umbrella_out_xvg/joined_wham_profile.svg')  # Save the figure
    plt.close() 

def main():
    names = ['pmpnn_bias_2_5_pmpnn_3', 'pmpnn_bias_2_pmpnn_x', 'pmpnn_default_pmpnn_0_strong', 'wildtype_2', 'true_wildtype', 'true_wildtype_rna']
    color_inds = list(range(len(names)))

    exp_names = ['wildtype_2', 'true_wildtype', 'pmpnn_bias_2_pmpnn_x', 'pmpnn_bias_2_5_pmpnn_3', 'pmpnn_default_pmpnn_0_strong']
    color_inds_exp = [color_inds[names.index(name)] for name in exp_names]
    # names_label = ['P-MPNN, Bias 2.5', 'P-MPNN, Bias 2 (2)', 'P-MPNN, Bias 0', 'Wildtype (Predicted)', 'Wildtype (PDB)', 'Wildtype (PDB, RNA)']
    names_label = ['Wildtype (Predicted)', 'Wildtype (PDB)', 'P-MPNN, Bias 2 (2)', 'P-MPNN, Bias 2.5', 'P-MPNN, Bias 0']
    plot_profile(exp_names, names_label, color_inds=color_inds_exp)

if __name__=='__main__':
    main()