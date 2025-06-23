import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import scienceplots
from scipy.signal import savgol_filter

# Set Seaborn style
plt.style.use(['science'])

def moving_std(x, window):
    return np.array([np.std(x[max(0, i - window//2):min(len(x), i + window//2 + 1)]) for i in range(len(x))])

def rmsd_plot(out_dirs, names, names_label, flag='', color_inds=None):
    figsize = (5.91, 3.8)
    plt.figure(figsize=figsize)

    if color_inds is None:
        color_inds = list(range(len(names)))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for out_dir, name, label, color_ind in zip(out_dirs, names, names_label, color_inds):
        data = np.loadtxt(out_dir / 'rmsd.xvg', comments=['@', '#'], unpack=True)
        t = data[0]
        rmsd = data[1]

        # Apply smoothing
        smoothed_rmsd = savgol_filter(rmsd, window_length=51, polyorder=3)

        # Compute moving standard deviation
        std_dev = moving_std(rmsd, window=51)

        # Plot smoothed curve
        plt.plot(t, smoothed_rmsd, label=label, color=color_cycle[color_ind])

        # Fill standard deviation area
        plt.fill_between(t, smoothed_rmsd - std_dev, smoothed_rmsd + std_dev, alpha=0.3, color=color_cycle[color_ind])

    plt.title(f'RMSD {flag}')
    plt.xlabel('t (ns)')
    plt.ylabel('RMSD (nm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'graphics_generation/gromacs/data/simulation_out_xvg/joined_rmsd_plot{flag}.svg')
    plt.close()

def gyrate_plot(out_dirs, names, names_label, flag='', color_inds=None):
    figsize = (5.91, 3.8)
    plt.figure(figsize=figsize)

    if color_inds is None:
        color_inds = list(range(len(names)))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for out_dir, name, label, color_ind in zip(out_dirs, names, names_label, color_inds):
        data = np.loadtxt(out_dir / 'gyrate.xvg', comments=['@', '#'], unpack=True)
        t = data[0]
        rmsd = data[1]

        # Apply smoothing
        smoothed_rmsd = savgol_filter(rmsd, window_length=51, polyorder=3)

        # Compute moving standard deviation
        std_dev = moving_std(rmsd, window=51)

        # Plot smoothed curve
        plt.plot(t, smoothed_rmsd, label=label, color=color_cycle[color_ind])

        # Fill standard deviation area
        plt.fill_between(t, smoothed_rmsd - std_dev, smoothed_rmsd + std_dev, alpha=0.3, color=color_cycle[color_ind])

    plt.title(f'Radius of Gyration {flag}')
    plt.xlabel('t (ns)')
    plt.ylabel('$R_g$ (nm)')
    plt.legend()
    plt.savefig(f'graphics_generation/gromacs/data/simulation_out_xvg/joined_gyrate_plot{flag}.svg')  # Save the figure
    plt.close() 


def gyrate_plot_hard(out_dirs, names, flag=''):
    # Create and save plot
    for out_dir, name in zip(out_dirs, names):
        data = np.loadtxt(out_dir/'gyrate.xvg', comments=['@', '#'], unpack=True)
        t = data[0]
        g_r = data[1]
        plt.plot(t, g_r, label=name)

    plt.title(f'Radius of Gyration {flag}')
    plt.xlabel('t (ns)')
    plt.ylabel('$R_g$ (nm)')
    plt.legend()
    plt.savefig(f'graphics_generation/gromacs/data/simulation_out_xvg/joined_gyrate_plot{flag}.svg')  # Save the figure
    plt.close() 

def rmsd_plot_hard(out_dirs, names, names_label, flag=''):
    figsize = (5.91, 3.8)
    plt.figure(figsize=figsize)
    
    for out_dir, name, label in zip(out_dirs, names, names_label):
        data = np.loadtxt(out_dir/'rmsd.xvg', comments=['@', '#'], unpack=True)
        t = data[0]
        rmsd = data[1]
        plt.plot(t, rmsd, label=label)

    plt.title(f'RMSD {flag}')
    plt.xlabel('t (ns)')
    plt.ylabel('RMSD (nm)')
    plt.legend()
    plt.savefig(f'graphics_generation/gromacs/data/simulation_out_xvg/joined_rmsd_plot{flag}.svg')  # Save the figure
    plt.close() 

def main():
    all_names = ['pmpnn_bias_1_pmpnn_0', 'pmpnn_bias_1_pmpnn_3', 'pmpnn_bias_2_5_pmpnn_3', 'pmpnn_bias_2_pmpnn_3', 'pmpnn_bias_2_pmpnn_x', 'pmpnn_default_pmpnn_0', 'rf_diff_3_0_pmpnn_4', 'rf_diff_5_0_pmpnn_3', 'wildtype_2', 'true_wildtype', 'true_wildtype_rna']
    all_names_label = ['P-MPNN, Bias 0', 'P-MPNN, Bias 1', 'P-MPNN, Bias 2.5', 'P-MPNN, Bias 2 (1)', 'P-MPNN, Bias 2 (2)', 'P-MPNN, Bias 0', 'RFdiffusion (1)', 'RFdiffusion (2)', 'Wildtype (Predicted)', 'Wildtype (PDB)', 'Wildtype (PDB, RNA)']
    all_color_inds = list(range(len(all_names)))
    # names = [n for n in all_names if n not in ['rf_diff_5_0_pmpnn_3', 'rf_diff_3_0_pmpnn_4', 'pmpnn_bias_1_pmpnn_0']]

    names = ['pmpnn_bias_2_5_pmpnn_3', 'pmpnn_bias_2_pmpnn_x', 'pmpnn_default_pmpnn_0', 'wildtype_2', 'true_wildtype', 'true_wildtype_rna']
    color_inds = list(range(len(names)))
    names_label = ['P-MPNN, Bias 2.5', 'P-MPNN, Bias 2 (2)', 'P-MPNN, Bias 0', 'Wildtype (Predicted)', 'Wildtype (PDB)', 'Wildtype (PDB, RNA)']

    names_sel = ['true_wildtype', 'true_wildtype_rna']
    names_sel_label = ['Wildtype (PDB)', 'Wildtype (PDB, RNA)']
    color_inds_sel = [color_inds[names.index(name)] for name in names_sel]

    all_names_low_T = [f'{n}_low_T' for n in all_names]
    names_low_T = [f'{n}_low_T' for n in names]
    out_dirs = [Path(f'graphics_generation/gromacs/data/simulation_out_xvg/out_{name}') for name in names]
    out_dirs_sel = [Path(f'graphics_generation/gromacs/data/simulation_out_xvg/out_{name}') for name in names_sel]
    out_dirs_all = [Path(f'graphics_generation/gromacs/data/simulation_out_xvg/out_{name}') for name in all_names]
    # out_dirs_low_T = [Path(f'graphics_generation/gromacs/data/simulation_out_xvg/out_{name}') for name in names_low_T]
    # out_dirs_all_low_T = [Path(f'graphics_generation/gromacs/data/simulation_out_xvg/out_{name}') for name in all_names_low_T]


    rmsd_plot(out_dirs, names, names_label, color_inds=color_inds)
    gyrate_plot(out_dirs, names, names_label, color_inds=color_inds)
    rmsd_plot(out_dirs_sel, names_sel, names_sel_label, flag=' (Wildtype)', color_inds=color_inds_sel)
    gyrate_plot(out_dirs_sel, names_sel, names_sel_label, flag=' (Wildtype)', color_inds=color_inds_sel)



if __name__=='__main__':
    main()