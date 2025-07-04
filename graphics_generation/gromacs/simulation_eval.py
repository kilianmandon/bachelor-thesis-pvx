import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import scienceplots
from scipy.signal import savgol_filter

# Set Seaborn style

def moving_std(x, window):
    return np.array([np.std(x[max(0, i - window//2):min(len(x), i + window//2 + 1)]) for i in range(len(x))])

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

def combined_rmsd_gyrate_plot(out_dirs, names, names_label, flag='', color_inds=None, for_powerpoint=False, show_both=False):
    if for_powerpoint:
        fontsize = 10
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": fontsize,
        })
        if show_both:
            figsize = (12, 4)
            fig, axs = plt.subplots(1, 2, figsize=figsize)
        else:
            figsize = (6, 4)
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            axs = [axs]
    else:
        fontsize=8
        # figsize = (5.91, 4)
        # fig, axs = plt.subplots(1, 2, figsize=figsize)

        if show_both:
            figsize = (5.91, 6)
            fig, axs = plt.subplots(2, 1, figsize=figsize)
        else:
            figsize = (5.91, 3)
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            axs = [axs]
        plt.rcParams.update({"font.size": fontsize})

    if color_inds is None:
        color_inds = list(range(len(names)))

    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # color_cycle = 2*color_cycle


    for out_dir, name, label, color_ind in zip(out_dirs, names, names_label, color_inds):
        # RMSD
        data_rmsd = np.loadtxt(out_dir / 'rmsd.xvg', comments=['@', '#'], unpack=True)
        t_rmsd = data_rmsd[0]
        y_rmsd = data_rmsd[1]
        smoothed_rmsd = savgol_filter(y_rmsd, window_length=51, polyorder=3)
        std_rmsd = moving_std(y_rmsd, window=51)

        axs[0].plot(t_rmsd, smoothed_rmsd, label=label, color=color_cycle[color_ind])
        axs[0].fill_between(t_rmsd, smoothed_rmsd - std_rmsd, smoothed_rmsd + std_rmsd, alpha=0.3, color=color_cycle[color_ind])

        # Gyration
        data_gyr = np.loadtxt(out_dir / 'gyrate.xvg', comments=['@', '#'], unpack=True)
        t_gyr = data_gyr[0]
        y_gyr = data_gyr[1]
        smoothed_gyr = savgol_filter(y_gyr, window_length=51, polyorder=3)
        std_gyr = moving_std(y_gyr, window=51)

        if show_both:
            axs[1].plot(t_gyr, smoothed_gyr, label=label, color=color_cycle[color_ind])
            axs[1].fill_between(t_gyr, smoothed_gyr - std_gyr, smoothed_gyr + std_gyr, alpha=0.3, color=color_cycle[color_ind])

    # Axis labels and titles
    if for_powerpoint:
        axs[0].set_title(f'RMSD {flag}')
        if show_both:
            axs[1].set_title(f'Radius of Gyration {flag}')
    else:
        if show_both:
            axs[0].text(-0.1, 1.05, 'a)', va='bottom', ha='right', transform=axs[0].transAxes, fontsize=10)
            axs[1].text(-0.1, 1.05, 'b)', va='bottom', ha='right', transform=axs[1].transAxes, fontsize=10)
        # axs[0].text(0.01, 0.95, 'a)', va='top', ha='left', transform=axs[0].transAxes, fontsize=2*fontsize)
        # axs[1].text(0.01, 0.95, 'b)', va='top', ha='left', transform=axs[1].transAxes, fontsize=2*fontsize)
        # axs[0].text(-0.5, 0.5, 'a)', va='center', ha='center', multialignment='center',
        #     transform=axs[0].transAxes, fontsize=fontsize)
        # axs[1].text(-0.5, 0.5, 'b)', va='center', ha='center', multialignment='center',
        #     transform=axs[1].transAxes, fontsize=fontsize)
    
    axs[0].set_xlabel('t (ns)')
    axs[0].set_ylabel('RMSD (nm)')
    axs[0].legend(loc='lower right')

    if show_both:
        axs[1].set_xlabel('t (ns)')
        axs[1].set_ylabel('$R_g$ (nm)')
        axs[1].legend(loc='lower right')

    fig.tight_layout()
    prefix = 'colloq_' if for_powerpoint else ''
    fig.savefig(f'images/modeling/{prefix}gromacs_joined_rmsd_gyrate_plot{flag}.svg')
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
    plt.legend(loc='lower right')
    plt.savefig(f'graphics_generation/gromacs/data/simulation_out_xvg/joined_rmsd_plot{flag}.svg')  # Save the figure
    plt.close() 

def main():
    for_powerpoint = False

    if not for_powerpoint:
        plt.style.use(['science'])

    all_names = ['pmpnn_bias_1_pmpnn_0', 'pmpnn_bias_1_pmpnn_3', 'pmpnn_bias_2_5_pmpnn_3', 'pmpnn_bias_2_pmpnn_3', 'pmpnn_bias_2_pmpnn_x', 'pmpnn_default_pmpnn_0', 'rf_diff_3_0_pmpnn_4', 'rf_diff_5_0_pmpnn_3', 'wildtype_2', 'true_wildtype', 'true_wildtype_rna']
    all_names_label = ['P-MPNN, Bias 0', 'P-MPNN, Bias 1', 'P-MPNN, Bias 2.5', 'P-MPNN, Bias 2 (1)', 'P-MPNN, Bias 2 (2)', 'P-MPNN, Bias 0', 'RFdiffusion (1)', 'RFdiffusion (2)', 'Wildtype (Predicted)', 'Wildtype (PDB)', 'Wildtype (PDB, RNA)']
    all_color_inds = list(range(len(all_names)))
    # names = [n for n in all_names if n not in ['rf_diff_5_0_pmpnn_3', 'rf_diff_3_0_pmpnn_4', 'pmpnn_bias_1_pmpnn_0']]

    # names = ['pmpnn_bias_2_5_pmpnn_3', 'pmpnn_bias_2_pmpnn_x', 'pmpnn_default_pmpnn_0', 'wildtype_2', 'true_wildtype', 'true_wildtype_rna']
    names = ['pmpnn_bias_2_pmpnn_x', 'pmpnn_default_pmpnn_0', 'pmpnn_bias_2_5_pmpnn_3', 'wildtype_2', 'true_wildtype', 'true_wildtype_rna', 'rf_diff_3_0_pmpnn_4', 'rf_diff_5_0_pmpnn_3']
    color_inds = list(range(len(names)))
    # names_label = ['P-MPNN, Bias 2.5', 'P-MPNN, Bias 2 (2)', 'P-MPNN, Bias 0', 'Wildtype (Predicted)', 'Wildtype (PDB)', 'Wildtype (PDB, RNA)']
    if for_powerpoint:
        names_label = ['Design A (90% Identity)', 'Design B (53% Identity)', 'Design C (92% Identity)', 'Wildtype (Predicted)', 'Wildtype (PDB)', 'Wildtype (PDB, RNA)']
    else:
        names_label = ['Design A (90\% Identity)', 'Design B (53\% Identity)', 'Design C (92\% Identity)', 'Wildtype (Predicted)', 'Wildtype (PDB)', 'Wildtype (PDB, RNA)']

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

    combined_rmsd_gyrate_plot(out_dirs, names, names_label, color_inds=color_inds, for_powerpoint=for_powerpoint)
    combined_rmsd_gyrate_plot(out_dirs_sel, names_sel, names_sel_label, flag=' (Wildtype)', color_inds=color_inds_sel, for_powerpoint=for_powerpoint)

    # rmsd_plot(out_dirs, names, names_label, color_inds=color_inds, for_powerpoint=for_powerpoint)
    # gyrate_plot(out_dirs, names, names_label, color_inds=color_inds, for_powerpoint=for_powerpoint)
    # rmsd_plot(out_dirs_sel, names_sel, names_sel_label, flag=' (Wildtype)', color_inds=color_inds_sel, for_powerpoint=for_powerpoint)
    # gyrate_plot(out_dirs_sel, names_sel, names_sel_label, flag=' (Wildtype)', color_inds=color_inds_sel, for_powerpoint=for_powerpoint)



if __name__=='__main__':
    main()