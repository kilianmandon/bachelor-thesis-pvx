import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.transforms import ScaledTranslation
import scienceplots

def load_elisa_data(filepath):
    # Read Excel, relevant data section is B29:M34
    df = pd.read_excel(filepath, header=None, usecols='B:K', skiprows=28, nrows=6)
    data = df.to_numpy()

    # print(data)

    # Group triplicates: 0-2, 3-5, 6-8, 9-11
    col_groups = [data[:, i*3:(i+1)*3] for i in range(3)]
    blank = data[:, 9]

    # if omit_edge_columns:
        # omit group 0 and group 3
        # col_groups = col_groups[1:3]

    return col_groups, blank

def average_triplicates(group):
    return np.nanmean(group, axis=1)

def evaluate_elisa(filepaths, pvx_antibody=True, for_powerpoint=False):
    if for_powerpoint:
        figsize = (8, 4)
        fontsize = 10
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": fontsize,
        })
    else:
        fontsize=8
        plt.style.use(['science'])
        plt.rcParams.update({
            "font.size": fontsize
        })
        figsize = (5.91, 3)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, pvx_antibody in zip(axes, [True, False]):
        if pvx_antibody:
            print(f'-------------------------- For PVX -------------------------')
            reg_slice = slice(4, None)
            non_reg_slice = slice(None, 4)
            filepath = filepaths['pvx']
        else:
            reg_slice = slice(None, None)
            print(f'------------------------- For S-Tag ------------------------')
            non_reg_slice = slice(0, 0)
            filepath = filepaths['s-tag']

        col_groups, blanks = load_elisa_data(filepath)

        # Calibration (ng/µL): rows 0-2 from group 0, and rows 3-5 from group 1
        calibration_conc = np.array([150, 100, 50, 25, 10, 5, 2.5, 1]) * 10
        calibration_od = np.concatenate([
            np.nanmean(col_groups[1][4:6], axis=1),
            np.nanmean(col_groups[2][0:6], axis=1)
        ])

        calibration_od_std = np.concatenate([
            np.nanstd(col_groups[1][4:6], axis=1),
            np.nanstd(col_groups[2][0:6], axis=1)
        ])

        calibration_od_std = (calibration_od_std**2 + np.nanstd(blanks)**2) ** 0.5

        # Blanks: group 1, rows 3-5
        blank_mean = np.nanmean(blanks)

        # Apply blanking
        calibration_od_blanked = calibration_od - blank_mean
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        print('Calibration:')
        for cal_conc, cal_od, cal_std in zip(calibration_conc, calibration_od_blanked, calibration_od_std):
            print(f'concentration: {cal_conc:.3f} | OD: {cal_od:.3f} ± {cal_std:.3f}')
        # print('Calibration: ', np.array2string(np.stack((calibration_od_blanked, calibration_conc)), prefix='Calibration: '))

        # Fit linear regression model (OD = a * concentration + b)
        model = LinearRegression(fit_intercept=fit_intercept)
        X = calibration_conc[reg_slice].reshape(-1, 1)
        y = calibration_od_blanked[reg_slice]
        model.fit(X, y)
        r_squared = model.score(X, y)
        print(f'Coeff of determination: {r_squared:.2f}')

        # Invert regression: concentration = (OD - b) / a
        def predict_concentration(od):
            return (od - model.intercept_) / model.coef_[0]

        print(f'Model: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}')

        # Extract and process sample data
        sample_sets = [
            col_groups[0][0:2],  # Sample 1
            col_groups[0][2:4],  # Sample 2
            col_groups[0][4:6],  # Sample 3
            col_groups[1][0:2],  # Sample 4
            col_groups[1][2:4],  # Sample 5
        ]
        sample_labels = ['d29 Orange', 'S-Tag Orange', 'S-Tag Lila', 'WT PVX', 'H2O']

        print("Blank mean OD:", round(blank_mean, 3))
        print("\n--- Sample Results ---")
        for label, sample in zip(sample_labels, sample_sets):
            od_means = np.nanmean(sample, axis=1) - blank_mean
            od_std = (np.nanstd(sample, axis=1)**2 + np.nanstd(blanks) ** 2) ** 0.5
            est_conc = predict_concentration(od_means)
            est_conc_std = od_std / model.coef_[0]
            dil_facs = np.array([500, 1000])
            est_full_std = est_conc_std * dil_facs / 1000
            print(f"\n{label}:")
            for i, (dilution, od, conc) in enumerate(zip(dil_facs, od_means, est_conc)):
                print(f"  Dilution 1:{dilution}: OD = {od:.3f}±{od_std[i]:.3f}, Est. Conc = {conc:.2f}±{est_conc_std[i]:.3f} ng/mL | Full: {dilution*conc/1000:.2f}±{est_full_std[i]:.3f} µg/mL")

        # Plot calibration curve
        if pvx_antibody:
            x_range = np.linspace(0, 300, 100).reshape(-1, 1)
        else:
            x_range = np.linspace(0, 1400, 100).reshape(-1, 1)
        y_fit = model.predict(x_range)

        ax.plot(calibration_conc[reg_slice], calibration_od_blanked[reg_slice], 'o', label='Calibration points')
        if pvx_antibody:
            ax.plot(calibration_conc[non_reg_slice], calibration_od_blanked[non_reg_slice], 'o', color='gray', label='Calibration points (unused)')
        ax.plot(x_range, y_fit, '-', label=f'Linear fit (R²={r_squared:.2f})')
        ax.set_xlabel('Concentration (ng/mL)')
        if pvx_antibody:
            ax.set_ylabel('OD (blank corrected)')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        # plt.title('ELISA Calibration Curve')
        ax.grid(False)
        ax.legend(loc='lower right', frameon=True)

        label = 'a)' if pvx_antibody else 'b)'
        ax.annotate(
        label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset fontsize',
        fontsize=fontsize, verticalalignment='top')
        # bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
        # if pvx_antibody:
        #     plt.savefig('images/lab/elisa_anti_pvx.svg')
        # else:
        #     plt.savefig('images/lab/elisa_anti_s_tag.svg')

    plt.tight_layout()
    save_path = 'images/lab/colloq_elisa_calibration.svg' if for_powerpoint else 'images/lab/elisa_calibration.svg'
    plt.savefig(save_path)
    plt.show()

# Run evaluation
# reg_slice = slice(None, None)
# evaluate_elisa('graphics_generation/elisa/tecan/elisa_anti_s_15_min.xlsx')
fit_intercept=False
file_paths = {
    'pvx': 'graphics_generation/elisa/tecan/elisa_anti_pvx_15_min.xlsx',
    's-tag': 'graphics_generation/elisa/tecan/elisa_anti_s_15_min.xlsx',
}
for_powerpoint=True

evaluate_elisa(file_paths, for_powerpoint=for_powerpoint)
