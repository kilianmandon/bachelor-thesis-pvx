import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scienceplots


def load_fluorescense_data(filepath):

    # Read Excel, relevant data section is B29:M34
    df = pd.read_excel(filepath, header=None, usecols='B:K', skiprows=30, nrows=6)
    data = df.to_numpy()

    print(data)

    # Group triplicates: 0-2, 3-5, 6-8, 9-11
    col_groups = [data[:, i*3:(i+1)*3] for i in range(3)]
    blank = data[:, 9]

    # if omit_edge_columns:
        # omit group 0 and group 3
        # col_groups = col_groups[1:3]

    return col_groups, blank

def average_triplicates(group):
    return np.nanmean(group, axis=1)

def evaluate_eyfp(filepath, for_powerpoint=False):
    if for_powerpoint:
        fontsize = 14
        figsize = (7, 5)
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "font.size": fontsize,
        })
    else:
        plt.style.use(['science'])
        fontsize=10
        # fontsize=12
        # plt.rcParams.update({
            #"font.size": fontsize,
        # })
        figsize = (5.91, 3.8)
    col_groups, blanks = load_fluorescense_data(filepath)

    # Calibration (ng/well): rows 0-2 from group 0, and rows 3-5 from group 1
    calibration_conc = np.array([100, 50, 25, 10, 5, 2.5])

    # Blanks
    blank_mean = np.nanmean(blanks)
    # Compute SDs for calibration
    calibration_flu_raw = np.nanmean(col_groups[2], axis=1)
    calibration_flu_sd = np.nanstd(col_groups[2], axis=1, ddof=1)
    calibration_flu_blanked = calibration_flu_raw - blank_mean

    print("\n=== Calibration Data ===")
    print("Concentration (ng/well)\tFluorescence (mean ± SD)")
    for conc, flu, sd in zip(calibration_conc, calibration_flu_blanked, calibration_flu_sd):
        print(f"{conc:.1f}\t\t\t\t{flu:.3f} ± {sd:.3f}")

    # Fit linear regression
    model = LinearRegression(fit_intercept=fit_intercept)
    X = calibration_conc[reg_slice].reshape(-1, 1)
    y = calibration_flu_blanked[reg_slice]
    model.fit(X, y)
    r_squared = model.score(X, y)
    print(f"\nLinear model fit: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}")
    print(f"R² = {r_squared:.4f}")

    def predict_concentration(od):
        return (od - model.intercept_) / model.coef_[0]

    # Extract and process sample data
    sample_sets = [
        col_groups[0][0:4],  # Sample 1
        col_groups[1][0:4],  # Sample 2
        np.concatenate((col_groups[0][4:6], col_groups[1][4:6]), axis=0)
    ]
    sample_labels = ['#2', '#3', 'H2O']
    dil_fac = [50, 100, 200, 500]

    print(f"\nBlank Mean Fluorescence: {blank_mean:.3f}")
    print("\n=== Sample Measurements ===")
    for label, sample in zip(sample_labels, sample_sets):
        sample_blanked = sample - blank_mean
        flu_means = np.nanmean(sample_blanked, axis=1)
        flu_sds = np.nanstd(sample_blanked, axis=1, ddof=1)
        est_conc = predict_concentration(flu_means)
        conc_sds = flu_sds / model.coef_[0]
        full_conc_sds = conc_sds * dil_fac

        print(f"\nSample: {label}")
        print("Dilution\tFluorescence (mean ± SD)\tEstimated Conc. (µg/mL)\tBack-calc (µg/mL)")
        for i, (dilution, mean, sd, conc) in enumerate(zip(dil_fac, flu_means, flu_sds, est_conc)):
            print(f"1:{dilution}\t\t{mean:.3f} ± {sd:.3f}\t\t\t{conc:.2f} ± {conc_sds[i]:.3f}\t\t\t{dilution * conc:.2f} ± {full_conc_sds[i]:.3f}")

    # (Optional) Plot — keep or remove depending on need
    x_range = np.linspace(0, 110, 100).reshape(-1, 1)
    # x_range = np.linspace(0, 30, 100).reshape(-1, 1)
    y_fit = model.predict(x_range)

    plt.figure(figsize=figsize)
    plt.plot(calibration_conc[reg_slice], calibration_flu_blanked[reg_slice], 'o', label='Calibration points')
    plt.plot(calibration_conc[non_reg_slice], calibration_flu_blanked[non_reg_slice], 'o', label='Calibration points (unused)', color='gray')
    # plt.errorbar(calibration_conc, calibration_flu_blanked, calibration_flu_sd, fmt='x', capsize=5,label='Calibration points')
    plt.plot(x_range, y_fit, '-', label=f'Linear fit ($R^2$={r_squared:.2f})')
    plt.xlabel('Concentration ($\mu$g/mL)')
    plt.ylabel('Fluorescence (blank corrected)')
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.grid(False)
    plt.legend()
    plt.tight_layout()
    if for_powerpoint:
        save_path = 'images/lab/colloq_fluorescense_intensity.svg'
    else:
        save_path = 'images/lab/fluorescense_intensity.svg'
    plt.savefig(save_path)

    plt.show()

# Run evaluation
# reg_slice = slice(None, None)
# evaluate_elisa('graphics_generation/elisa/tecan/elisa_anti_s_15_min.xlsx')
reg_slice = slice(2, None)
non_reg_slice = slice(None, 2)
fit_intercept=False
evaluate_eyfp('graphics_generation/eyfp_fluorescense/eyfp_fluorescense.xlsx', for_powerpoint=False)