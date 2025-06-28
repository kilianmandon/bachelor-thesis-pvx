import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_elisa_data(filepath, omit_edge_columns=True):
    # Read Excel, relevant data section is B29:M34
    df = pd.read_excel(filepath, header=None, usecols='B:M', skiprows=28, nrows=6)
    data = df.to_numpy()
    if omit_edge_columns:
        data[:, 0] = np.nan
        data[:, -1] = np.nan


    print(data)

    # Group triplicates: 0-2, 3-5, 6-8, 9-11
    col_groups = [data[:, i*3:(i+1)*3] for i in range(4)]

    # if omit_edge_columns:
        # omit group 0 and group 3
        # col_groups = col_groups[1:3]

    return col_groups

def average_triplicates(group):
    return np.nanmean(group, axis=1)

def evaluate_elisa(filepath, omit_edge_columns=True):
    col_groups = load_elisa_data(filepath, omit_edge_columns)

    # Calibration (ng/well): rows 0-2 from group 0, and rows 3-5 from group 1
    calibration_conc = np.array([100, 50, 25, 10, 5, 2.5])
    calibration_od = np.concatenate([
        np.nanmean(col_groups[2][3:], axis=1),
        np.nanmean(col_groups[3][0:3], axis=1)
    ])

    # Blanks: group 1, rows 3-5
    blanks = np.nanmean(col_groups[3][3:6], axis=1)
    blank_mean = np.mean(blanks)

    # Apply blanking
    calibration_od_blanked = calibration_od - blank_mean

    # Fit linear regression model (OD = a * concentration + b)
    model = LinearRegression()
    X = calibration_conc.reshape(-1, 1)
    y = calibration_od_blanked
    model.fit(X, y)

    # Invert regression: concentration = (OD - b) / a
    def predict_concentration(od):
        return (od - model.intercept_) / model.coef_[0]

    print(f'Model: y = {model.coef_[0]:.4f} * x + {model.intercept_:.4f}')

    # Extract and process sample data
    sample_sets = [
        col_groups[0][0:3],  # Sample 1
        col_groups[0][3:6],  # Sample 2
        col_groups[1][0:3],  # Sample 3
        col_groups[1][3:6],  # Sample 4
        col_groups[2][0:3],  # Sample 5
    ]
    sample_labels = ['d29 Orange', 'S-Tag Orange', 'S-Tag Lila', 'WT PVX', 'H2O']

    print("Blank mean OD:", round(blank_mean, 3))
    print("\n--- Sample Results ---")
    for label, sample in zip(sample_labels, sample_sets):
        od_means = np.nanmean(sample, axis=1) - blank_mean
        est_conc = predict_concentration(od_means)
        print(f"\n{label}:")
        for dilution, od, conc in zip(['1:500', '1:1000', '1:2000'], od_means, est_conc):
            print(f"  Dilution {dilution}: OD = {od:.3f}, Est. Conc = {conc:.2f} ng/well")

    # Plot calibration curve
    x_range = np.linspace(0, 110, 100).reshape(-1, 1)
    y_fit = model.predict(x_range)

    plt.figure(figsize=(6, 4))
    plt.plot(calibration_conc, calibration_od_blanked, 'o', label='Calibration points')
    plt.plot(x_range, y_fit, '-', label='Linear fit')
    plt.xlabel('Concentration (ng/well)')
    plt.ylabel('OD (blank corrected)')
    plt.title('ELISA Calibration Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run evaluation
evaluate_elisa('graphics_generation/elisa/tecan/elisa_anti_pvx_30_min.xlsx', omit_edge_columns=True)