import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Sample names and dilutions
samples = ['d28-A', 'S-Tag-A', 'S-Tag-B', 'PVX CP', 'Non-template']
dilutions = ['1:500', '1:1000']

# Back-calculated concentrations (µg/mL) and standard deviations
# Left table (anti-PVX)
pvx_values = [
    [16.0, 30.7],   # d28-A
    [18.0, 33.4],   # S-Tag-A
    [0.4, 3.1],     # S-Tag-B
    [66.9, 134.2],  # CP-PVX
    [2.8, 1.9],     # Non-template
]
pvx_errors = [
    [2.2, 4.9],
    [2.5, 4.0],
    [2.0, 4.6],
    [2.2, 4.0],
    [2.5, 6.2],
]

# Right table (anti-S-Tag)
stag_values = [
    [22, 44],
    [1317, 2399],
    [813, 1198],
    [12, 50],
    [21, 38],
]
stag_errors = [
    [12, 22],
    [73, 86],
    [40, 107],
    [10, 22],
    [10, 21],
]

# Convert to NumPy arrays for easier plotting
pvx_vals = np.array(pvx_values)
pvx_errs = np.array(pvx_errors)
stag_vals = np.array(stag_values)
stag_errs = np.array(stag_errors)

# Bar plot parameters
x = np.arange(len(samples))  # group locations
width = 0.35  # width of the bars

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Anti-PVX plot
axs[0].bar(x - width/2, pvx_vals[:, 0], width, yerr=pvx_errs[:, 0], label='1:500', capsize=4)
axs[0].bar(x + width/2, pvx_vals[:, 1], width, yerr=pvx_errs[:, 1], label='1:1000', capsize=4)
axs[0].set_xticks(x)
axs[0].set_xticklabels(samples, rotation=45)
axs[0].set_ylabel('Back-calculated (µg/mL)')
axs[0].set_title('ELISA with anti-PVX antibody')
axs[0].legend()

# Anti-S-Tag plot
axs[1].bar(x - width/2, stag_vals[:, 0], width, yerr=stag_errs[:, 0], label='1:500', capsize=4)
axs[1].bar(x + width/2, stag_vals[:, 1], width, yerr=stag_errs[:, 1], label='1:1000', capsize=4)
axs[1].set_xticks(x)
axs[1].set_xticklabels(samples, rotation=45)
axs[1].set_ylabel('Back-calculated (µg/mL)')
axs[1].set_title('ELISA with anti-S-Tag antibody')
axs[1].legend()

# Layout and display
plt.tight_layout()
plt.savefig('images/modeling/colloq_elisa_bar_plot.svg')
plt.show()