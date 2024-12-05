import numpy as np
import form_factor_methods as form_factor
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import os

# Define the parameter arrays for Rg and variance
rgs = np.linspace(4.6, 8, 1)  # Different values for Rg
variances = np.linspace(0.01, 0.5, 5)  # Different values for variance
sigmas_x= np.linspace(0.1, 0.33, 5)    # Array of different x axis sigmas
sigmas_y = np.linspace(3.78, 5, 5)    # Array of different y axis sigmas
# Fixed parameters
resolution = 500
px_size = 0.075
sample_detector_distance = 1500
wavelength = 0.154
dc = resolution / 2

# Generate xy_table and calculate qtable (these values do not change across iterations)
xy_table = np.indices((resolution, resolution), dtype=float)
r = px_size * np.sqrt((dc - xy_table[0]) ** 2 + (dc - xy_table[1]) ** 2)
sint = np.sin(np.arctan(r / sample_detector_distance))
qtable = 4 * np.pi * sint / wavelength

# Loop through each combination of Rg and variance
for rg, variance, sigma_x, sigma_y in product(rgs, variances, sigmas_x, sigmas_y):
    # Initialize the detector array for each iteration
    detector_array = np.zeros([resolution, resolution])

    # Generate the distribution of Rgs
    rg_array, distribution_ = form_factor.generate_gaussian_distribution(rg, np.sqrt(variance) * rg)

    # For every pixel in the detector, calculate the intensity from the form factor. CHANGE FORM FACTOR HERE
    for i in range(resolution):
        for j in range(resolution):
            detector_array[i, j] += form_factor.polyP(qtable[i, j], form_factor.guinier_ff, rg_array, distribution_)

    # Gaussian convolution of the 2D array based on the given sigmas
    Convoluted_detector = gaussian_filter(detector_array, sigma=[5, 2])
    # a = 12
    # Save the Convoluted_detector as a CSV file
    path = 'path'
    rg_folder = f'rg_{rg:.2f}'
    variance_folder = f'variance_{variance:.2f}'
    output_folder = os.path.join(rg_folder, variance_folder)
    os.makedirs(path+output_folder, exist_ok=True)
    file_name = f'/guinier_sigmax_{sigma_x:.2f}_sigmay_{sigma_y:.2f}_rg_{rg:.2f}_variance_{variance:.2f}.csv'
    pd.DataFrame(Convoluted_detector).to_csv(path + output_folder + "/" + file_name, index=False, header=False)

    # Optional: Plot the log difference (for visualization purposes)
    # log_difference =  np.log(Convoluted_detector)
    # plt.imshow(log_difference, vmin=np.min(log_difference), vmax=np.max(log_difference))
    # plt.colorbar()
    # plt.title(f'Rg: {rg:.2f}, Variance: {variance:.2f}')
    # plt.show()
