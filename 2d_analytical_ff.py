import numpy as np
import form_factor_methods as form_factor
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Choose detector resolution
resolution = 500
# Define the detector array
detector_array = np.zeros([resolution, resolution])
# Define Rg
rg = 4
# Define the variance
variance = 0.0005
# Generate the distribution of Rgs
rg_array, distribution_ = form_factor.generate_gaussian_distribution(rg, np.sqrt(variance) * rg)
# Define the size of the pixels
px_size = 0.075
# Define the sample-detector distance
sample_detector_distance = 200
# Define the wavelength
wavelength = 0.154
# Define the sigmas for the anisotropic smearing
sigma_y = 1
sigma_x = 0.1
# Define the detector center
dc = resolution/2
# Generate the q table. Start by generating the indices along the detector
xy_table = np.indices((resolution, resolution), dtype=float)
# Calculate r, the distance of any pixel from the center of the detector
r = px_size * np.sqrt((dc - xy_table[0]) ** 2 + (dc - xy_table[1]) ** 2)
# Calculate the scattering angle
sint = np.sin(np.arctan(r / sample_detector_distance))
# q is then given by 4pi/lambda * sin(theta)
qtable = 4 * np.pi * sint / wavelength
# For every pixel in the detector, calculate the intensity from the form factor
for i in range(resolution):
    for j in range(resolution):
        detector_array[i, j] += form_factor.polyP(qtable[i, j], form_factor.sphere, rg_array, distribution_)
# Gaussian convolution of the 2D array based on the given sigmas
Convoluted_detector = gaussian_filter(detector_array, sigma=[sigma_y, sigma_x])
# Plot
plt.imshow(np.log(Convoluted_detector))
plt.show()
