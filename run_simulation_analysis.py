import tifffile as tiff
from simulation_methods import flatten_intensity, bin_signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path to files
path = "/Users/Mathar/measurements/Simulations/test/"

dist_list = [100, 200]
s_p = []
p_p = []
q_p = []
for dist in dist_list:
    s_p.append(path + f"dist{dist}_intensity.tif")
    p_p.append(path + f"poni{dist}.poni")
    q_p.append(path + f"dist{dist}_qtable.tif")

# Read the files and flatten the intensity arrays
q_dict = {}
i_dict = {}

# Define range for reading the data
q_min = 0.05
q_max = 0.25
for i in range(len(dist_list)):
    q_dict[i], i_dict[i] = flatten_intensity(tiff.imread(q_p[i]), tiff.imread(s_p[i]), px_area=75e-4 ** 2,
                                             wavelength=0.154, distance=dist_list[i], q_min=q_min, q_max=q_max)

# (OPTIONAL: If you want to bin the data, you can use the "bin_signal" method of simulation methods)
q_binned = np.linspace(q_min, q_max, 75)
y1_binned = bin_signal(q_dict[0], i_dict[0], q_binned)
y2_binned = bin_signal(q_dict[1], i_dict[1], q_binned)

# Save the binned data as CSV files for further analysis
df1 = pd.DataFrame({'q': q_binned, 'Intensity': y1_binned})
df1.to_csv(path+f"y1_binned.dat", sep=",", index=False)
df2 = pd.DataFrame({'q': q_binned, 'Intensity': y2_binned})
df2.to_csv(path+f"y2_binned.dat", sep=",", index=False)
# Or, Save the binned data as CSV files for further analysis
df1 = pd.DataFrame({'q': q_dict[0], 'Intensity': i_dict[0]})
df1.to_csv(path+f"y1_flattened.dat", sep=",", index=False)
df2 = pd.DataFrame({'q': q_dict[0], 'Intensity': i_dict[0]})
df2.to_csv(path+f"y2_flattened.dat", sep=",", index=False)
