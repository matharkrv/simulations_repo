import tifffile as tif
from simulation_methods import flatten_intensity, bin_signal
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import form_factor_methods as form_factor
from new_simulation import IqMC_2D, example_simulation

# To work with the simulation you must run it in __main__
if __name__ == "__main__":
    wavelength_ = 1.54e-8  # X-ray wavelength in centimeters
    slit1_radius_ = [0.05, 0.05]  # Half-length of slit 1 in centimeters
    slit2_radius_ = [0.03, 0.03]  # Half-length of slit 2 in centimeters
    sample_detector_distance_ = 100  # Distance between the sample and the detector
    slit_distance_ = 200  # Distance between the slit
    detector_length_ = 7.5  # Length of the detector in centimeters
    detector_center_ = [3, 3]  # Center of the detector in centimeters
    pixel_number_ = 1000  # Number of pixels on the detector
    photon_number_ = int(5e8)  # Number of photons (simulation runs)
    LUT_resolution_ = int(1e4)  # Resolution of the initial scattering probability distribution LUT
    sampling_size_ = int(1e4)  # Amount of scattering incidents to be generated at once at each simulation run
    Rg_average_ = 4e-7  # Average radius of gyration, in centimeters
    Rg_variance_ = 0.05  # Relative variance of the radius of gyration, unitless

    # Generate the distribution of Rgs
    rg_array, distribution_ = form_factor.generate_gaussian_distribution(Rg_average_,
                                                                         np.sqrt(Rg_variance_) * Rg_average_)
    # Set an output path
    output_path = "/Users/Mathar/measurements/Simulations/test/"
    q_max_ = 5e7
    # Run the simulation with the provided parameters
    det_2d, qtable = IqMC_2D(wavelength_, slit1_radius_, slit2_radius_, sample_detector_distance_, slit_distance_,
                             detector_length_,
                             detector_center_, pixel_number_, photon_number_, LUT_resolution_,
                             sampling_size_, form_factor.guinier_ff, rg_array, distribution_, q_max=q_max_)

    # Save the simulation results
    path = "/Users/Mathar/measurements/Simulations/test/"
    tif.imwrite(path + f"dist100_intensity.tif", det_2d, photometric='minisblack')
    tif.imwrite(output_path + f"dist100_qtable.tif", qtable * 1e-7, photometric='minisblack')
    # How to work with the saved simulations:
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
        q_dict[i], i_dict[i] = flatten_intensity(tif.imread(q_p[i]), tif.imread(s_p[i]), px_area=75e-4 ** 2,
                                                 wavelength=0.154, distance=dist_list[i], q_min=q_min, q_max=q_max)

    # (OPTIONAL: If you want to bin the data, you can use the "bin_signal" method of simulation methods)
    q_binned = np.linspace(q_min, q_max, 75)
    y1_binned = bin_signal(q_dict[0], i_dict[0], q_binned)
    y2_binned = bin_signal(q_dict[1], i_dict[1], q_binned)

    # Save the binned data as CSV files for further analysis
    df1 = pd.DataFrame({'q': q_binned, 'Intensity': y1_binned})
    df1.to_csv(path + f"y1_binned.dat", sep=",", index=False)
    df2 = pd.DataFrame({'q': q_binned, 'Intensity': y2_binned})
    df2.to_csv(path + f"y2_binned.dat", sep=",", index=False)
    # Or, Save the binned data as CSV files for further analysis
    df1 = pd.DataFrame({'q': q_dict[0], 'Intensity': i_dict[0]})
    df1.to_csv(path + f"y1_flattened.dat", sep=",", index=False)
    df2 = pd.DataFrame({'q': q_dict[0], 'Intensity': i_dict[0]})
    df2.to_csv(path + f"y2_flattened.dat", sep=",", index=False)
