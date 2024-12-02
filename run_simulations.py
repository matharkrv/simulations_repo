import numpy as np
import pandas as pd
import tifffile as tif
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import form_factor_methods as form_factor
from new_simulation import IqMC_2D
from simulation_methods import flatten_intensity, bin_signal
from itertools import product
import os


def zero_ff(x, rg):
    return np.zeros_like(x) * rg


# To work with the simulation you must run it in __main__
if __name__ == "__main__":
    wavelength_ = 1.54e-8  # X-ray wavelength in centimeters
    slit1_radius_ = [0.01, 0.05]  # Half-length of slit 1 in centimeters
    slit2_radius_ = [0.01, 0.06]  # Half-length of slit 2 in centimeters
    sample_detector_distance_ = 50  # Distance between the sample and the detector
    slit_distance_ = 200  # Distance between the slit
    detector_length_ = 7.5  # Length of the detector in centimeters
    detector_center_ = [3, 3]  # Center of the detector in centimeters
    pixel_number_ = 1000  # Number of pixels on the detector
    photon_number_ = int(1e9)  # Number of photons (simulation runs)
    LUT_resolution_ = int(1e4)  # Resolution of the initial scattering probability distribution LUT
    sampling_size_ = int(1e4)  # Amount of scattering incidents to be generated at once at each simulation run
    Rg_average_ = 4e-7  # Average radius of gyration, in centimeters
    Rg_variance_ = 0.0005  # Relative variance of the radius of gyration, unitless

    # Set an output path - change this to whatever folder you want
    output_path = ('/Users/Mathar/Library/CloudStorage/GoogleDrive-matark@mail.tau.ac.il/Drive partagés/Beck lab 3'
                   '/Matar/PAZY_xray/generation_of_simulations/guinier/')

    # Create output directories if they do not exist
    os.makedirs(output_path, exist_ok=True)
    # Set an output path
    # Run the simulation with the provided parameters
    slits = np.linspace(0.02, 0.08, 5)
    variances = np.linspace(0.001, 0.05, 5)
    rgs = np.linspace(2, 8, 5) * 1e-7
    distances = np.linspace(50, 200, 5)
    for slit1, slit2, variance, rg, dist in product(slits, slits, variances, rgs, distances):
        rg_array, distribution_ = form_factor.generate_gaussian_distribution(rg, np.sqrt(variance) * rg)
        det_2d, qtable = IqMC_2D(wavelength_, [0.01, slit1], [0.01, slit2], dist, slit_distance_,
                                 detector_length_,
                                 detector_center_, pixel_number_, photon_number_, LUT_resolution_,
                                 sampling_size_, form_factor.guinier_ff, rg_array, distribution_, q_max=10.3e7)
        sim_name = f"guinier_rg_{np.round(rg*1e7, 2)}_v_{variance}_slit2_y_{slit2}__slit1_y_{slit1}_distance_{dist}"
        out_path = os.path.join(output_path, f"rg_{rg:.2f}/", f"var_{variance:.2f}/")
        out_path_sample = os.path.join(out_path, sim_name)
        out_path_qtable = os.path.join(out_path, "qtable/", sim_name)

        os.makedirs(out_path_sample, exist_ok=True)

        tif.imwrite(out_path_sample + "/sample.tif",
                    det_2d, photometric='minisblack')
        # Here the q values are normalized by 1e-7 to change their units to nm^-1.
        tif.imwrite(out_path_sample + "/qtable.tif",
                    qtable * 1e-7, photometric='minisblack')

        px_size_ = ((detector_length_ * 1e4) / pixel_number_) / 10
        poni1 = detector_center_[0] / 100  # meters
        poni2 = detector_center_[1] / 100  # meters
        with open(out_path_sample + f"/poni.poni", 'w') as file:
            file.write("Detector: Simulation")
            file.write(f"Detector_config: pixel1: {px_size_}, pixel2: {px_size_}, max_shape: [{pixel_number_},"
                       f" {pixel_number_}]")

            file.write(f'Distance: {detector_length_ / 100}\n')
            file.write(f'Poni1: {poni1}\n')
            file.write(f'Poni2: {poni2}\n')
            file.write("Rot1: 0.0\n")
            file.write("Rot2: 0.0\n")
            file.write("Rot3: 0.0\n")
            file.write(f"wavelength: {wavelength_ / 100}")

