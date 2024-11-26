import multiprocessing as mp
import multiprocessing.pool as mpp
import matplotlib.pyplot as plt
import tifffile as tif
from numba import prange
from numpy import cos, sin
from numpy.random import SeedSequence
from tqdm import tqdm
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import form_factor_methods as form_factor
from istarmap import istarmap
from simulation_methods import *
from typing import List, Callable, Optional, Tuple

# Import istarmap for simultaneous tracking of the multiprocess and its execution
mpp.Pool.istarmap = istarmap


def _simulation_process_2D(*args):
    """
        Perform the 2D simulation process for X-ray scattering.

        This function simulates the scattering of X-rays through a two-slit system and onto a 2D detector.
        It calculates the intensity distribution on the detector based on various input parameters. All units are in
        centimeters.

        Parameters:
        -----------
        *args : tuple
            A tuple containing the following parameters:
            - batch (int): Number of iterations to perform.
            - seed (int): Random number generator seed.
            - sampling_size (int): Number of samples to generate in each iteration.
            - slit_1_x (float): Half-width of the first slit in x-direction.
            - slit_1_y (float): Half-height of the first slit in y-direction.
            - slit_2_x (float): Half-width of the second slit in x-direction.
            - slit_2_y (float): Half-height of the second slit in y-direction.
            - slit_distance (float): Distance between the two slits.
            - q_out_array (array): Array of q values for the output.
            - delta_q (array): Array of delta q values.
            - inv_cdf_interp (callable): Inverse cumulative distribution function for q values.
            - sample_detector_distance (float): Distance between the sample and the detector.
            - k (float): Wave vector scale.
            - pixel_number (int): Number of pixels in each dimension of the detector.
            - centers (tuple): Center coordinates of the detector (x, y).
            - px_size (float): Size of each pixel.

        Returns:
        --------
        numpy.ndarray
            A 2D array representing the intensity distribution on the detector.
            Each element corresponds to the photon count at that pixel position.
    """
    (
        batch, seed, sampling_size, slit_1_x, slit_1_y, slit_2_x, slit_2_y, slit_distance, q_out_array,
        delta_q,
        inv_cdf_interp,
        sample_detector_distance, k, pixel_number, centers, px_size) = args
    rng = np.random.default_rng(seed)
    intensity_update = np.zeros([pixel_number, pixel_number]).astype(np.float64)
    for _ in prange(batch):
        q_random = inv_cdf_interp(rng.uniform(0, 1, size=sampling_size))
        if np.any(np.isfinite(q_random)):
            angle_delta = np.arcsin(q_random / (2 * k))
            azimuth_angle = rng.uniform(0, 2 * np.pi - np.finfo(np.float64).eps, size=sampling_size)
        else:
            angle_delta = 0
            azimuth_angle = 0
        theta_in, chi_in, sl2_x, sl2_y = sample_from_slits(rng, slit_1_x, slit_1_y, slit_2_x, slit_2_y, slit_distance,
                                                           sampling_size)
        sin_T = sin(angle_delta)
        cos_T = cos(angle_delta)
        sin_t = sin(theta_in)
        cos_t = cos(theta_in)
        sin_chi = sin(chi_in)
        cos_chi = cos(chi_in)
        sin_Chi = sin(azimuth_angle)
        cos_Chi = cos(azimuth_angle)
        r_x = sin_T * (cos_chi * cos_Chi * cos_t - sin_chi * sin_Chi) + cos_chi * sin_t * cos_T
        r_y = sin_T * (sin_chi * cos_Chi * cos_t + cos_chi * sin_Chi) + sin_chi * sin_t * cos_T
        r_z = cos_Chi * sin_t * sin_T - cos_t * cos_T
        x1 = sample_detector_distance * (r_x / r_z) + sl2_x
        y1 = sample_detector_distance * (r_y / r_z) + sl2_y
        x = x1 / px_size + centers[0]
        y = y1 / px_size + centers[1]
        mask = (x <= pixel_number - 1) & (y <= pixel_number - 1) & (x >= 0) & (y >= 0)
        y_masked = np.round(y[mask], 0).astype(int)
        x_masked = np.round(x[mask], 0).astype(int)
        # Get the unique pairs and their counts
        unique_pairs, counts = np.unique(np.vstack((y_masked, x_masked)).T, axis=0, return_counts=True)
        # Update intensity_update with the counts for each unique pair
        intensity_update[unique_pairs[:, 0], unique_pairs[:, 1]] += counts
    return intensity_update


def IqMC_2D(wavelength: float,
            slit1_radius: List[float],
            slit2_radius: List[float],
            sample_detector_distance: float,
            slit_distance: float,
            detector_length: float,
            detector_center: List[float],
            pixel_number: int,
            photon_number: int,
            LUT_resolution: int,
            sampling_size: int,
            scattering_function: Callable,
            Rg_array: np.ndarray,
            distribution: np.ndarray,
            q_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation for X-ray scattering intensity profile calculation.

    Parameters:
    wavelength (float): X-ray wavelength in centimeters.
    slit1_radius (list): Half-length of slit 1 in centimeters.
    slit2_radius (list): Half-length of slit 2 in centimeters.
    sample_detector_distance (float): Distance between the sample and the detector in centimeters.
    slit_distance (float): Distance between the slit in centimeters.
    detector_length (float): Length of the detector in centimeters.
    detector_center (list): Center of the detector in centimeters.
    pixel_number (int): Number of pixels on the detector.
    photon_number (int): Number of photons (simulation runs).
    LUT_resolution (int): Resolution of the initial scattering probability distribution LUT.
    sampling_size (int): Amount of scattering incidents to be generated at once at each simulation run.
    scattering_function (function): Scattering form factor function.
    Rg_array (array): Array of all possible Rgs.
    distribution (array): Distribution for the array of Rgs.
    q_max (float, optional): Maximum q value. If not provided, it is calculated based on the given parameters.

    Returns:
    tuple: A tuple containing the final Intensity profile and the array of exit wave vectors.
    """

    k = 2 * np.pi / wavelength  # Wave vector scale in centimeters
    px_size = detector_length / pixel_number  # Size of each pixel in centimeters.
    pixel_centers = (np.array(detector_center) / px_size).astype(int)  # Location of beam center in pixels.
    # Calculate the grid of all possible distances from the beam center.
    detector_grid = np.sqrt(np.add.outer((np.arange(pixel_number) * px_size - pixel_centers[1] * px_size) ** 2,
                                         (np.arange(pixel_number) * px_size - pixel_centers[0] * px_size) ** 2))
    theta_out_array = np.arctan(np.abs(detector_grid) / sample_detector_distance)  # Array of all possible exit
    q_out_array = 2 * k * np.sin(theta_out_array)  # Array of exit wave vectors, generated by the exit angles
    if q_max is None:
        delta_q = np.linspace(np.finfo(float).eps,
                              np.max(q_out_array), LUT_resolution)
    else:
        delta_q = np.linspace(np.finfo(float).eps, q_max, LUT_resolution)  # Array of possible
    # scattering q values
    scattering_pdf = form_factor.polyP(delta_q, scattering_function, Rg_array, distribution)
    scattering_pdf = scattering_pdf / (np.sum(scattering_pdf) * np.diff(delta_q)[0])
    # Final Intensity profile (i.e. photon count on each detector position)
    Intensity_profile = np.zeros([pixel_number, pixel_number])
    # Calculate the continuous inverse cumulative distribution function (inv_cdf_interp) by an interpolation.
    inv_cdf_interp = generate_inverse_cdf(scattering_pdf, delta_q)
    # Num batches: How many scattering events are needed to be simulated.
    num_batches = int(np.round(photon_number / sampling_size))
    # How many workers (CPU threads) are available to work with
    num_workers = mp.cpu_count()
    # Determine the range of batches each worker will handle
    batching = 100
    batch_number = int(num_batches / batching)
    batches = (np.ones(batch_number) * batching).astype(int)
    seed_seq = SeedSequence(1)
    seeds = seed_seq.spawn(batch_number)
    # Run the simulation
    args = [(batches[i], seeds[i], sampling_size, slit1_radius[0], slit1_radius[1], slit2_radius[0], slit2_radius[1],
             slit_distance, q_out_array, delta_q, inv_cdf_interp,
             sample_detector_distance, k, pixel_number, pixel_centers, px_size) for i in range(batch_number)]
    with mp.Pool(processes=num_workers) as pool:
        results = []
        for result in tqdm(pool.istarmap(_simulation_process_2D, args), total=batch_number):
            results.append(result)
    for result in results:
        Intensity_profile += result.astype(float)
    # Replace any zeroes with eps to avoid division by zero
    Intensity_profile[Intensity_profile == 0] = np.finfo(np.float64).eps
    return Intensity_profile, q_out_array


def example_simulation():
    """
    This function performs an example Monte Carlo simulation for a "Guinier form factor".

    Parameters:
    wavelength_ (float): X-ray wavelength in centimeters.
    slit1_radius_ (list): Half-length of slit 1 in centimeters.
    slit2_radius_ (list): Half-length of slit 2 in centimeters.
    sample_detector_distance_ (float): Distance between the sample and the detector.
    slit_distance_ (float): Distance between the slit.
    detector_length_ (float): Length of the detector in centimeters.
    detector_center_ (list): Center of the detector in centimeters.
    pixel_number_ (int): Number of pixels on the detector.
    photon_number_ (int): Number of photons (simulation runs).
    LUT_resolution_ (int): Resolution of the initial scattering probability distribution LUT.
    sampling_size_ (int): Amount of scattering incidents to be generated at once at each simulation run.
    Rg_average_ (float): Average radius of gyration, in centimeters.
    Rg_variance_ (float): Relative variance of the radius of gyration, unitless.

    Returns:
    tuple: A tuple containing the final Intensity profile and the array of exit wave vectors.
    """

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

    # Save the intensity profile and the q-table as TIF files
    tif.imwrite(output_path + f"dist100_intensity.tif",
                det_2d, photometric='minisblack')
    # Here the q values are normalized by 1e-7 to change their units to nm^-1.
    tif.imwrite(output_path + f"dist100_qtable.tif",
                qtable * 1e-7, photometric='minisblack')
    # Flatten the intensity profile of the scattering. Here the Q values are normalized to be in nm^-1 (multiplication
    # by 1e-7).
    xq, yi = flatten_intensity(qtable * 1e-7, det_2d, px_area=75e-4 ** 2, wavelength=0.154,
                               distance=sample_detector_distance_)

    # For compatibility with the PyFAI library, a corresponding poni file (calibration file) can be generated as follows
    px_size_ = ((detector_length_ * 1e4) / pixel_number_) / 100
    poni1 = detector_center_[0] / 100  # meters
    poni2 = detector_center_[1] / 100  # meters
    with open(output_path + f"poni100.poni", 'w') as file:
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

    # Example for plotting the 1D data
    q_range = (xq >= 0.05) & (xq <= 1)
    plt.scatter(xq[q_range], yi[q_range], color='cyan', s=5, label='simulated', zorder=1)
    plt.plot(xq[q_range], yi[q_range][0] * form_factor.guinier_ff(xq[q_range], 4),
             color='black', ls='dashed', lw=2, label='form factor', zorder=3)
    plt.xlabel('q (nm^-1)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('1D data')
    plt.legend(frameon=False)
    plt.yscale('log')
    plt.xlim(xq[q_range][0], xq[q_range][-1])
    plt.show()
    #Plotting the 2D data
    f = plt.figure(figsize=(6.2, 5.6))
    ax = f.add_axes([0.17, 0.02, 0.72, 0.79])
    axcolor = f.add_axes([0.90, 0.02, 0.03, 0.79])
    im = ax.matshow(det_2d, cmap=cm.viridis, norm=LogNorm(vmin=np.min(det_2d), vmax=np.max(det_2d)))
    f.colorbar(im, cax=axcolor, format="$%.2f$")
    f.show()
    plt.show()
    return det_2d, qtable
