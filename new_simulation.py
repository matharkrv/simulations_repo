import numpy as np
from scipy.interpolate import interp1d
from numba import njit, prange

import form_factor_methods as form_factor
from scipy.integrate import cumulative_trapezoid
import multiprocessing as mp
import multiprocessing.pool as mpp
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile as tif
from numpy.random import SeedSequence
import pandas as pd
from numpy import cos, sin
from istarmap import istarmap

# Import istarmap for simultaneous tracking of the multiprocess and its execution
mpp.Pool.istarmap = istarmap


def flatten_intensity(r_array, count_array, px_area, wavelength, distance, normalization=1, return_unique=True,
                      q_min=None, q_max=None):
    """
        Flatten and normalize intensity data from 2D arrays to 1D arrays.
        This function takes 2D arrays of radial distances and corresponding intensity counts,
        flattens them, and applies normalization and optional filtering. It can return either
        unique radial values with averaged intensities or the full flattened arrays.

        Parameters:
        -----------
        r_array : numpy.ndarray
            2D array of radial distances.
        count_array : numpy.ndarray
            2D array of intensity counts corresponding to r_array.
        px_area : float
            Area of a single pixel.
        wavelength : float
            Wavelength of the radiation used.
        distance : float
            Sample-to-detector distance.
        normalization : float, optional
            Normalization factor for r_array (default is 1).
        return_unique : bool, optional
            If True, return unique radial values with averaged intensities (default is True).
        q_min : float, optional
            Minimum q value for filtering (only applied if q_max is also provided).
        q_max : float, optional
            Maximum q value for filtering.

        Returns:
        --------
        tuple
            If return_unique is True:
                (unique_R, normalized_average_counts) where
                unique_R is an array of unique radial values, and
                normalized_average_counts is an array of corresponding normalized average intensities.
            If return_unique is False:
                (r_flattened, normalized_intensity_flattened) where
                r_flattened is the flattened and sorted radial array, and
                normalized_intensity_flattened is the corresponding normalized intensity array.
        """
    # Flatten the arrays using the logic from flatten_intensity
    sorted_idx = np.argsort(r_array.flatten())
    r_flattened = r_array.flatten()[sorted_idx] * normalization
    intensity_flattened = count_array.flatten()[sorted_idx]
    if return_unique:
        # Create a DataFrame
        df = pd.DataFrame({'R': r_flattened, 'count': intensity_flattened})
        # Group by the 'R' values and compute the mean of the 'count' values
        grouped_df = df.groupby('R', as_index=False)['count'].mean()
        # Extract the unique R values and their corresponding average counts
        unique_R = grouped_df['R'].values
        average_counts = grouped_df['count'].values
        # Compute the normalization factor
        k = 2 * np.pi / wavelength
        norm_factor = np.pi * distance ** 2 * unique_R / (2 * k ** 2 * px_area * (1 - (unique_R / (2 * k)) ** 2) ** 2)
        # Mask the data based on the given q_min and q_max if applicable
        if q_min is not None:
            mask = (unique_R >= q_min) & (unique_R <= q_max) & (average_counts > 1e-8)
        else:
            mask = (average_counts > 1e-8)
        return unique_R[mask], (average_counts * norm_factor)[mask]
    else:
        # Return the flattened and sorted arrays directly
        norm_factor = ((wavelength / 10) ** 2 * distance ** 2 / (4 * np.pi ** 2 * px_area) /
                       (1 - (wavelength / 10) ** 2 / 8 / np.pi ** 2 * r_flattened ** 2) ** 3)
        mask = (r_flattened >= q_min) & (r_flattened <= q_max)
        return r_flattened[mask], (intensity_flattened * 2 * np.pi * r_flattened * normalization * norm_factor)[mask]


def sample_from_cdf(cdf, num_samples=1, continuous=True):
    """
    Generate samples from an inverse cumulative distribution function (CDF).

    This function takes a CDF and generates random samples from it. It can handle
    both continuous and discrete CDFs.

    Parameters:
    -----------
    cdf : callable or array-like
        The inverse cumulative distribution function. If continuous, it should be a callable
        function (i.e, a Scipy interpolation). If discrete, it should be an array-like object representing the CDF.
    num_samples : int, optional
        The number of samples to generate (default is 1).
    continuous : bool, optional
        Whether the CDF is continuous (True) or discrete (False) (default is True).

    Returns:
    --------
    numpy.ndarray
        An array of samples drawn from the given CDF.
    """
    if continuous:
        return cdf(np.random.uniform(0, 1, size=num_samples))
    else:
        return cdf[np.random.choice(len(cdf), num_samples, replace=False)]


def sample_from_slits(random_seed, sl1_x, sl1_y, sl2_x, sl2_y, slit_distance, sampling):
    """
    This function generates random incidents from two slits and returns the exit angle theta, the azimuth angle chi,
    and the coordinates of the exit points for the two slits.

    Parameters:
    - random_seed: A numpy random number generator seed.
    - sl1_x, sl1_y: The half-lengths of slit 1 in the x and y directions, respectively.
    - sl2_x, sl2_y: The half-lengths of slit 2 in the x and y directions, respectively.
    - slit_distance: The distance between the two slits.
    - sampling: The number of samples to generate.

    Returns:
    - theta: An array of exit angles, in radians, for the sampled particles.
    - chi: An array of azimuth angles, in radians, for the sampled particles.
    - s2x, s2y: Arrays of x and y coordinates, respectively, for the exit points of the particles from slit 2.
    """
    # Generate random samples from the two slits
    s1x = random_seed.uniform(-sl1_x, sl1_x, sampling)
    s1y = random_seed.uniform(-sl1_y, sl1_y, sampling)
    s2x = random_seed.uniform(-sl2_x, sl2_x, sampling)
    s2y = random_seed.uniform(-sl2_y, sl2_y, sampling)

    # Calculate the exit angle and azimuth angle for each sample
    theta = np.arctan(np.sqrt((s2y - s1y) ** 2 + (s2x - s1x) ** 2) / slit_distance)
    chi = np.mod(np.arctan2(s2y - s1y, s2x - s1x), 2 * np.pi)

    return np.abs(theta), chi, s2x, s2y


def gaussian_function(x, a, b):
    # Implementation of a Gaussian function.
    exponent = np.exp(-0.5 * ((x - a) / b) ** 2) * (1 / (b * np.sqrt(2 * np.pi)))
    return exponent


def ones_function(x, R):
    # Implementation of an 'ones' function (f(q) = 1), used for testing.
    return np.ones_like(x * R)


def generate_pdf(scattering_function, q_sampling):
    # Calculate the PDF (probability distribution function) of the given scattering signal.
    pdf = scattering_function(q_sampling)
    return pdf / np.sum(pdf * np.diff(q_sampling)[0])


def generate_inverse_cdf(pdf, samples, continuous=True, resolution=int(1e6)):
    """
       This function generates an inverse cumulative distribution function (CDF-1)
       from a probability density function (PDF).

       Parameters:
       - pdf: A 1D array representing the probability density function.
       - samples: A 1D array representing the sample points for the CDF (q range).
       - continuous: A boolean flag indicating whether the CDF should be continuous or discrete.
       - resolution: An integer specifying the number of points in the final CDF. Only applicable in the discrete case.
       ```
       """
    # Calculate the CDF using a cumulative trapezoid integration
    cdf_values = cumulative_trapezoid(pdf, samples, initial=np.min(samples))
    # Calculate the inverse CDF by interpolating q as a function of the calculated CDF
    inv_cdf_interp = interp1d(cdf_values, samples, bounds_error=False, fill_value=(samples[0], samples[-1]))
    if continuous is False:
        # If the function is chosen to be returned as discrete, returns the values of the interpolated function within
        # the chosen resolution.
        return inv_cdf_interp(np.linspace(0, 1, resolution))
    else:
        # Else, return the interpolated function (continuous case)
        return inv_cdf_interp


def clean_data(data, threshold):
    """
    Clean the input data by replacing values below a threshold with the previous value.

    This function iterates through the input data and replaces any value that is
    less than a threshold times the previous value with the previous value itself.

    Parameters:
    -----------
    data : array-like
        The input data to be cleaned.
    threshold : float
        The threshold value used for comparison. If a value is less than
        threshold times the previous value, it is replaced.

    Returns:
    --------
    numpy.ndarray
        A new array containing the cleaned data, where values below the threshold
        have been replaced with the previous value.
    """
    cleaned_data = np.copy(data)
    for index in range(len(data) - 1):
        if cleaned_data[index + 1] < threshold * cleaned_data[index]:
            cleaned_data[index + 1] = cleaned_data[index]

    return cleaned_data


def simulation_process_2D(*args):
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


def IqMC_2D(wavelength, slit1_radius, slit2_radius, sample_detector_distance, slit_distance, detector_length,
            detector_center, pixel_number, photon_number, LUT_resolution, sampling_size,
            scattering_function, Rg_array, distribution, q_max=None):
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
    Distribution (array): Distribution for the array of Rgs.
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
        for result in tqdm(pool.istarmap(simulation_process_2D, args), total=batch_number):
            results.append(result)
    for result in results:
        Intensity_profile += result.astype(float)
    # Replace any zeroes with eps to avoid division by zero
    Intensity_profile[Intensity_profile == 0] = np.finfo(np.float64).eps
    return Intensity_profile, q_out_array


if __name__ == "__main__":
    wavelength_ = 1.54e-8  # X-ray wavelength in centimeters
    slit1_radius_ = [0.02, 0.02]  # Half-length of slit 1 in centimeters
    slit2_radius_ = [0.01, 0.01]  # Half-length of slit 2 in centimeters
    sample_detector_distance_ = 200  # Distance between the sample and the detector
    slit_distance_ = 200  # Distance between the slit
    detector_length_ = 7.5  # Length of the detector in centimeters
    detector_center_ = [3, 3]  # Center of the detector in centimeters
    pixel_number_ = 1000  # Number of pixels on the detector
    photon_number_ = int(1e8)  # Number of photons (simulation runs)
    LUT_resolution_ = int(1e4)  # Resolution of the initial scattering probability distribution LUT
    sampling_size_ = int(1e4)  # Amount of scattering incidents to be generated at once at each simulation run
    Rg_average_ = 4e-7  # Average radius of gyration, in centimeters
    Rg_variance_ = 0.0001  # Relative variance of the radius of gyration, unitless

    # Generate the distribution of Rgs
    rg_array, distribution = form_factor.generate_gaussian_distribution(Rg_average_,
                                                                        np.sqrt(Rg_variance_) * Rg_average_)
    # Set an output path
    output_path = "/Users/Mathar/measurements/Simulations/test/"

    # Run the simulation with the provided parameters
    det_2d, qtable = IqMC_2D(wavelength_, slit1_radius_, slit2_radius_, sample_detector_distance_, slit_distance_,
                             detector_length_,
                             detector_center_, pixel_number_, photon_number_, LUT_resolution_,
                             sampling_size_, form_factor.guinier_ff, rg_array, distribution)

    # Save the intensity profile and the q-table as TIF files
    tif.imwrite(output_path + f"intensity.tif",
                det_2d, photometric='minisblack')
    # Here the q values are normalized by 1e-7 to change their units to nm^-1.
    tif.imwrite(output_path + f"q_table.tif",
                qtable * 1e-7, photometric='minisblack')
    # Flatten the intensity profile of the scattering. Here the Q values are normalized to be in nm^-1 (multiplication
    # by 1e-7).
    xq, yi = flatten_intensity(qtable * 1e-7, det_2d, px_area=75e-4 ** 2, wavelength=0.154,
                               distance=detector_length_)

    # For compatibility with the PyFAI library, a corresponding poni file (calibration file) can be generated as follows
    px_size_ = ((detector_length_ * 1e4) / pixel_number_) / 100
    poni1 = detector_center_[0] / 100  # meters
    poni2 = detector_center_[1] / 100  # meters
    with open(output_path + f"poni.poni", 'w') as file:
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
    plt.plot(xq[q_range], yi[q_range][0]*form_factor.guinier_ff(xq[q_range], 4),
             color='black', ls='dashed', lw=2, label='form factor', zorder=3)
    plt.xlabel('q (nm^-1)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('1D data')
    plt.legend(frameon=False)
    plt.yscale('log')
    plt.xlim(xq[q_range][0], xq[q_range][-1])
    plt.show()

