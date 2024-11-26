import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from typing import Tuple, Optional, Callable, Union, List


def flatten_intensity(r_array: np.ndarray,
                      count_array: np.ndarray,
                      px_area: float,
                      wavelength: float,
                      distance: float,
                      normalization: float = 1,
                      return_unique: bool = True,
                      q_min: Optional[float] = None,
                      q_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
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


def sample_from_cdf(cdf: Union[Callable, List[float], np.ndarray],
                    num_samples: int = 1,
                    continuous: bool = True) -> np.ndarray:
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


def sample_from_slits(random_seed: np.random.Generator,
                      sl1_x: float,
                      sl1_y: float,
                      sl2_x: float,
                      sl2_y: float,
                      slit_distance: float,
                      sampling: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def generate_pdf(scattering_function: Callable,
                 q_sampling: np.ndarray) -> np.ndarray:
    # Calculate the PDF (probability distribution function) of the given scattering signal.
    pdf = scattering_function(q_sampling)
    return pdf / np.sum(pdf * np.diff(q_sampling)[0])


def generate_inverse_cdf(pdf: np.ndarray,
                         samples: np.ndarray,
                         continuous: bool = True,
                         resolution: int = int(1e6)) -> Union[np.ndarray, Callable]:
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


def clean_data(data: np.ndarray,
               threshold: float) -> np.ndarray:
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


def bin_signal(x, y, x_binned):
    """
    Bin the signal y(x) according to the specified x_binned array.

    Parameters:
        x (array-like): The input x values of the signal.
        y (array-like): The input y values of the signal corresponding to x.
        x_binned (array-like): The bin edges to be used for binning x and y.

    Returns:
        y_binned (numpy array): The binned y values corresponding to x_binned.
    """
    # Step 1: Initialize y_binned list
    y_binned = []

    # Step 2: Compute the mean of y values in each bin range
    for i in range(len(x_binned) - 1):
        mask = (x >= x_binned[i]) & (x < x_binned[i + 1])
        if np.any(mask):
            y_binned.append(y[mask].mean())
        else:
            y_binned.append(0)  # Or use np.nan or another placeholder if no data falls into the bin

    # Step 3: Add a placeholder for the last bin
    y_binned.append(0)  # Placeholder for the last bin (no data falls here)
    y_binned[-1] = y_binned[-2]
    return np.array(y_binned)
