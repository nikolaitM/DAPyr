"""test"""

# author HMS 06/2025
# obs when using the local particle filter (or other nongaussian DA method)

# sampled errors are controlled by the used_obs_err and used_obs_err_params parameters
# in the experiment class. to sample obs using a different error distribution,
# all that needs to happen is a case needs to be added to sample_errors corresponding
# to the distribution of interest. the case number and parameters then are passed to the above
# parameters when constructing an Expt.

# to assimilate obs assuming a given likelihood, add the likelihood to get_likelihood.
# this involves defining the entire likelihood function including its parameters
# and then also adding a case to return the function with the provided parameters set.
# the corresponding parameters in the Expt class are prescribed_obs_err and
# prescribed_obs_err_params.

import numpy as np
from functools import partial

GAUSSIAN = 0
STATE_DEP_GAUSSIAN = 1


def sample_errors(
    states: np.ndarray,
    true_obs_err_dist: int,
    true_obs_err_params: dict,
    rng: np.random.Generator,
):
    """Generate samples of observation error according to a prescribed distribution.

    Parameters
    ----------
    states : np.array(M, T)
        M-dimensional model states valid at T different times that each observation error will correspond to.

    true_obs_err_dist : int
        Flag for specifying what distribution observation errors should be sampled from.

    true_obs_err_params : dict
        Parameters of the distribution specified by true_obs_err_dist.

    rng : numpy.random.Generator
        Random number generator to sample errors with.

    Returns
    ---------
    errors : array(M, T)
        Sampled observation errors.

    """
    # GAUSSIAN = 0
    # STATE_DEP_GAUSSIAN = 1

    errors = -999 * np.zeros_like(states)

    match used_obs_error:
        case 0:
            try:
                mu, sigma = params["mu"], params["sigma"]
            except KeyError:
                raise KeyError(f"Parameters mu and sigma not provided in {params}")
            errors = rng.normal(mu, sigma, size=states.shape)
        case 1:
            try:
                mu1 = params["mu1"]
                mu2 = params["mu2"]
                sigma1 = params["sigma1"]
                sigma2 = params["sigma2"]
                threshold = params["threshold"]
            except KeyError:
                raise KeyError(
                    f"Parameters mu1, sigma1, mu2, sigma2, and threshold not provided in {params}"
                )

            errs1 = rng.normal(mu1, sigma1, states.shape)
            errs2 = rng.normal(mu2, sigma2, states.shape)

            errors = np.where(states < threshold, errs1, errs2)

    return errors


def get_likelihood(assumed_obs_err_dist: int, assumed_obs_err_params: dict):
    """Create likelihood fucntion corresponding to a prescribed observation error distribution. If a Gaussian observation error is assumed, this also determines the variance used by the ensemble Kalman filter.

    Parameters
    ----------
    assumed_obs_err_dist : int
        Flag for specifying what distribution observation errors should be assumed to come from.

    assumed_obs_err_params : dict
        Parameters of the distribution specified by assumed_obs_err_dist.

    Returns
    ---------
    L : functools.partial
        Callable likelihood function L(y, hx). The first argument is a vector of scalar observations, and the second argument is the model state projected into observation space.

    """

    def gaussian_l(y, hx, mu, sigma):
        d = (y - hx - mu) ** 2 / (2 * sigma**2)
        d -= np.min(d, axis=-1)[:, None]
        return np.exp(-d)

    def state_dep_gaussian_l(y, hx, mu1, mu2, sigma1, sigma2, threshold):
        l_low = np.exp(-((y - hx - mu1) ** 2 / (2 * sigma1**2)))
        l_high = np.exp(-((y - hx - mu2) ** 2 / (2 * sigma2**2)))
        return np.where(hx < threshold, l_low, l_high)

    match prescribed_obs_error:
        case 0:
            try:
                return partial(gaussian_l, mu=params["mu"], sigma=params["sigma"])
            except KeyError:
                raise KeyError(f"Parameters mu and sigma not provided in {params}")
        case 1:
            try:
                return partial(
                    state_dep_gaussian_l,
                    mu1=params["mu1"],
                    sigma1=params["sigma1"],
                    mu2=params["mu2"],
                    sigma2=params["sigma2"],
                    threshold=params["threshold"],
                )
            except KeyError:
                raise KeyError(
                    f"Parameters mu1, sigma1, mu2, sigma2, and threshold not provided in {params}"
                )
