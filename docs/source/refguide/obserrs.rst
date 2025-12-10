Specifying Observation Errors
==============================

DAPyr can specify both a "true" observation error distribution, which is used when generating observations from a nature run, and an "assumed" observation error distribution, which is used when observations are assimilated. Currently, two different forms of the observation error are supported; to add a new one, add an additional section to :py:func:`~dapyr.obs_errors.sample_errors` or :py:func:`~dapyr.obs_errors.get_likelihood`. Observation error distributions are set by means of the *true_obs_err_dist* and *assumed_obs_err_dist* flags when initializing experiments, and the parameters for each distribution are specified in *true_obs_err_params* and *assumed_obs_err_params*, respectively.


Setting the True Observation Error Distribution
-----------------------------------------------
.. autofunction:: DAPyr.OBS_ERRORS.sample_errors


Setting the Assumed Observation Error Distribution
--------------------------------------------------
.. autofunction:: DAPyr.OBS_ERRORS.get_likelihood



Currently Supported Observation Error Distributions
---------------------------------------------------

.. _obs-err-supported:
.. list-table::
   :header-rows: 1

   * - Name
     - Distribution Flag
     - Parameters
     - Description
   * - Gaussian
     - 0
     - 'mu', 'sigma'
     - Gaussian distribution with mean 'mu' and standard deviation 'sigma'.
   * - State-Dependent Gaussian
     - 1
     - 'mu1', 'sigma1', 'mu2', 'sigma2', 'threshold'
     -
       | If the model state `x` < 'threshold', draws from a Gaussian distribution with mean
       | 'mu1' and standard deviation 'sigma1'. Otherwise, draws from a Gaussian
       | distribution with mean 'mu2' and standard deviation 'sigma2'.
