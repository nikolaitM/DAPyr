Configuring an Experiment
==============================

.. autoclass:: DAPyr::Expt
      :class-doc-from: class
      :members:

Configurable Paramters
-----------------------

basicParams
^^^^^^^^^^^^
Parameters in the *basicParams* attribute store information that define the basic structure of the experiment, such as how long to run the experiment for, how many ensemble members to run, and what data assimilation method to use. 

.. list-table::
   :header-rows: 1

   * - Paramter
     - Default
     - Description
   * - `T`
     - 100
     - Total number of time steps in experiment
   * - `dt`
     - Model Specific
     - Model time step increment
   * - `Ne`
     - 10
     - Number of ensemble members
   * - `expt_flag`
     - 0
     - Flag to select which Data Assimilation method to use
            | 0: EnSRF 
            | 1: Local Particle Filter  
            | 2: No Data Assimilation 
   * - `seed`
     - -1
     - Seed for random number generator. Set to -1 to turn off.


obsParams
^^^^^^^^^^^^^^^

The *obsParams* attribute stores all parameters relating to the observations to assimilation, from what measurement operator to use to prescribing observation error. Additionally, the *obsParam* attribute stores parameters relating to coniguring the data assimilation methods.

.. list-table::
   :header-rows: 1

   * - Paramter
     - Default
     - Description
   * - `h_flag`
     - 0
     - Flag to select measurement operator to use
               | 0: Linear (Hx = x) 
               | 1: Quadratic (Hx = x^2)
               | 2: Logarithmic (Hx = log(abs(x)))
   * - `sig_y`
     - 1.0
     - Observations Standard Deviation
   * - `tau`
     - 1
     - Number of cycles between data assimilation cycles
   * - `obf`
     - 1
     - Observation spatial frequency. Spacing between assimilated observations
   * - `obb`
     - 0
     - Observation buffer. Initial number of observations to skip, used to simulate large spatial gaps in observation coverage.
   * - `localize`
     - 1
     - Whether to turn on localization. 0 is off, 1 is on.
   * - `roi`
     - 0.005
     - Localization radius of influence
   * - `true_obs_err_dist`
     - 0
     - Flag to select observation error distribution to generate observations with (see :doc:`sample_errors <../refguide/obserrs>` )
               | 0: Gaussian 
               | 1: State-dependent Gaussian
   * - `true_obs_err_params`
     - {'mu': 0, 'sigma' : 1}
     - Parameters for observation error distribution associated with *true_obs_err_dist*
   * - `assumed_obs_err_dist`
     - 0
     - Flag to specify likelihood functions for observations given model states. (see :doc:`get_likelihood <../refguide/obserrs>` )
               | 0: Gaussian 
               | 1: State-dependent Gaussian
   * - `assumed_obs_err_params`
     - {'mu': 0, 'sigma' : 1}
     - Parameters for observation error distribution associated with *assumed_obs_err_dist*


**Kalman Filter Parameters**

Below are all parameters directly configuring the ensemble square root filter.

.. list-table::
   :header-rows: 1

   * - Paramter
     - Default
     - Description
   * - `gamma`
     - 0.00
     - Inflation parameter for Relaxation to Prior Spread (RTPS).

**Local Particle Filter Parameters**

Below are all parameters directly configuring the local particle filter.
  
.. list-table::
   :header-rows: 1
   
   * - Paramter
     - Default
     - Description
   * - `mixing_gamma`
     - 0.3
     - Mixing coefficient between prior resampled particles and current particles
   * - `kddm_flag`
     - 0
     - Flag to turn on kernel density estimation step. 0 is off, 1 is on
   * - `min_res`
     - 0.0
     - Minimum residual for beta inflation
   * - `maxiter`
     - 1
     - Maximum number of incremental steps of LPF updates to make
   * - `Nt_eff`
     - 0.4
     - Effective ensemble size

modelParams
^^^^^^^^^^^^^

The *modelParams* attribute stores parameters configuring the chosen model for the experimenting. Inside the *modelParams* dictionary is another dictionary, `model_params`, which separately stores the parameters for tuning model physics.

.. list-table::
   :header-rows: 1

   * - Paramter
     - Default
     - Description
   * - `model_flag`
     - 0
     - Flag to select model to use
         | 0 : Lorenz (1963)
         | 1 : Lorenz (1996)
         | 2 : Lorenz (2005)
   * - `model_params`
     - See :ref:`Model Parameters <modelparams>`
     - Configuration of parameters that drive each Lorenz model.

.. _modelparams:

**Model Parameters**

.. list-table::
   :header-rows: 1

   * - Model
     - Paramter
     - Default
   * - Lorenz (1963)
     - s
     - 10
   * - 
     - r
     - 28
   * - 
     - b
     - 8/3
   * - Lorenz (1996)
     - F
     - 8
   * - Lorenz (2005)
     - l05_F
     - 15
   * - 
     - l05_Fe
     - 15
   * - 
     - l05_K
     - 32
   * - 
     - l05_I
     - 12
   * - 
     - l05_b
     - 10.0
   * - 
     - l05_c
     - 2.5

miscParams
^^^^^^^^^^^

The *miscParams* attribute stores all parameters relating to experiment output, as well as other miscellaneous features.

.. list-table::
   :header-rows: 1

   * - Paramter
     - Default
     - Description
   * - `output_dir`
     - './'
     - The default file path for saving the experiment
   * - `numPool`
     - 8
     - Number of subprocesses to spawn when conducting model integration and model spinup
   * - `saveEns`
     - 0
     - Should the analysis ensemble state be stored each time step in `x_ens`. 0 is False, 1 is True.
   * - `saveEnsMean`
     - 1
     - Should the analysis ensemble mean state be stored each time in `x_ensmean`. 0 is False, 1 is True.
   * - `saveForecastEns`
     - 0
     - Should the forecast ensemble stae be stored each time step in `x_fore_ens`. 0 is False, 1 is True.
   
**Singular Vector Parameters**

.. list-table::
   :header-rows: 1

   * - Paramter
     - Default
     - Description
   * - `doSV``
     - 0
     - Should singular vectors be calculated according to `Whitaker and Hamill (2002) <https://journals.ametsoc.org/view/journals/mwre/131/8/_2559.1.xml>`_
   * - `stepSV`
     - 1
     - Number of time steps to skip between SV calculations
   * - `forecastSV`
     - 4
     - Optimization time interval
   * - outputSV
     - `output_dir`
     - Output directory for SV calculation files
   * - storeCovar
     - 0 
     - Flag to determine whether to store the analysis and forecast SV ensemble states. 0 is False, 1 is True




