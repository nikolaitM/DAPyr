Calling Data Assimilation Methods Independently
====================================================

Ensemble Square Root Filter
----------------------------

.. autofunction:: DAPyr.DA.EnSRF_update

\*\*kwargs Parameters
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Property
     - Description
   * - `gamma`
     - Inflation parameter for RTPP and RTPS inflation methods
   * - `infs`
     - Array of size (Nx) containing initial estimates of inflation means.
   * - `infs_y`
     - Array of size (Ny) containing initial estimates of inflation means in obs space.
   * - `var_infs`
     - Array of size (Nx) containing initial estimates of inflation variances.
   * - `var_infs_y`
     - Array of size (Nx) containing initial estimates of inflation variances in obs space.



Local Particle Filter 
-----------------------

.. autofunction:: DAPyr.DA.lpf_update
