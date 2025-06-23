Installation
=============

To use DAPyr, first install it using pip:

.. code-block:: console

   $ pip install DAPyr


Troubleshooting
---------------

Failure to Build `numbalsoda`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A common issue with installation involves a failure to build the `numbalsoda` package, which requires a working Fortran compilier to build. 

.. code-block:: console

   $ pip insall DAPyr
   .....
   .....
   .....
         Not searching for unused variables given on the command line.
      -- The Fortran compiler identification is unknown
      -- The CXX compiler identification is AppleClang 16.0.0.16000026
      CMake Error at CMakeLists.txt:3 (project):
        No CMAKE_Fortran_COMPILER could be found.
   .....
   .....
   ERROR: Failed building wheel for numbalsoda
   Failed to build numbalsoda
   ERROR: Failed to build installable wheels for some pyproject.toml based projects (numbalsoda)


To solve this issue, first verify you have a Fortran compiler installed on your system, then manually export the path to your compilier

.. code-block:: console

   $ export FC=<path to a Fortran compiler>


Alternatively, if you are using a environment manager such as *conda*, you may install `numbalsoda` first using *conda*, then install DAPyr using `pip`

.. code-block:: console
   
   $ conda install -c conda-forge numbalsoda
   $ pip install DAPyr