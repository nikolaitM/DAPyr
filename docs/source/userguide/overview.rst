Overview
=========

DAPyr's functionality can be divided into two core features: the :doc:`Expt <../refguide/expt>` class and the :doc:`runDA <../refguide/runda>` function. The :py:class:`~DAPyr.Expt` class is used to configure and initialize an instance of a data assimilation run through, while the :py:func:`~DAPyr.runDA` function actually runs the simulation and performs the data assimilation based on the configurations in your Expt instance.

Basic usage
-------------

The :py:class:`~DAPyr.Expt` sets up the experiment you want to run. Through this class, you can configure things like the type of model you wish to run, how long to run it for, and what data assimilation method to use. 

The :py:class:`~DAPyr.Expt` class takes in a string for the name of the experiment, and an optional `params` dictionary containing any settings you with to change. Initializing an experiment without providing a `params` dictionary sets all configurable variables to their default values.

Below, we will create an experiment call "Basic_Expt". We will run a 20 member ensemble 200 time steps using Lorenz 96 as our toy model.

.. code-block:: python

      import DAPyr as dap
      expt  = dap.Expt('Basic_Expt', {'expt_flag': 0, "Ne": 20, 'model_flag':1, 'T': 200, 'saveEns': 1, 'saveEnsMean': 1})


For a summary of all configured parameters in an experiment, simply print the experiment instance using `print(expt)`.


To get the results of a single parameter in the experiment use the :py:meth:`~DAPyr.Expt.getParam` method to retrieve it. 


.. code-block:: python

      print('Getting the model_flag used')
      expt.getParam('model_flag')
      Getting the model_flag used
      1


To change a parameter in the experimental setup, use the :py:meth:`~DAPyr.Expt.modExpt` method. 

.. code-block:: python

      #Lets change the DA method from EnSRF to LPF
      print('This is the first observation before experiment modification: {}'.format(expt.getParam('Y')[0, 0, 0]))
      #Now let's modify the experiment
      expt.modExpt({'expt_flag': 1})
      print('The observation is the same after modification: {}'.format(expt.getParam('Y')[0, 0, 0]))
      #Now let's modify something that will change the state, such as the observation error
      expt.modExpt({'sig_y': 1.5})
      print('The observation wlll be different after modification: {}'.format(expt.getParam('Y')[0, 0, 0]))

Once you are happy with your experiment, it is time to run the actual data assimilation cycles. To do this, use the :py:func:`~DAPyr.runDA` function. It simply takes in an :py:class:`~DAPyr.Expt` instance, and runs through all specifications of the experiment.

.. code-block:: python

      dap.runDA(expt)

From here, there are a number of outputs you can reference.

* `Expt.rmse`: The posterior RMSE over the experiment 
* `Expt.prior_rmse`: The prior RMSE over the experiment 
* `Expt.spread`: A (T x 2) matrix where [:, 0] are the prior ensemble spreads, and [:, 1] are the posterior ensemble spreads
* `Expt.x_ens`: If "saveEns" is set to 1 in configuration, the full posterior ensemble states are stored 
* `Expt.x_ensmean`: If "saveEnsMean" is set to 1 in configuration, the full posterior ensemble means are stored
* `Expt.x_fore_ens`: If "saveForecastEns" is set to 1 in configuration, the full prior ensemble states are stored