# DAPyr: (D)ata (A)ssimilation (Py)ton (R)elease

DAPyr (pronounced "dapper") is a Python package for running, testing, and experimenting with data assimilation methods using Lorenz models. DAPyr supports quick configuration and running of experiments, allowing one to setup and run data assimilation cycles in as little as two lines. In addition, DAPyr utilizes [numbalsoda](https://github.com/Nicholaswogan/numbalsoda/tree/main) and [numba](https://numba.pydata.org/) for model integration, allowing for faster runtimes.

DAPyr currently supports the following functions:

### Models:
1. Lorenz 1963
2. Lorenz 1996
3. Lorenz 2005

### Data Assimilation Methods:
1. Ensemble Square Root Filter [(Whitake and Hamill, 2002)](https://journals.ametsoc.org/view/journals/mwre/130/7/1520-0493_2002_130_1913_edawpo_2.0.co_2.xml)
2. Localized Particle Filter [(Poterjoy et al., 2022)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4328)

### Measurement Operators:
1. Linear
2. Quadratic
3. Logarithmic

## Installation

Pip:
```
python -m pip install DAPyr
```

## Basic Usage

```python
import DAPyr as dap
import numpy as np

# Use the Expt class to initialize an new experiment and calculate all necessary initial states
# Experiment will use Lorenz 63 (model_flag: 0), run for 300 time steps, and use the EnSRF method (expt_flag: 0)
expt = dap.Expt('Basic_L63', {'model_flag': 0, 'expt_flag': 0, 'T': 300})

# Run the experiment using the runDA function
dap.runDA(expt)

# Access RMSE and spread using the rmse and spread attributes
print('Avg. RMSE: {:.3f}'.format(np.mean(expt.rmse)))
print('Avg. Prior Spread: {:.3f}'.format(np.mean(expt.spread[:, 0])))
print('Avg. Posteriod Spread: {:.3f}'.format(np.mean(expt.spread[:, 1])))

#Save the experiment to filesystem using saveExpt, the file will be called it's experiment name
dap.saveExpt('./', expt)

#Load in previous experiments using loadExpt
expt2 = dap.loadExpt('./Basic_L63.expt')

```

For more examples, see `tutorial.ipynb` in `src`.