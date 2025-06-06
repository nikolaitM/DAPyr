import sys
sys.path.insert(0,'../src/')
import unittest
import DAPpyr as dap
from unittest.mock import patch
import numpy as np
import copy
import DAPpyr.MISC
import DAPpyr.Exceptions
import matplotlib.pyplot as plt

mod_bias = [-0.5, -0.25, 0.0, 0.25, 0.5]
obs_bias = [-2.0, -1.0, 0.0, 1.0, 2.0]

for x in mod_bias:
    for y in obs_bias:
        expt_name = f"test_modbias{x:+.1f}_obsbias{y:+.1f}"
        print(expt_name)
        expt  = dap.Expt(expt_name, {'expt_flag': 0, # EnKF
                                    "Ne": 80, 
                                    'model_flag': 2, # Lorenz 05 Model III
                                    'sig_y': 0.1,
                                    'obf': 4,
                                    'localize': 1,
                                    'T': 200,
                                    'xbias': x,
                                    'ybias': y,
                                    'NumPool': 20,
                                    'output_dir': '/Users/knisely/pyDA_data/'
                                    })

        print(expt)
        dap.runDA(expt)
        expt.saveExpt()

        plt.plot(expt.rmse)
        plt.title('Posterior RMSE for Experiment {}'.format(expt.exptname))
        plt.xlabel('T')
        plt.ylabel('Posterior RMSE')
        figname = f"/homes/metogra/jknisely/pyDA/figs/{expt_name}.png"
        print(figname)
        plt.savefig(figname)
        #plt.show()
        plt.close()

print('done all')
