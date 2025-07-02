import sys
sys.path.insert(0,'../src/')
import unittest
import DAPyr as dap
from unittest.mock import patch
import numpy as np
import copy
import DAPyr.MISC
import DAPyr.Exceptions
import matplotlib.pyplot as plt
import gc

obs_freq = [2,8,16]
obs_bias = [-2.0,-1.0,0.0,1.0,2.0]

mod_bias = [-0.5,-0.25,0.0,0.25,0.5]
roi_kf = [0.005,0.005,0.3,0.005,0.005]
gamma = [0.9,0.9,0.3,0.9,0.9]

for f in obs_freq:
    for i, x in enumerate(mod_bias):
        for y in obs_bias:
            expt_name = f"test_std{f:g}_modbias{x:+.2f}_obsbias{y:+.1f}"
            print(expt_name)
            expt  = dap.Expt(expt_name, {'expt_flag': 0, # EnSRF
                                        "Ne": 350, 
                                        'model_flag': 2, # Lorenz 05 Model III
                                        'sig_y': 0.5,
                                        'obf': f,
                                        'T': 500,
                                        'localize': 1,  # Turns localization on/off
                                        'roi_kf': roi_kf[i],  # Localization radius
                                        'gamma': gamma[i],  # RTPS parameter
                                        'xbias': x,
                                        'ybias': y,
                                        'numPool': 40,
                                        'output_dir': '/Users/knisely/pyDA_data/'
                                        })

            #process = psutil.Process(os.getpid())
            #memory_before = process.memory_info().rss / 1024 / 1024  # MB
            #print(f"Memory before {expt_name}: {memory_before:.1f} MB")
            
            print(expt)
            dap.runDA(expt)
            expt.saveExpt()

            #memory_after = process.memory_info().rss / 1024 / 1024  # MB
            #print(f"Memory after {expt_name}: {memory_after:.1f} MB")

            plt.plot(expt.rmse)
            plt.title('Posterior RMSE for Experiment\n{}'.format(expt.exptname))
            plt.xlabel('T')
            plt.ylabel('Posterior RMSE')
            figname = f"/homes/metogra/jknisely/pyDA/figs/{expt_name}.png"
            print(figname)
            plt.savefig(figname)
            #plt.show()
            plt.close()

            del expt
            gc.collect()
print('done all')
