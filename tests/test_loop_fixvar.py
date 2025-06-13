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
import psutil
import os
import gc

obs_freq = [1]
mod_bias = [-0.5,-0.25,0,0.25,0.5]
roi_kf = [0.1,0.3,0.5,0.7,0.9] #Check localization flag
gamma = [0.1,0.3,0.5,0.7,0.9]
ens_n = [350]

for f in obs_freq:
    for x in mod_bias:
        for kf in roi_kf:
            for gam in gamma:
                for nens in ens_n:
                    expt_name = f"test_std{f:g}_modbias{x:+.2f}_roikf{kf:g}_gam{gam:g}"
                    #expt_name = f"test_std{f:g}_nens{nens:g}"
                    print(expt_name)
                    expt  = dap.Expt(expt_name, {'expt_flag': 0, # EnSRF
                                                "Ne": nens, 
                                                'model_flag': 2, # Lorenz 05 Model III
                                                'sig_y': 0.5,
                                                'obf': f,
                                                'T': 200,
                                                'localize': 1,  # Turns localization on/off
                                                'roi_kf': kf,  # Localization radius
                                                'gamma': gam,  # RTPS parameter
                                                'xbias': x,
                                                'ybias': 0,
                                                'NumPool': 40,
                                                'output_dir': '/Users/knisely/pyDA_data/'
                                                })

                    process = psutil.Process(os.getpid())
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    print(f"Memory before {expt_name}: {memory_before:.1f} MB")
                    
                    #print(expt)
                    dap.runDA(expt)
                    expt.saveExpt()

                    memory_after = process.memory_info().rss / 1024 / 1024  # MB
                    print(f"Memory after {expt_name}: {memory_after:.1f} MB")

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
