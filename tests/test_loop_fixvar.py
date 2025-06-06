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

obs_freq = [1,2,4,8,16]
mod_bias = [0] #[-0.5, -0.25, 0.0, 0.25, 0.5]
roi_kf = [0.005]
gamma = [0]
ens_n = [40,80,160,240,320]

for f in obs_freq:
    for x in mod_bias:
        for kf in roi_kf:
            for gam in gamma:
                for nens in ens_n:
                    expt_name = f"test_std{f:g}_modbias{x:+.1f}_roikf{kf:f}_gam{gam:f}"
                    print(expt_name)
                    expt  = dap.Expt(expt_name, {'expt_flag': 0, # EnSRF
                                                "Ne": nens, 
                                                'model_flag': 2, # Lorenz 05 Model III
                                                'sig_y': 0.32,
                                                'obf': f,
                                                'T': 300,
                                                'localize': 0,  # Turns localization on/off
                                                'roi_kf': kf,  # Localization radius
                                                'gamma': gam,  # RTPS parameter
                                                'xbias': mod_bias,
                                                'ybias': 0,
                                                'NumPool': 40,
                                                'output_dir': '/Users/knisely/pyDA_data/'
                                                })

                    print(expt)
                    dap.runDA(expt)
                    expt.saveExpt()

                    plt.plot(expt.rmse)
                    plt.title('Posterior RMSE for Experiment\n{}'.format(expt.exptname))
                    plt.xlabel('T')
                    plt.ylabel('Posterior RMSE')
                    figname = f"/homes/metogra/jknisely/pyDA/figs/{expt_name}.png"
                    print(figname)
                    plt.savefig(figname)
                    #plt.show()
                    plt.close()
print('done all')
