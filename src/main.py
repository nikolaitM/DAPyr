import DAPpyr as dap
import matplotlib.pyplot as plt

f = dap.Expt('enkf_test', {'expt_flag': 0, 'T': 150, 'model_flag': 1, 
                          'Ne': 1000, 'dt': 0.05, 'sig_y': 1, 
                          'roi_kf': 0.001, 'gamma':0.0, 'obf': 5, 
                          'doSV':False, 'stepSV' : 3, 'localize': False})

dap.runDA(f)
plt.plot(f.rmse, label = f.exptname)
plt.legend()
plt.show()