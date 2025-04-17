import numpy as np
import multiprocessing as mp
import copy
from functools import partial
from . import MODELS
from . import MISC

class Expt:
      def __init__(self, name, params = None):
            #Expt Name
            self.exptname = name 

            #Dictionaries to store various experimental parameters
            self.modelParams= {}
            self.obsParams = {}
            self.basicParams = {}
            self.miscParams = {}

            #Initial the default parameters of an experiment
            self._initBasic()
            self._initModel()
            self._initObs()
            self._initMisc()

            #If additional changes to the parameters are specified, change them
            if params is not None:
                  self.modExpt(params)
            else:
            #If not, update and initalize basic ensemble member states
                  self._updateParams()

      
      def _updateParams(self):
            #Model Truth
            self.states = {} #Dictionary to store all the model states
            model_flag = self.modelParams['model_flag']
            dt = self.basicParams['dt']
            Ne = self.basicParams['Ne']
            T = self.basicParams['T']
            tau = self.obsParams['tau']

            match model_flag:
                  case 0: #Lorenz 63
                        self.modelParams['rhs'] = MODELS.make_rhs_l63(self.modelParams['params'])
                        self.modelParams['Nx']  = 3
                  case 1: #Lorenz 96
                        self.modelParams['rhs'] = MODELS.make_rhs_l96(self.modelParams['params'])
                        self.modelParams['Nx'] = 40
                  case 2: #Lorenz 05
                        self.modelParams['rhs'] = MODELS.make_rhs_l05(self.modelParams['params'])
                        self.modelParams['Nx'] = 480
            self.modelParams['funcptr'] = self.modelParams['rhs'].address

            #Spin Up
            #model = self.modelParams['model']
            xt_0 = 3*np.sin(np.arange(self.modelParams['Nx'])/(6*2*np.pi))
            xt_0 = MODELS.model(xt_0, dt, 100, self.modelParams['funcptr'])

            #Extra Observation stuff
            Nx = self.modelParams['Nx']
            H = np.eye(Nx) #Linear Measurement Operator
            H = H[self.obsParams['obb']:Nx-self.obsParams['obb']:self.obsParams['obf'], :]
            self.obsParams['H'] = H
            self.obsParams['Ny'] = len(H)
            #Create localization matrices
            if self.obsParams['localize']:
                  C_kf = MISC.create_periodic(self.obsParams['roi_kf'], Nx, 1/Nx)
                  C_pf = MISC.create_periodic(self.obsParams['roi_pf'], Nx, 1/Nx)
            else:
                  C_kf = np.ones((Nx, Nx))
                  C_pf = np.ones((Nx, Nx))
            self.obsParams['C_kf'] = np.matmul(H, C_kf)
            self.obsParams['C_pf'] = np.matmul(H, C_pf)

            
            #Initial Ensemble
            
            #Multiprocessing
            xf = xt_0[:, np.newaxis] + 1*np.random.randn(Nx, Ne)
            pfunc = partial(MODELS.model, dt = dt, T = T, funcptr=self.modelParams['funcptr'])
            with mp.Pool(self.getParam('NumPool')) as pool:
                  xf = np.stack(pool.map(pfunc, [xf[:, i] for i in range(Ne)]), axis = -1)
            #

            #No Multiprocessing
            #xf = np.empty((Nx, Ne))
            #for n in range(Ne):
            #      dum = xt_0 + 1*np.random.randn(Nx)
            #      xf[:, n] = model(dum, dt, 100, self.modelParams['funcptr'])
            xf_0 = copy.deepcopy(xf)
            #Create Model Truth
            xt = np.zeros((Nx, T))
            xt[:,0] = MODELS.model(xt_0, dt, 100, self.modelParams['funcptr'])
            for t in range(T-1):
                  xt[:, t+1] = MODELS.model(xt[:, t], dt, tau, self.modelParams['funcptr'])

            #Synthetic Observations
            dum = np.random.randn(T, Nx).T*self.obsParams['sig_y']
            match self.obsParams['h_flag']:
                  case 0:
                        Y = np.matmul(H,(xt + dum))[:, :, np.newaxis]
                  case 1:
                        Y = np.matmul(H, (xt + dum)**2)[:, :, np.newaxis]
                  case 2:
                        Y = np.matmul(H, np.log(np.abs(xt + dum)))[:, :, np.newaxis]
            
            self.states['xf_0'] = xf_0
            self.states['xf'] = xf
            self.states['xt'] = xt
            self.states['Y'] = Y

            #Initialize Variables for storage
            self.x_ens = np.zeros((Nx, Ne, T)) #All Ensemble Members over time period
            self.rmse = np.zeros((T,)) #RMSE of Expt
            self.spread = np.zeros((T,)) #Spread of Expt

      #Modify the Experiment Parameters
      def _initBasic(self):
            self.basicParams['T'] = 100
            self.basicParams['dt'] = 0.01
            self.basicParams['Ne'] = 10
            self.basicParams['expt_flag'] = 0
            self.basicParams['error_flag'] = 0
            self.basicParams['NumPool'] = 8
      def _initObs(self):
            self.obsParams['h_flag'] = 0 #Linear Operator
            self.obsParams['sig_y'] = 1   #Observation error standard deviation
            self.obsParams['tau'] = 3     #Model steps between observations
            self.obsParams['obf'] = 1   #Observation spatial frequency: spacing between variables
            self.obsParams['obb'] = 0   #Observation buffer: number of variables to skip when generating obs
            self.obsParams['var_y'] = self.obsParams['sig_y']**2
            #Localization
            self.obsParams['localize'] = True
            #EnKF Parameters
            self.obsParams['roi_kf'] = 0.005
            self.obsParams['gamma'] = 0.03
            #LPF Parameters
            self.obsParams['roi_pf'] = 0.005
      def _initModel(self):
            self.modelParams['model_flag'] = 0
            #Store the default parameters for all the possible models here
            params = {'s': 10, 'r': 28, 'b':8/3, 'F': 8, 
                      'l05_F':15, 'l05_Fe':15,
                      'l05_K':32, 'l05_I':12, 
                      'l05_b':10.0, 'l05_c':2.5}
            self.modelParams['params'] = params

      def _initMisc(self):
            #Parameters for Miscellaneous calculations
            self.miscParams['doSV'] = False #0 for false, 1 for true
            self.miscParams['stepSV'] = 1 #how many timesteps to skip for each SV calculation
            self.miscParams['forecastSV'] = 4 # Optimization time interval for SV calculation
            self.miscParams['outputSV'] = './SVs/' #output directory for SV calculation files


      def resetParams(self):
            self.__init__(self.exptname)

      def modExpt(self, params : dict):
            for key, val in params.items():
                  if self.basicParams.get(key) is not None:
                        self.basicParams[key] = val
                  elif self.modelParams.get(key) is not None:
                        if key =='params':
                              params = self.modelParams[key]
                              for pkey, pval in val.items():
                                    params[pkey] = pval
                              self.modelParams['params'] = params
                        else:
                              self.modelParams[key] = val
                  elif self.obsParams.get(key) is not None:
                        self.obsParams[key] = val
                  elif self.miscParams.get(key) is not None:
                        self.miscParams[key] = val
                  else:
                        print('({}, {}) key value pair not in available parameters'.format(key, val))
            self._updateParams()
      def __str__(self):
            #Basic Model Setup Print
            ret_str = f'''
            ------------------
            Basic Information
            ------------------
            Experiment Name: {self.exptname}
            Ne: {self.basicParams['Ne']} # Number of Ensemble Members
            T: {self.basicParams['T']} # Number of Time Periods
            dt: {self.basicParams['dt']} # Width of Timesteps
            expt_flag: {self.basicParams['expt_flag']} # DA method for update step
                  0: EnKF
                  1: Bootstrap Particle Filter
                  2: No update (xa = xf)
                  ...

            ------------------
            Model Information
            ------------------
            model_flag: {self.modelParams['model_flag']} # Model used in forward integration
                  0: Lorenz 1963
                  1: Lorenz 1996
                  2: Lorenz 2005
            Nx: {self.modelParams['Nx']} # The number of state variables
            
            params: {self.modelParams['params']} # Parameters to tune each forecast model
            Above is a list of all the parameters stored for use in the forecast model
                  Lorenz 1963: [s, r, b]
                  Lorenz 1996: [F]
                  Lorenz 2005: []

            ------------------------
            Observation Information
            ------------------------
            h_flag: {self.obsParams['h_flag']} # Type of measurement operator to use
                  0: Linear (x)
                  1: Quadratic (x^2)
                  2: Lognormal (log(abs(x)))
            sig_y: {self.obsParams['sig_y']} # Standard Deviation of observation error
            tau: {self.obsParams['tau']} # Model steps between observations
            obb: {self.obsParams['obb']} # Observation buffer: number of variables to skip when generating obs
            obf: {self.obsParams['obf']} # Observation spatial frequency: spacing between variables
            Ny: {self.obsParams['Ny']} # Number of observation each cycle

            ------------------------
            DA Method Information
            ------------------------
            
            roi_kf: {self.getParam('roi_kf')} # Kalman Filter Localization Radius
            roi_pf: {self.getParam('roi_pf')} # Particle Filter Localization Radius

            gamma: {self.getParam('gamma')} # RTPS parameter

            '''
            return ret_str
      
      def getBasicParams(self):
            return self.basicParams['Ne'], self.modelParams['Nx'], self.basicParams['T'], self.basicParams['dt']
      
      def getStates(self):
            return self.states['xf'], self.states['xt'], self.states['Y']
      
      def getParam(self, param):
            if self.basicParams.get(param) is not None:
                  return self.basicParams.get(param)
            elif self.modelParams.get(param) is not None:
                  return self.modelParams.get(param)
            elif self.obsParams.get(param) is not None:
                  return self.obsParams.get(param)
            elif self.modelParams['params'].get(param) is not None:
                  return self.modelParams['params'].get(param)
            elif self.states.get(param) is not None:
                  return self.states.get(param)
            elif self.miscParams.get(param) is not None:
                  return self.miscParams.get(param)
            else:
                  return None
      def copyStates(self, expt):
            #Check if model is the same 
            if self.getParam('Nx') == expt.getParam('Nx'):
                  xf, xt, Y = expt.getStates()
                  self.states['xf'] = copy.deepcopy(xf)
                  self.states['xt'] = copy.deepcopy(xt)
                  self.states['Y'] = copy.deepcopy(Y)
            else:
                  raise Exception #Change this to a meaningful exception eventually
            