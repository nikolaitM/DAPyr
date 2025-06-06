__all__ = ['MISC', 'MODELS', 'DA']


import numpy as np
import multiprocessing as mp
import copy
import ast
from functools import partial
from . import MODELS
from . import MISC
from . import DA
import xarray as xr
import DAPpyr.Exceptions as dapExceptions
import pickle
import matplotlib.pyplot as plt

class Expt:
      def __init__(self, name, params = None):
            #Expt Name
            self.exptname = name 

            #Dictionaries to store various experimental parameters
            self.modelParams= {}
            self.obsParams = {}
            self.basicParams = {}
            self.miscParams = {}
            self.states = {} #Dictionary to store all the model states

                    
            #Initial the default parameters of an experiment
            self._initBasic()
            self._initModel()
            self._initObs()
            self._initMisc()


            #If additional changes to the parameters are specified, change them
            if params is not None:
                  self.modExpt(params, reqUpdate=True)
            else:
            #If not, update and initalize basic ensemble member states
                  self._updateParams()

      def getParamNames(self):
            paramList = []
            for d in [self. basicParams, self.obsParams, self.modelParams]:
                  paramList.extend(list(d.keys()))
            paramList.remove('rhs')
            paramList.remove('funcptr')
            return paramList

      def __eq__(self, other):
            if not isinstance(other, Expt):
                  return False
            keys = self.getParamNames()
            equality = True
            for key in keys:
                  val1 = self.getParam(key)
                  val2 = other.getParam(key)
                  if isinstance(val1, np.ndarray):
                        equality = np.array_equal(val1, val2)
                  else:
                        equality = val1 == val2
                  if not equality:
                        return equality
            return equality


      def _spinup(self, Nx, Ne, dt, T, tau, funcptr, NumPool, sig_y, ybias, h_flag, H):
            #Initial Ensemble
            #Spin Up
            xt_0 = 3*np.sin(np.arange(Nx)/(6*2*np.pi))
            xt_0 = MODELS.model(xt_0, dt, 100, funcptr)
            
            #Multiprocessing
            xf_0 = xt_0[:, np.newaxis] + 1*np.random.randn(Nx, Ne)
            pfunc = partial(MODELS.model, dt = dt, T = 100, funcptr=funcptr)
            
            with mp.get_context('fork').Pool(NumPool) as pool:
                  xf_0 = np.stack(pool.map(pfunc, [xf_0[:, i] for i in range(Ne)]), axis = -1)
            
            #for n in range(Ne):
            #      xf_0[:, n] = MODELS.model(xf_0[:, n], dt, 100, funcptr)

            #Create Model Truth
            xt = np.zeros((Nx, T))
            xt[:,0] = MODELS.model(xt_0, dt, 100, funcptr)

            for t in range(T-1):
                  xt[:, t+1] = MODELS.model(xt[:, t], dt, tau, funcptr)

            #Synthetic Observations
            dum = np.random.randn(T, Nx).T*sig_y
            match h_flag:
                  case 0:
                        Y = np.matmul(H,(xt + dum))[:, :, np.newaxis]
                  case 1:
                        Y = np.matmul(H, (xt**2 + dum))[:, :, np.newaxis]
                  case 2:
                        Y = np.matmul(H, np.log(np.abs(xt + dum)))[:, :, np.newaxis]
            Y = Y + ybias           #Knisely 

            return xf_0, xt, Y

      def _configModel(self):
            model_flag = self.getParam('model_flag')
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
      def _configObs(self):
            #Extra Observation stuff
            Nx = self.modelParams['Nx']
            H = np.eye(Nx) #Linear Measurement Operator
            H = H[self.obsParams['obb']:Nx-self.obsParams['obb']:self.obsParams['obf'], :]
            self.obsParams['H'] = H
            self.obsParams['Ny'] = len(H)
            #Create localization matrices
            if self.obsParams['localize']==1:
                  C_kf = MISC.create_periodic(self.obsParams['roi_kf'], Nx, 1/Nx)
                  C_pf = MISC.create_periodic(self.obsParams['roi_pf'], Nx, 1/Nx)
            else:
                  C_kf = np.ones((Nx, Nx))
                  C_pf = np.ones((Nx, Nx))
            self.obsParams['C_kf'] = np.matmul(H, C_kf)
            self.obsParams['C_pf'] = np.matmul(H, C_pf)

      def _updateParams(self):
            #Model Truth
            model_flag = self.modelParams['model_flag']
            dt = self.basicParams['dt']
            Ne = self.basicParams['Ne']
            T = self.basicParams['T']
            tau = self.obsParams['tau']

            #Reset Error Flag
            self.basicParams['error_flag'] = 0
            
            self._configModel()

            self._configObs()

            Nx = self.getParam('Nx')
            h_flag = self.getParam('h_flag')
            H = self.getParam("H")
            #Do model spinup
            xf_0, xt, Y = self._spinup(Nx, Ne, dt, T, tau, self.getParam('funcptr'), self.getParam('NumPool'), self.getParam('sig_y'), self.getParam('ybias'), h_flag, H)

            self.states['xf_0'] = xf_0
            self.states['xt'] = xt
            self.states['Y'] = Y

            #Initialize Variables for storage

            if self.getParam('saveEns') != 0:
                  self.x_ens = np.zeros((Nx, Ne, T))*np.nan #All Ensemble Members over time period
            if self.getParam('saveForecastEns') !=0:
                  self.x_fore_ens = np.zeros((Nx, Ne, T))
            if self.getParam('saveEnsMean') != 0:
                  self.x_ensmean = np.zeros((Nx, T))*np.nan
                  self.xf_ensmean = np.zeros((Nx, T))*np.nan
            self.rmse = np.zeros((T,)) #RMSE of Expt
            self.rmse_prior = np.zeros((T,))
            self.spread = np.zeros((T,2)) #Spread of Expt Prio/Posterior

      #Modify the Experiment Parameters
      def _initBasic(self):
            self.basicParams['T'] = 100
            self.basicParams['dt'] = 0.01
            self.basicParams['Ne'] = 80
            self.basicParams['expt_flag'] = 0
            self.basicParams['error_flag'] = 0
      def _initObs(self):
            self.obsParams['h_flag'] = 0 #Linear Operator
            self.obsParams['sig_y'] = 1   #Observation error standard deviation
            self.obsParams['tau'] = 1     #Model steps between observations
            self.obsParams['obf'] = 1   #Observation spatial frequency: spacing between variables
            self.obsParams['obb'] = 0   #Observation buffer: number of variables to skip when generating obs
            self.obsParams['ybias'] = 0   #Observation bias applied to all obs
            self.obsParams['var_y'] = self.obsParams['sig_y']**2
            #Localization
            self.obsParams['localize'] = 1
            #EnKF Parameters
            self.obsParams['roi_kf'] = 0.005
            self.obsParams['inflation'] = 1
            self.obsParams['inf_flag'] = 0
            self.obsParams['gamma'] = 0.03

            #LPF Parameters
            self.obsParams['roi_pf'] = 0.005
            self.obsParams['mixing_gamma'] = 0.3
            self.obsParams['kddm_flag'] = 1
            self.obsParams['min_res'] = 0.0
            self.obsParams['maxiter'] = 1
            self.obsParams['Nt_eff'] = 0.4
      def _initModel(self):
            self.modelParams['model_flag'] = 0
            #Store the default parameters for all the possible models here
            params = {'s': 10, 'r': 28, 'b':8/3, 'F': 8, 
                      'l05_F':15, 'l05_Fe':15,
                      'l05_K':32, 'l05_I':12, 
                      'l05_b':10.0, 'l05_c':2.5}
            self.modelParams['params'] = params
            self.modelParams['xbias'] = 0

      def _initMisc(self):
            #Parameters for Miscellaneous calculations
            #Singular Vector Parameters

            #Output Parameters
            self.miscParams['status'] = 'init'
            self.miscParams['output_dir'] = './'
            self.miscParams['saveEns'] = 1
            self.miscParams['saveEnsMean'] = 1
            self.miscParams['saveForecastEns'] = 0

            self.miscParams['doSV'] = 0 #0 for false, 1 for true
            self.miscParams['stepSV'] = 1 #how many timesteps to skip for each SV calculation
            self.miscParams['forecastSV'] = 4 # Optimization time interval for SV calculation
            self.miscParams['outputSV'] = self.getParam('output_dir') #output directory for SV calculation files
            self.miscParams['storeCovar']= 0 #Store the covariances 
            self.miscParams['NumPool'] = 8

      def resetParams(self):
            self.__init__(self.exptname)

      def modExptName(self, exptname):
            self.exptname = exptname

      def modExpt(self, params : dict, reqUpdate = False):
            #Check if updating ensemble spinup required
            updateRequired = False
            for key, val in params.items():
                  if self.basicParams.get(key) is not None:
                        self.basicParams[key] = val
                        if key != 'expt_flag':
                              updateRequired = True
                  elif self.modelParams.get(key) is not None:
                        updateRequired = True
                        if key =='params':
                              params = self.modelParams[key]
                              for pkey, pval in val.items():
                                    params[pkey] = pval
                              self.modelParams['params'] = params
                        else:
                              self.modelParams[key] = val
                  elif self.obsParams.get(key) is not None:
                        updateRequired = True
                        self.obsParams[key] = val
                  elif self.miscParams.get(key) is not None:
                        self.miscParams[key] = val
                  else:
                        #Turn this into a formal warning eventually
                        print('({}, {}) key value pair not in available parameters'.format(key, val))
            #Only update parameters if they effect the model spinup
            if updateRequired or reqUpdate:
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

            ------------------
            Model Information
            ------------------
            model_flag: {self.modelParams['model_flag']} # Model used in forward integration
                  0: Lorenz 1963 (Nx = 3)
                  1: Lorenz 1996 (Nx = 40)
                  2: Lorenz 2005 (Nx  = 480)
            Nx: {self.modelParams['Nx']} # The number of state variables
            
            params: {self.modelParams['params']} # Parameters to tune each forecast model
            Above is a list of all the parameters stored for use in the forecast model
                  Lorenz 1963: [s, r, b]
                  Lorenz 1996: [F]
                  Lorenz 2005: [l05_F, l05_Fe, l05_K, l05_I, l05_b, l05_c]

            xbias: {self.modelParams['xbias']} # Linear model bias applied to prior at each time step

            ------------------------
            Observation Information
            ------------------------
            h_flag: {self.obsParams['h_flag']} # Type of measurement operator to use
                  0: Linear (x)
                  1: Quadratic (x^2)
                  2: Lognormal (log(abs(x)))
            sig_y: {self.obsParams['sig_y']} # Standard Deviation of observation error
            tau: {self.obsParams['tau']} # Number of model time steps between data assimilation cycles
            obb: {self.obsParams['obb']} # Observation buffer: number of variables to skip when generating obs
            obf: {self.obsParams['obf']} # Observation spatial frequency: spacing between variables
            Ny: {self.obsParams['Ny']} # Number of observations to assimilate each cycle
            ybias: {self.obsParams['ybias']} # Linear observation bias applied to all obs at each time step

            ------------------------
            DA Method Parameter Information
            ------------------------
            expt_flag: {self.basicParams['expt_flag']} # DA method for update step
                  0: Ensemble Square Root Filter (EnSRF)
                  1: Local Particle Filter (LPF)
                  2: No update (xa = xf)
                  ...
            localize: {self.getParam('localize')} # Determines whether to apply localization
                  0: Off
                  1: On

            -----Kalman Filter (EnSRF)-----
            roi_kf: {self.getParam('roi_kf')} # Kalman Filter Localization Radius
            gamma: {self.getParam('gamma')} # RTPS parameter

            -----Local Particle Filter (LPF)-----
            roi_pf: {self.getParam('roi_pf')} # Particle Filter Localization Radius
            mixing_gamma: {self.getParam('mixing_gamma')} # Mixing coefficient for LPF
            kddm_flag: {self.getParam('kddm_flag')} # Determine whether to apply additional kernal density estimator in LPF step
                  0: Off
                  1: On
            maxiter: {self.getParam('maxiter')} # Maximum number of tempering iterations to run
            min_res: {self.getParam('min_res')} # Minimum residual
            Nt_eff: {self.getParam('Nt_eff')} # Effective Ensemble Size
            ------------------------
            Miscellaneous Information
            ------------------------
            status: {self.getParam('status')} # Notes the status of the given experiment
                  init: The experiment has been initialized and spun-up, but not run using runDA
                  init error: An error occured while spinning up the experiment
                  run error: An error occured while running the experiment 
                  completed: runDA has been called and the experiment completed without errors
            output_dir: {self.getParam('output_dir')} # Default output dir for saving experiment-related material
            saveEns: {self.getParam('saveEns')} # Determines whether full posterior ensemble state is saved at each time step
                  0: Off
                  1: On (Default)
            saveEnsMean: {self.getParam('saveEnsMean')} # Determines whether post & prior ensemble mean is saved at each time step
                  0: Off
                  1: On (Default)
            saveForecastEns: {self.getParam('saveForecastEns')} #Determines whether full prior ensemble state is saved at each time step
                  0: Off (Default)
                  1: On
            NumPool: {self.getParam('NumPool')} # Number of CPU cores to use when multiprocessing
            
            -----Singular Vector Configuration-----
            doSV: {self.getParam('doSV')} # Flag to switch on signular value (SV) calculation
            stepSV: {self.getParam('stepSV')} # Number of time steps between SV calculations
            forecastSV: {self.getParam('forecastSV')} # SV optimization interval (in increments of time step)
            outputSV: {self.getParam('outputSV')} # Output Directory for SV output
            storeCovar: {self.getParam('storeCovar')} # Flag to determine whether to store the Analysis and Forecast States to estimate the covariance matrices
                  0: Off (Default)
                  1: On
            '''
            return ret_str
      
      def getBasicParams(self):
            return self.basicParams['Ne'], self.modelParams['Nx'], self.basicParams['T'], self.basicParams['dt']
      
      def getStates(self):
            return self.states['xf_0'], self.states['xt'], self.states['Y']
      
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
      def __deepcopy__(self, memo):
            expt = type(self)("", None)
            memo[id(self)] = expt
            #Change 'rhs' and 'funcptr' because 
            #they are incompatible with the deepycopy method
            self.modelParams['rhs'] = ''
            self.modelParams['funcptr'] = ''
            for name, attr in self.__dict__.items():
                  expt.__setattr__(name, copy.deepcopy(attr, memo))
            self._configModel()
            expt._configModel()
            return expt

      def __copy__(self):
            expt = type(self)("", None)
            expt.__dict__.update(self.__dict__)
            return expt
      

      def copyStates(self, expt):
            #Check if model is the same 
            Nx1, Nx2 = self.getParam('Nx'), expt.getParam('Nx')
            T1, T2 = self.getParam('T'), expt.getParam('T')
            obf1, obf2 = self.getParam('obf'), expt.getParam('obf')
            obb1, obb2 = self.getParam('obb'), expt.getParam('obb')
            Ne1, Ne2 = self.getParam('Ne'), expt.getParam('Ne')
            if Nx1 != Nx2:
                  raise dapExceptions.MismatchModelSize(Nx1, Nx2)
            elif T1 > T2:
                  raise dapExceptions.MismatchTimeSteps(T1, T2)
            elif (obf1 != obf2) or (obb1 != obb2):
                  raise dapExceptions.MismatchObs(obf1, obf2)
            elif Ne1 > Ne2:
                  raise dapExceptions.MisMatchEnsSize(Ne1, Ne2)
            else:
                  xf_0, xt, Y = expt.getStates()
                  self.states['xf_0'] = copy.deepcopy(xf_0[:, :Ne1])
                  self.states['xt'] = copy.deepcopy(xt)[:, :T1]
                  self.states['Y'] = copy.deepcopy(Y)[:, :T1, :]
      def saveExpt(self, outputdir = None):
            if outputdir is None:
                  outputdir = self.getParam('output_dir')
            saveExpt(outputdir, self)

def loadParamFile(filename):
      f = open(filename)
      lines = f.readlines()
      filtered = [x[:x.find('#')] for x in lines]
      expt_params = {}
      params = {}
      model_param_list = ['s', 'b', 'r', 'F', 
                          'l05_F', 'l05_Fe', 'l05_K', 'l05_I',
                          'l05_b', 'l05_c']
      for s in filtered:
            if s == '':
                  continue
            split = s.replace(' ', '').split('=')
            if split[0] == 'expt_name':
                  
                  expt_name = split[1].replace('\'', '').replace('\"', '')
            elif split[0] in model_param_list:
                  params[split[0]] = ast.literal_eval(split[1])
            else:
                  val = ast.literal_eval(split[1])
                  if val is str:
                        val = val.replace('\'', '').replace('\"', '')
                  expt_params[split[0]] = val
            expt_params['params'] = params
      f.close()
      e = Expt(expt_name, expt_params)
      return e



def saveExpt(outputdir, expt: Expt):
      expt.modelParams['rhs'] = ''
      expt.modelParams['funcptr'] = ''
      with open('{}/{}.expt'.format(outputdir, expt.exptname), 'wb') as f:
            pickle.dump(expt, f)
      expt._configModel()

def loadExpt(file):
      with open(file, 'rb') as f:
            expt = pickle.load(f)
      expt._configModel()
      return expt

def plotExpt(expt: Expt, T: int, ax = None, plotObs = False, plotEns = True, plotEnsMean = False):
      '''Plots the model truth, obs, ensembles, and ensemble mean at time T'''
      if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
      #TODO Add check to make sure the expt ran before plotting
      #TODO Add check to make sure plotting time is withtin range of valid values
      Nx = expt.getParam('Nx')
      Ne = expt.getParam('Ne')
      Nt = expt.getParam('T')
      H = expt.getParam('H')
      model_flag = expt.getParam('model_flag')
      xf, xt, Y = expt.getStates()
      if plotEnsMean:
            x_ens = expt.x_ensmean[:, :T+1]
            xf_ens = expt.xf_ensmean[:, :T+1]
      if plotEns:
            x = expt.x_ens[:, :, :T+1]
      if model_flag == 0:
            #Model Truth
            #Plot last 10 steps, if they exist
            diff = 10
            startT = T+1 - diff
            if startT < 0:
                  startT = 0 
            xs_t, ys_t, zs_t = xt[0, startT:T+1], xt[1, startT:T+1], xt[2, startT:T+1]
            #Obs
            #xs_y, ys_y, zs_y = copy.deepcopy(xs_t[:, -1]), copy.deepcopy(ys_t[:, -1]), copy.deepycopy(zs_t[:, -1])

            if plotEns:
                  xs, ys, zs = x[0, :, -diff:].T, x[1, :,  -diff:].T, x[2, :,  -diff:].T
            if plotEnsMean:
                  xs_ens, ys_ens, zs_ens = x_ens[0,  -diff:], x_ens[1,  -diff:], x_ens[2,  -diff:]
      else:
            ts = np.linspace(0, 2*np.pi, Nx)
            xs, ys = np.cos(ts), np.sin(ts)
            #Ensemble
            if plotEns:
                  zs = x[:, :, -1]
            if plotEnsMean:
                  #Ensemble Mean
                  xs_ens, ys_ens = xs, ys 
                  zs_ens = x_ens[:, -1]

            #Obs
            xs_y, ys_y = np.matmul(H, xs[:, np.newaxis]), np.matmul(H, ys[:, np.newaxis])
            zs_y = Y[:, T, 0]
            
            #Model Truth
            xs_t, ys_t = xs, ys
            zs_t = xt[:, T]
      
      #Plot Truth

      if plotEns:
            #Plot Ensemble Members
            for n in range(Ne):
                  if model_flag == 0:
                        ax.plot3D(xs[:, n], ys[:, n], zs[:, n], c = 'grey', alpha = 0.5)
                        ax.scatter(xs[-1, n], ys[-1, n], zs[-1, n], c = 'grey', alpha = 0.5)
                  else:
                        ax.plot3D(xs, ys, zs[:, n], c = 'grey', alpha = 0.5)

      #Truth
      ax.plot3D(xs_t, ys_t, zs_t, 'blue', label = 'True')
      if model_flag==0:
            ax.scatter(xs_t[-1], ys_t[-1], zs_t[-1], c = 'blue')

      #Ensemble Mean
      if plotEnsMean:
            ax.plot3D(xs_ens, ys_ens, zs_ens, c = 'red', label = 'Post. Mean')
            if model_flag == 0:
                  ax.scatter(xs_ens[-1], ys_ens[-1], zs_ens[-1], c = 'red', label = 'Post. Mean')

      if plotObs:
            pass
      if ax is None:
            return fig, ax
      else:
            return ax
      
def copyExpt(expt:Expt):
      return copy.deepcopy(expt)


#TODO Add a helper function that will pring out all configurable settings in the Expt class and list what they do and their potential values
def listParams():
      pass

def runDA(expt: Expt, maxT = None, debug = False):
      #np.random.seed(1)
      # Load in all the variables I need
      Ne, Nx, T, dt = expt.getBasicParams()
      if maxT is not None:
            T = maxT
      numPool = expt.getParam('NumPool')
      #Obs Stuff
      var_y = expt.getParam('var_y')
      H = expt.getParam('H')
      Ny = expt.getParam('Ny')
      tau = expt.getParam('tau')
      C_kf = expt.getParam('C_kf')
      C_pf = expt.getParam('C_pf')
      Nt_eff = expt.getParam('Nt_eff')
      mixing_gamma = expt.getParam('mixing_gamma')
      min_res = expt.getParam('min_res')
      kddm_flag = expt.getParam('kddm_flag')
      maxiter = expt.getParam('maxiter')
      HC_kf = np.matmul(C_kf,H.T)
      HC_pf = np.matmul(C_pf,H.T)
      gamma = expt.getParam('gamma')
      #Flags
      h_flag, expt_flag = expt.getParam('h_flag'), expt.getParam('expt_flag')

      #Model Stuff
      params, funcptr = expt.getParam('params'), expt.getParam('funcptr')
      xbias = expt.getParam('xbias')
      saveEns = expt.getParam('saveEns')
      saveEnsMean = expt.getParam('saveEnsMean')
      saveForecastEns = expt.getParam('saveForecastEns')

      e_flag = expt.getParam('error_flag')
      rmse = expt.rmse
      rmse_prior = expt.rmse_prior
      spread = expt.spread
      if saveEns:
            x_ens = expt.x_ens
      if saveEnsMean:
            x_ensmean = expt.x_ensmean
            xf_ensmean = expt.xf_ensmean
      if saveForecastEns:
            x_fore_ens = expt.x_fore_ens
      #Open pool      
      pool = mp.get_context('fork').Pool(numPool)
      pfunc = partial(MODELS.model, dt = dt, T = tau, funcptr = funcptr)

      #Misc Stuff
      doSV = expt.getParam('doSV')
      #SV calculation
      if doSV==1:
            countSV = 0
            stepSV = expt.getParam('stepSV')
            forecastSV = expt.getParam('forecastSV')
            outputSV = expt.getParam('outputSV')
            storeCovar = expt.getParam('storeCovar')
                  #SV output variables
            xf_sv = np.zeros((Nx, Ne))
            sv_meta = {'expt_name': expt.exptname,
                  'T': T,
                  'stepSV': stepSV
                  }
            sv_coords = {'Nx': ('Nx', np.arange(Nx)),
                        'member': ('mem', np.arange(Ne)),
                        'time': ('t', np.arange(0, T, stepSV))}
            sv_t = len(sv_coords['time'][1])
            sv_data = {'initial': (['t', 'Nx', 'mem'], np.zeros((sv_t, Nx, Ne))*np.nan),
                  'evolved': (['t', 'Nx', 'mem'], np.zeros((sv_t, Nx, Ne))*np.nan),
                  'energy': (['t', 'mem'], np.zeros((sv_t, Ne))*np.nan),
                  'evalue': (['t', 'mem'], np.zeros((sv_t, Ne))*np.nan)}
            
            if storeCovar != 0:
                  sv_covar = {'Xa': (['t', 'Nx','mem'], np.zeros((sv_t, Nx, Ne))*np.nan), 
                                            'Xf': (['t', 'Nx','mem'], np.zeros((sv_t, Nx, Ne))*np.nan)}
            svpfunc = partial(MODELS.model, dt = dt, T = forecastSV, funcptr = funcptr)

      # Time Loop
      xf_0, xt, Y = expt.getStates()
      xf = copy.deepcopy(xf_0)

      for t in range(T):
            xf = xf + xbias         # Knisely

            #Observation
            xm = np.mean(xf, axis = -1)[:, np.newaxis]
            rmse_prior[t] = np.sqrt(np.mean((xt[:, t] - xm[:, 0])**2))
            spread[t, 0] = np.sqrt(np.mean(np.sum((xf - xm)**2, axis = -1)/(Ne - 1)))
            if saveForecastEns:
                  x_fore_ens[:, :, t] = xf
            match h_flag:
                  case 0:
                        hx = np.matmul(H, xf)
                        #hxm = np.mean(hx, axis = -1)
                        #hxm = np.matmul(H, xm)
                  case 1:
                        hx = np.matmul(H, np.square(xf))
                        #hxm = np.mean(hx, axis = -1)
                  case 2:
                        hx = np.matmul(H, np.log(np.abs(xf)))
                        #hxm = np.mean(hx, axis = -1)
                        #hxm = np.matmul(H, np.log(np.abs(xm)))            

            hxm = np.mean(hx, axis = -1)[:, None]
            qaqcpass = np.zeros((Ny,))           # Knisely, turn off QC pass
            #qaqc pass
            #for i in range(Ny):
            #      d = np.abs((Y[i, t, :] - hxm[i, :])[0])
            #      if d > 4 * np.sqrt(np.var(hx[i, :]) + var_y):
            #            qaqcpass[i] = 1
            #Data Assimilation
            match expt_flag:
                  case 0: #Deterministic EnKF
                        xa, e_flag = DA.EnSRF_update(xf, hx, xm ,hxm, Y[:, t], C_kf, HC_kf, var_y, gamma, e_flag, qaqcpass)
                        #xa = enkf_update(xf, hx, xm, hxm, Y[:, t], var_y)
                  case 1: #LPF
                              xa, e_flag = DA.lpf_update(xf, hx, Y[:, t], var_y, H, C_pf, Nt_eff*Ne, mixing_gamma, min_res, maxiter, kddm_flag, e_flag, qaqcpass)
                  case 2: # Stochastic EnKF
                        xa = DA.StochEnKF_update(xf, hx, xm ,hxm, Y[:, t], var_y)
                  case 3: #Nothing
                        xa = xf

            if e_flag != 0:
                  expt.modExpt({'status': 'run_error'})
                  return

            #Store the previous analysis into the matrix
            if saveEns:
                  x_ens[:, :, t] = xa
            if saveEnsMean:
                  x_ensmean[:, t] = np.mean(xa, axis = -1)
                  xf_ensmean[:, t] = np.mean(xf, axis = -1)
                 
            if doSV == 1 and t % stepSV == 0:
                  #Run SV calculation  
                  #xa_sv = copy.deepcopy(xa)
                  xf_sv= np.stack(pool.map(svpfunc, [xa[:, i] for i in range(Ne)]), axis = -1)
                  #for n in range(Ne):
                  #      xf_sv[:, n] = MODELS.model(xa[:, n], dt, forecastSV, funcptr)
                  sv_data['initial'][1][countSV, :, :], sv_data['evolved'][1][countSV, :, :], sv_data['energy'][1][countSV, :], sv_data['evalue'][1][countSV, :] = MISC.calc_SV(xa, xf_sv)
                  if storeCovar != 0:
                        sv_covar['Xa'][1][countSV, :, :] = xa
                        sv_covar['Xf'][1][countSV, :, :] = xf_sv
                  countSV+=1                  

            rmse[t] = np.sqrt(np.mean((xt[:, t] - np.mean(xa, axis = -1))**2))
            spread[t, 1] = np.sqrt(np.mean(np.sum((xa - np.mean(xa, axis = -1)[:, np.newaxis])**2, axis = -1)/(Ne - 1)))

            #Model integrate forward

            #Multiprocessing
            xf = np.stack(pool.map(pfunc, [xa[:, i] for i in range(Ne)]), axis = -1)
            #if t % 5 == 0:
            #      print('Time: {} / RMSE: {}'.format(t, rmse[t]))

            #No multiprocessing
            #for n in range(Ne):
            #      tmp= copy.deepcopy(xa[:, n])
            #      xf[:, n] = MODELS.model(tmp, dt, tau, funcptr)
      pool.close()
      # Save everything into a nice xarray format potentially
      if doSV == 1:
            #Save everything into a netCDF here
            cdf = xr.Dataset(data_vars = sv_data, coords = sv_coords, attrs=sv_meta)
            if storeCovar != 0:
                  cdf = cdf.assign(sv_covar)
            cdf.to_netcdf('{}/SV_{}.cdf'.format(outputSV,expt.exptname), mode = 'w')

      #Output stuff
      expt.modExpt({'status': 'completed'})
      return expt.getParam('status')
