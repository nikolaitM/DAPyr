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


      def _spinup(self, Nx, Ne, dt, T, tau, funcptr, NumPool, sig_y, h_flag, H):
            #Initial Ensemble
            #Spin Up
            xt_0 = 3*np.sin(np.arange(Nx)/(6*2*np.pi))
            xt_0 = MODELS.model(xt_0, dt, 100, funcptr)
            
            #Multiprocessing
            xf_0 = xt_0[:, np.newaxis] + 1*np.random.randn(Nx, Ne)
            pfunc = partial(MODELS.model, dt = dt, T = 100, funcptr=funcptr)
            with mp.Pool(NumPool) as pool:
                  xf_0 = np.stack(pool.map(pfunc, [xf_0[:, i] for i in range(Ne)]), axis = -1)
            
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
                        Y = np.matmul(H, (xt + dum)**2)[:, :, np.newaxis]
                  case 2:
                        Y = np.matmul(H, np.log(np.abs(xt + dum)))[:, :, np.newaxis]

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
            self.states = {} #Dictionary to store all the model states
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
            xf_0, xt, Y = self._spinup(Nx, Ne, dt, T, tau, self.getParam('funcptr'), self.getParam('NumPool'), self.getParam('sig_y'), h_flag, H)

            self.states['xf_0'] = xf_0
            self.states['xt'] = xt
            self.states['Y'] = Y

            #Initialize Variables for storage

            if self.getParam('saveEns') != 0:
                  self.x_ens = np.zeros((Nx, Ne, T))*np.nan #All Ensemble Members over time period
            if self.getParam('saveEnsMean') != 0:
                  self.x_ensmean = np.zeros((Nx, T))*np.nan
            self.rmse = np.zeros((T,)) #RMSE of Expt
            self.spread = np.zeros((T,2)) #Spread of Expt Prio/Posterior

      #Modify the Experiment Parameters
      def _initBasic(self):
            self.basicParams['T'] = 100
            self.basicParams['dt'] = 0.01
            self.basicParams['Ne'] = 10
            self.basicParams['expt_flag'] = 0
            self.basicParams['error_flag'] = 0
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

      def _initMisc(self):
            #Parameters for Miscellaneous calculations
            #Singular Vector Parameters
            self.miscParams['doSV'] = False #0 for false, 1 for true
            self.miscParams['stepSV'] = 1 #how many timesteps to skip for each SV calculation
            self.miscParams['forecastSV'] = 4 # Optimization time interval for SV calculation
            self.miscParams['outputSV'] = './' #output directory for SV calculation files
            
            self.miscParams['NumPool'] = 8

            #Output Parameters
            self.miscParams['status'] = 'init'
            self.miscParams['output_dir'] = './'
            self.miscParams['saveEns'] = 0
            self.miscParams['saveEnsMean'] = 1


      def resetParams(self):
            self.__init__(self.exptname)

      def modExpt(self, params : dict):
            #Check if updating ensemble spinup required
            updateRequired = False
            for key, val in params.items():
                  if self.basicParams.get(key) is not None:
                        self.basicParams[key] = val
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
            if updateRequired:
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
                  Lorenz 2005: [l05_F, l05_Fe, l05_K, l05_I, l05_b, l05_c]

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
      def copyStates(self, expt):
            #Check if model is the same 
            Nx1, Nx2 = self.getParam('Nx'), expt.getParam('Nx')
            T1, T2 = self.getParam('T'), expt.getParam('T')
            Ny1, Ny2 = self.getParam('Ny'), expt.getParam('Ny')
            if Nx1 != Nx2:
                  raise dapExceptions.MismatchModelSize(Nx1, Nx2)
            elif T1 > T2:
                  raise dapExceptions.MismatchTimeSteps(T1, T2)
            elif Ny1 != Ny2:
                  raise dapExceptions.MismatchObs(Ny1, Ny2)
            else:
                  xf_0, xt, Y = expt.getStates()
                  self.states['xf_0'] = copy.deepcopy(xf_0)
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


def runDA(expt: Expt):
      #np.random.seed(1)
      # Load in all the variables I need
      Ne, Nx, T, dt = expt.getBasicParams()
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
      saveEns = expt.getParam('saveEns')
      saveEnsMean = expt.getParam('saveEnsMean')

      e_flag = expt.getParam('error_flag')
      rmse = expt.rmse
      spread = expt.spread
      if saveEns:
            x_ens = expt.x_ens
      if saveEnsMean:
            x_ensmean = expt.x_ensmean
      #Open pool      
      pool = mp.Pool(numPool)
      pfunc = partial(MODELS.model, dt = dt, T = tau, funcptr = funcptr)

      #Misc Stuff
      doSV = expt.getParam('doSV')
      #SV calculation
      if doSV==1:
            countSV = 0
            stepSV = expt.getParam('stepSV')
            forecastSV = expt.getParam('forecastSV')
            outputSV = expt.getParam('outputSV')
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
            svpfunc = partial(MODELS.model, dt = dt, T = forecastSV, funcptr = funcptr)

      # Time Loop
      xf_0, xt, Y = expt.getStates()
      xf = copy.deepcopy(xf_0)

      for t in range(T):
            #Observation
            xm = np.mean(xf, axis = -1)[:, np.newaxis]
            spread[t, 0] = np.sqrt(np.mean(np.sum((xf - xm)**2, axis = -1)/(Ne - 1)))
            match h_flag:
                  case 0:
                        hx = np.matmul(H, xf)
                        hxm = np.matmul(H, xm)
                  case 1:
                        hx = np.matmul(H, np.square(xf))
                        hxm = np.matmul(H, np.square(xm))
                  case 2:
                        hx = np.matmul(H, np.log(np.abs(xf)))
                        hxm = np.matmul(H, np.log(np.abs(xm)))            


            qaqcpass = np.zeros((Ny,))
            #qaqc pass
            for i in range(Ny):
                  d = np.abs((Y[i, t, :] - hxm[i, :])[0])
                  if d > 4 * np.sqrt(np.var(hx[i, :]) + var_y):
                        qaqcpass[i] = 1
            #Data Assimilation
            match expt_flag:
                  case 0: #Deterministic EnKF
                        xa, e_flag = DA.EnSRF_update(xf, hx, xm ,hxm, Y[:, t], C_kf, HC_kf, var_y, gamma, e_flag, qaqcpass)
                        #xa = enkf_update(xf, hx, xm, hxm, Y[:, t], var_y)
                  case 1: #LPF
                        xa, e_flag = DA.pf_update(xf, hx, Y[:, t, :].T, C_pf, HC_pf, Nt_eff*Ne, min_res, mixing_gamma, var_y, kddm_flag, qaqcpass, maxiter)
                        #xa = DA.pf_update(xf, hx, Y[:, t], var_y)
                        #xa = DA.lpf_update(xf, hx, Y[:, t], var_y, H, C_pf, Nt_eff*Ne, mixing_gamma)
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
                 
            if doSV and t % stepSV == 0:
                  #Run SV calculation  
                  #xa_sv = copy.deepcopy(xa)
                  xf_sv= np.stack(pool.map(svpfunc, [xa[:, i] for i in range(Ne)]), axis = -1)
                  #for n in range(Ne):
                  #      xf_sv[:, n] = MODELS.model(xa[:, n], dt, forecastSV, funcptr)
                  sv_data['initial'][1][countSV, :, :], sv_data['evolved'][1][countSV, :, :], sv_data['energy'][1][countSV, :], sv_data['evalue'][1][countSV, :] = MISC.calc_SV(xa, xf_sv)
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
            #      xf[:, n] = model(xa[:, n], dt, tau, funcptr)
      pool.close()
      # Save everything into a nice xarray format potentially
      if doSV:
            #Save everything into a netCDF here
            cdf = xr.Dataset(data_vars = sv_data, coords = sv_coords, attrs=sv_meta)
            cdf.to_netcdf('{}/SV_{}.cdf'.format(outputSV,expt.exptname), mode = 'w')

      #Output stuff
      expt.modExpt({'status': 'completed'})
      return expt.getParam('status')