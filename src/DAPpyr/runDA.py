
from . import EXPT
from . import DA
from . import MODELS
import numpy as np
import xarray as xr
import multiprocessing as mp
from functools import partial


def runDA(expt: EXPT.Expt):
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
      HC_kf = np.matmul(C_kf,H.T)
      gamma = expt.getParam('gamma')
      #Flags
      h_flag, expt_flag = expt.getParam('h_flag'), expt.getParam('expt_flag')

      #Model Stuff
      params, funcptr = expt.getParam('params'), expt.getParam('funcptr')
      e_flag = expt.getParam('error_flag')
      rmse = expt.rmse
      x_ens = expt.x_ens

      pool = mp.Pool(numPool)
      pfunc = partial(MODELS.model, dt = dt, T = T, funcptr = funcptr)

      #Misc Stuff
      doSV = expt.getParam('doSV')
      #SV calculation
      if doSV:
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


      # Time Loop
      xf, xt, Y = expt.getStates()
      xf = expt.getParam('xf_0')

      for t in range(T):
            #Observation
            xm = np.mean(xf, axis = -1)[:, np.newaxis]
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
                  case 1: #Bootstrap PF
                        xa = DA.pf_update(xf, hx, Y[:, t], var_y)
                  case 2: # Stochastic EnKF
                        xa = DA.StochEnKF_update(xf, hx, xm ,hxm, Y[:, t], var_y)
                  case 3: #Nothing
                        xa = xf

            if e_flag != 0:
                  xa = xa*np.nan
                  x_ens[:, :, t] = xa
                  continue

            #Store the previous analysis into the matrix
            x_ens[:, :, t] = xa
            if doSV and t % stepSV == 0:
                  #Run SV calculation  
                  for n in range(Ne):
                        xf_sv[:, n] = MODELS.model(xa[:, n], dt, forecastSV, funcptr)
                  sv_data['initial'][1][countSV, :, :], sv_data['evolved'][1][countSV, :, :], sv_data['energy'][1][countSV, :], sv_data['evalue'][1][countSV, :] = calc_SV(xa, xf_sv)
                  countSV+=1                  

            rmse[t] = np.sqrt(np.mean((xt[:, t] - np.mean(xa, axis = -1))**2))
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
