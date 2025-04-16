import copy
import numpy as np

def EnSRF_update(xf, hx, xm, hxm, y, HC, HCH, var_y, gamma, e_flag, qc):
      #Ensemble mean
      Ny = len(y[:, 0])
      Nx, Ne = xf.shape
      xp = xf - xm #Nx x Ne
      xpo = copy.deepcopy(xp) #Original perturbation
      hxp = hx - hxm # Ny x Ne
      #one obs at a time
      if e_flag !=0:
            return np.nan, e_flag
      if np.sum(qc) == Ny:
            e_flag = 1
            return np.nan, e_flag

      for i in range(Ny):
            d = (y[i, :] - hxm[i, :])
            hxo = hxp[i, :]
            var_den = np.dot(hxo, hxo)/(Ne-1) + var_y
            P = np.dot(xp, hxo)/(Ne - 1)
            P = P*HC[i, :]
            K = P/var_den
            xm = xm + K[:, np.newaxis]*d[:, np.newaxis]

            beta = 1/(1 + np.sqrt(var_y/var_den))
            xp = xp - beta*np.dot(K[:, np.newaxis], hxo[np.newaxis, :])

            P = np.dot(hxp, hxo)/(Ne - 1)
            P = P*HCH[i, :]
            K = P/var_den

            hxm = hxm + K[:, np.newaxis]*d[:, np.newaxis]
            beta = 1/(1 + np.sqrt(var_y/var_den))
            hxp = hxp - beta*np.dot(K[:, np.newaxis], hxo[np.newaxis, :])

      #RTPS
      var_xpo = np.sqrt((1/(Ne-1))*np.sum(xpo*xpo, axis = 1)) #Nx x 1
      var_xp = np.sqrt((1/(Ne-1))*np.sum(xp*xp, axis = 1)) #Nx x 1
      inf_factor = gamma*((var_xpo-var_xp)/var_xp) + 1
      xp = xp*inf_factor[:, np.newaxis]
      return xm + xp, e_flag