import numpy as np
def StochEnKF_update(xf, hx, xm, hxm, y, var_y):
      #Emsemble mean
      Ny = len(y[:, 0])
      Nx, Ne = xf.shape
      eps = np.random.normal(0, np.sqrt(var_y), (Ny, Ne))
      eps_mean = np.mean(eps, axis = -1)[:, np.newaxis]
      X = (1/np.sqrt(Ne-1))*(xf - xm)
      Y = (1/np.sqrt(Ne-1))*(hx + eps - hxm - eps_mean)
      XY = np.matmul(X, Y.T)
      YY = np.matmul(Y, Y.T)
      eta = np.zeros(xf.shape)
      for n in range(Ne):
            b = np.linalg.solve(YY, y[:, 0] - (hx[:, n] + eps[:, n]))[:, np.newaxis]
            eta[:, n] = xf[:, n] + np.matmul(XY, b)[:, 0]
      return eta
