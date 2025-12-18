import numpy as np
from . import MISC
import warnings

def do_RTPS(xo_prior : np.ndarray, xa: np.ndarray, gamma: float):
      '''
      Performs Relaxation to Prior Spread inflation on a set of forecast and analysis states given an inflation value.
      
      Parameters
      ----------
      xo_prior: np.ndarray
            Forecast ensemble model state of size (Nx x Ne)
      xa: np.ndarray
            Analysis ensemble model state of size (Nx x Ne)
      gamma: float
            Inflation parameter 
      
      Returns
      ---------
      xa : np.ndarray
            Inflated analysis model state of size (Nx x Ne)
      '''
      Nx, Ne = xo_prior.shape
      xo_m = np.mean(xo_prior, axis = -1)
      xa_m = np.mean(xa, axis = -1)
      xpo = xo_prior - xo_m[:, None]
      xp = xa - xa_m[:, None]
      var_xpo = np.sqrt((1/(Ne-1))*np.sum(xpo*xpo, axis = 1)) #Nx x 1
      var_xp = np.sqrt((1/(Ne-1))*np.sum(xp*xp, axis = 1)) #Nx x 1
      inf_factor = gamma*((var_xpo-var_xp)/var_xp) + 1
      xp_ret = xp*inf_factor[:, np.newaxis]
      return xa_m[:, None] + xp_ret


def do_RTPP(xf_prior:np.ndarray, xa_old: np.ndarray, gamma: float):
      '''
      Performs Relaxation to Prior Perturbation inflation on a set of forecast and analysis model states given an inflation parameter.
      
      Parameters
      ----------
      xo_prior: np.ndarray
            Forecast ensemble model state of size (Nx x Ne)
      xa: np.ndarray
            Analysis ensemble model state of size (Nx x Ne)
      gamma: float
            Inflation parameter 
      
      Returns
      ---------
      xa : np.ndarray
            Inflated analysis model state of size (Nx x Ne)
      '''
      xf_m = np.mean(xf_prior, axis = -1)
      xa_m = np.mean(xa_old, axis = -1)
      xpf = xf_prior - xf_m[:, None]
      xpa = xa_old - xa_m[:, None]
      xp_ret = (1 - gamma)*xpa + gamma*xpf
      return xa_m[:, None] + xp_ret


def do_Anderson2009(xf: np.ndarray, hx:np.ndarray, Y:np.ndarray,
                  gamma_p: np.ndarray, sigma_lambda_p: np.ndarray,
                  HC: np.ndarray, var_y: float):
      '''
      Docstring for do_Anderson2009
      
      Parameters
      -----------
      xf: np.ndarray
            Forecast ensemble model state of size (Nx x Ne)
      hx: np.ndarray
            Forecast ensemble model state projected into observation space of size (Ny x Ne)
      Y: np.ndarray
            Array of observations to assimilate of size (Ny x 1)
      gamma_p: np.ndarray
            Array of starting estimates of mean inflation parameter of size (Nx)
      sigma_lambda_p: np.ndarray
            Array of starting estimates of the variance of inflation parameter with size (Nx)
      HC: np.ndarray
            An array of the localization matrix projected into observation space of size (Ny x Nx)
      var_y: float
            The assumed variance of the observation error

      Returns
      --------
      xf_inf: np.ndarray
            Updated inflate ensemble model state of size (Nx x Ne)
      gamma_p: np.ndarray
            Updated estimates of inflation parameter means of size (Nx)
      sigma_lambda_p: np.ndarray
            Updated estimates of inflation parameter variances of size (Nx)
      '''
      Nx, Ne = xf.shape
      xf_m = np.mean(xf, axis = -1)[:, None]
      xfpo = xf - xf_m
      #For each observation
      for i, y in enumerate(Y[:, 0]):
            ys = hx[i, :]
            y_bar = np.mean(ys)
            ypo = ys - y_bar
            sigma_p = np.var(ys, ddof = 1)
            cov = np.matmul(xfpo, ypo[:, None])/(Ne - 1) #This is of size Nx

            D = np.abs(y - y_bar) #size Ne
            D_square = np.square(D)
            loc_gamma = HC[i, :]*(cov[:, 0]/sigma_p) # size Nx
            lambda_o = (1 + loc_gamma*(np.sqrt(gamma_p) - 1))**2 # size Nx
            theta_bar = np.sqrt(lambda_o*sigma_p + var_y) # size Nx
            l_bar = (1/(np.sqrt(2*np.pi)*theta_bar))*np.exp((-0.5*D_square)/(theta_bar**2))
            dtheta_dlambda = 0.5*sigma_p*loc_gamma*(1-loc_gamma + loc_gamma*np.sqrt(gamma_p))/(theta_bar*np.sqrt(gamma_p))
            l_prime = (l_bar*(D_square/theta_bar**2 - 1)*(dtheta_dlambda))/theta_bar

            b = l_bar/l_prime - 2*gamma_p
            c = np.square(gamma_p) - sigma_lambda_p - (l_bar*gamma_p/l_prime)
            discriminant = np.sqrt(b**2 - 4*c)
            root1 = (-b + discriminant)/2.0
            root2 = (-b - discriminant)/2.0
            diff1, diff2 = np.abs(root1 - gamma_p), np.abs(root2 - gamma_p)
            gamma_u = np.where(diff1 < diff2, root1, root2)
            #If the root is negative, disregard
            gamma_u =np.where(gamma_u > 0.0, gamma_u, gamma_p)
            #Disregard inflation parameters in which covariance between observation and model space is negative
            gamma_u = np.where(loc_gamma > 0.0, gamma_u, gamma_p)
            
            #Adjust the variance of each inflation random variable
            gamma_u_inc = gamma_u + np.sqrt(sigma_lambda_p)

            theta = np.sqrt(gamma_u*sigma_p + var_y)
            num1 = (1/(np.sqrt(2*np.pi)*theta))*np.exp((-0.5*D_square/(2*theta)))*(1/(np.sqrt(2*np.pi)*theta))*(np.exp((-0.5*(gamma_u - gamma_p)**2/(2*sigma_lambda_p))))
            theta = np.sqrt(gamma_u_inc*sigma_p + var_y)
            num2 = (1/(np.sqrt(2*np.pi)*theta))*np.exp((-0.5*D_square/(2*theta)))*(1/(np.sqrt(2*np.pi)*theta))*(np.exp((-0.5*(gamma_u_inc - gamma_p)**2/(2*sigma_lambda_p))))

            R = num2/num1
            sigma_u = -sigma_lambda_p/(2*np.log(R))
            sigma_u = np.where((np.isnan(sigma_u)) | (sigma_u < 0.6), 0.6, sigma_u)
            
            #Make the posterior the prior
            gamma_p = gamma_u
            #Uncomment below in order to do updates to the standard devaiation estimate of the inflation parameter
            #sigma_lambda_p = sigma_u

      #Limit inflation to be greater than 1
      gamma_p = np.where((gamma_p < 1) | (np.isnan(gamma_p)), 1, gamma_p)
      #Inflate the prior ensemble member for each state vector component by the mean of hte corresponding updated inflation distribution
      xf_inf = np.sqrt(gamma_p[:, None])*(xfpo) + xf_m

      return xf_inf, gamma_p, sigma_lambda_p




