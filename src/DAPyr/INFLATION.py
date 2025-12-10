import numpy as np
from . import MISC
import warnings

def do_RTPS(xo_prior, xa, gamma):
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


def do_RTPP(xf_prior, xa_old, gamma):
      xf_m = np.mean(xf_prior, axis = -1)
      xa_m = np.mean(xa_old, axis = -1)
      xpf = xf_prior - xf_m[:, None]
      xpa = xa_old - xa_m[:, None]
      xp_ret = (1 - gamma)*xpa + gamma*xpf
      return xa_m[:, None] + xp_ret


def do_Anderson2009(xf, hx, Y, gamma_p, sigma_lambda_p, HC, var_y):
      #xf prior ensemble members (Nx by Ne)
      #hx obs space ensemble members (Ny by Ne)
      #Y observations
      #gamma_p, size Nx x 1
      #sigma_p size Nx x 1
      #C Localization matrix, Nx by Nx
      #HC Obs Space Localization Matrix, Ny by Nx
      # sig_y, observation error
      Nx, Ne = xf.shape
      xf_m = np.mean(xf, axis = -1)[:, None]
      xfpo = xf - xf_m
      #For each observation
      for i, y in enumerate(Y[:, 0]):
            ys = hx[i, :]
            y_bar = np.mean(ys)
            ypo = ys - y_bar
            sigma_p = np.var(ys, ddof = 1)
            #I need the covariance between perturbations and 
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
            a = 1
            c = np.square(gamma_p) - sigma_lambda_p - (l_bar*gamma_p/l_prime)
            discriminant = np.sqrt(b**2 - 4*a*c)
            root1 = (-b + discriminant)/(2*a*c)
            root2 = (-b - discriminant)/(2*a*c)
            diff1, diff2 = np.abs(root1 - gamma_p), np.abs(root2 - gamma_p)
            gamma_u = np.where(diff1 < diff2, root1, root2)
            
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
            sigma_lambda_p = sigma_u
      #Limit inflation to be greater than 1
      gamma_p = np.where((gamma_p < 1) | (np.isnan(gamma_p)), 1, gamma_p)
      #Inflate the prior ensemble member for each state vector component by the mean of hte corresponding updated inflation distribution
      xf_inf = np.sqrt(gamma_p[:, None])*(xfpo) + xf_m

      return xf_inf, gamma_p, sigma_lambda_p




