import numpy as np
import copy
from scipy.optimize import fmin
from . import MISC

def pf_update(xm, hx, Y, var_y):
      Nx, Ne = xm.shape
      Ny = len(Y)
      xa = xm
      hxa = hx
      for i in range(Ny):
            d = Y[i, :] - hxa[i,:] #innovation
            w = (1/np.sqrt(2*np.pi*var_y))*np.exp(-d**2/(2*var_y))
            w = w/np.sum(w) #normalize
            #Sample with replacement from probabilities 
            ind = np.random.choice(Ne, Ne, p=w)
            xa = xa[:, ind]
            hxa = hxa[:, ind]
      #Add some noise to the final product
      return xa + np.random.randn(Nx, Ne)*0.1

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


def lpf_update(x, hx, Y, var_y, H, C_pf, N_eff, gamma):
      Nx, Ne = x.shape
      Ny = len(Y)
      hx_prev = copy.deepcopy(hx) #From previous assimilated obs
      x_prev = copy.deepcopy(x) #from previously assimilated obs cycle Nx x Ne
      omega = np.ones((Nx, Ne))*(1/Ne) #Nx x Ne
      beta = np.zeros((Ny,))
      beta_carrot = np.zeros((Ny,))
      #Beta stuff
      for i in range(Ny):
            d_carrot = Y[i, :] - hx[i, :] #Ne
            def find_beta(b):
                  return N_eff - ( ( np.sum( np.exp( (-1*(d_carrot**2))/(2*b*var_y) ) ) )**2 / ( np.sum( np.exp( -1*(d_carrot**2)/(b*var_y) ) ) ) )
            min_f = fmin(find_beta, 1) #argmin here
            beta_carrot[i] = min_f.item()
            for k in range(Ny):
                  beta[k] = beta[k] + (beta_carrot[i] - 1)*np.matmul(H, C_pf[i, :][:, np.newaxis])[k, 0]
      #Obs loop
      for i in range(Ny):
            d_carrot = Y[i, :] - hx[i, :] #Ne
            d_tilde = Y[i, :] - hx_prev[i, :] #Ne
            w_carrot = np.exp((-1*(d_carrot**2))/2*beta[i]*var_y)#Ne
            w_tilde = np.exp((-1*(d_tilde**2))/2*beta[i]*var_y) #Ne

            
            W_carrot = np.sum(w_carrot)
            W_tilde = np.sum(w_tilde)

            w_carrot = w_carrot/W_carrot
            w_tilde = w_tilde/W_tilde

            ks = np.random.choice(Ne, Ne, p = w_tilde, replace=True)
            x_tilde = x_prev[:, ks]
            #C_pf is a Ny by Nx
            Omega_carrot = np.sum(w_carrot[np.newaxis, :]*omega, axis = -1) #Nx
            omega = omega*((np.matmul(C_pf[i, :][:, np.newaxis], (Ne*w_carrot[np.newaxis, :] -1)) + 1)/Ne) #Needs to be Nx x Ne
            Omega = np.sum(omega, axis = -1) #Nx
            omega = omega/Omega[:, np.newaxis]

            x_bar = np.sum(omega*x, axis = -1)[:, np.newaxis]
            sigma2 = 1/(1 - np.sum(omega, axis = -1))*(np.sum(omega*(x_prev - x_bar)**2 , axis = -1)) #Nx

            cs = (1 - C_pf[i, :])/(Omega_carrot*Ne*C_pf[i, :]) # Needs to be Nx
            r1 = np.sqrt(sigma2/((1/(Ne - 1))*np.sum((x_tilde - x_bar + cs[:, np.newaxis]*(x_prev - x_bar))**2 , axis = -1))) #Nx
            r2 = cs*r1
            #Update Particles
            x_prev = x_bar + gamma*r1*(x_tilde - x_bar) + (gamma*(r2 - 1) + 1)*(x_prev - x_bar)
            hx_prev = np.matmul(H, x_prev)


def pf_update(
    x, hx, y, HCo, HCHo, Neff, min_res, alpha, var_y, kddm_flag, qcpass, maxiter
):
    """lpf update"""

    # prior mean calculation
    xmpf = np.mean(x, axis=1)

    # modify localization matrix (why?)
    # this is probably unnecessary, but exists in
    # the GSI implementation of this code,
    # which does everything in single precision, in
    # which case it is useful to stabilize the filter.
    HC =  HCo * (1 - 1e-5)
    HCH = HCHo * (1 - 1e-5)

    e_flag = 0
    # I haven't implemented any QC checks, so qcpass is always
    # 0 everywhere in my code.
    if np.sum(qcpass) == len(y.T):
        return x, 1

    Nx, Ne = x.shape
    y = y[:,qcpass == 0]
    hx = hx[qcpass == 0, :]
    HC = HC[qcpass == 0, :]
    HCH = HCH[qcpass == 0, :]
    HCH = HCH[:, qcpass == 0]
    Ny = len(y[0])

    # the residual term corresponds to 1-kappa,
    # where kappa is as in Poterjoy (2022).
    # It starts at 1, then each time we perform
    # a tempered update we subtract that update's
    # regularization coefficient (beta) from it until we
    # either reach 0 or, if prescribed, we reach min_res
    # (which controls the point during tempering at which
    # we transition to using an EnKF).
    max_res = 1
    beta = np.ones(Nx)
    beta_y = np.ones(Ny)
    beta_max = 1e100

    res = np.ones(Nx) - min_res
    res_y = np.ones(Ny) - min_res

    niter = 0

    # tempering loop
    while max_res > 0 and min_res < 1:
        niter += 1

        xo = x.copy()
        hx = hx.squeeze()
        hxo = hx.copy()

        # weighing matrices that will store
        # log of weights and normal weights,
        # respectively, for state and obs space variables
        lomega = np.zeros((Nx, Ne))
        lomega_y = np.zeros((Ny, Ne))
        omega = np.ones((Nx, Ne)) / Ne
        omega_y = np.ones((Ny, Ne)) / Ne

        # plague be upon ye
        wo = np.zeros((Ny, Ne))

        # get likelihoods to be used in weight calculations
        # (here, we're just computing gaussian likelihoods from obs/priors)
        for i in range(Ny):

            for n in range(Ne):

                wo[i, n] = MISC.gaussian_L(hxo[i, n], y[:,i], var_y)

            wo[i, :] /= np.sum(wo[i, :])

        if np.isnan(wo).any():
            return x, 1

        # calculate regularization coefficients;
        # see pf_utils for more info
        beta_y, res_y = MISC.get_reg(Ny, Ny, Ne, HCH, wo, Neff, res_y, beta_max)
        beta, res = MISC.get_reg(Nx, Ny, Ne, HC, wo, Neff, res, beta_max)


        # observation loop
        for i in range(Ny):

            # skip obs if impact is low
            if 1 > 0.98 * Ne * np.sum(wo[i, :] ** 2):
                continue

            # used multiple times in localized weight calculations,
            # so we consolidate
            wt = Ne * wo[i, :] - 1
            loc = HC[i]
            locH = HCH[i]

            # (log of) localized weight vectors for model space variables
            # see Poterjoy (2019, equation 10)
            for j in range(Nx):
                if beta[j] == beta_max:
                    continue
                dum = np.log(wt * loc[j] + 1)
                lomega[j, :] -= dum
                # this is for numerical stability?
                lomega[j, :] -= np.min(lomega[j, :])

            # (log of) localized weight vectors for obs space variables
            # see Poterjoy (2019, equation 10)
            for j in range(Ny):
                if beta_y[j] == beta_max:
                    continue
                dum = np.log(wt * locH[j] + 1)
                lomega_y[j, :] -= dum
                lomega_y[j, :] -= np.min(lomega_y[j, :])

            # get weights from log weights...
            omega = np.exp(-lomega / beta[:, None])
            omega_y = np.exp(-lomega_y / beta_y[:, None])
            # ... and normalize them
            omega /= np.sum(omega, axis=1)[:, None]
            omega_y /= np.sum(omega_y, axis=1)[:, None]

            # localized posterior mean in model and obs space
            xmpf = np.sum(omega * xo, axis=1)
            hxmpf = np.sum(omega_y * hxo, axis=1)


            # skip update step if few particles removed
            w = omega_y[i, :]
            if 1 > 0.98 * Ne * np.sum(w**2):
                continue

            # localized posterior variance in model and obs space
            var_a_y = np.sum(omega_y * (hxo - hxmpf[:, None]) ** 2, axis=1)
            norm_a_y = 1 - np.sum(omega_y**2, axis=1)
            var_a_y /= norm_a_y

            var_a = np.sum(omega * (xo - xmpf[:, None]) ** 2, axis=1)
            norm_a = 1 - np.sum(omega**2, axis=1)
            var_a /= norm_a


            w = omega_y[i, :] 
            wneff = 1 / np.sum(w ** 2)

            if wneff < Neff - 0.1:
                betaw = MISC.find_beta(-np.log(w), Neff)
                w = w ** (1 / betaw)  
                w /= np.sum(w)  
                wneff = 1 / np.sum(w ** 2)


            if np.isnan(xmpf).any():
                return x, 1

            # resample particles according to the computed weights
            ind = MISC.sampling(hxo[i, :], w, Ne)

            # merge prior and sampled particles; see pf_utils.py
            x = _pf_merge(x, xo[:, ind], HC[i, :], Ne, xmpf, var_a, alpha)
            hx = _pf_merge(
                hx, hxo[:, ind], HCH[i, :], Ne, hxmpf, var_a_y, alpha
            )



        # see pf_utils for documentation
        if kddm_flag == 1:
            for j in range(Nx):
                if np.var(x[j, :]) > 0:
                    x[j, :] = MISC.kddm(x[j, :], xo[j, :], omega[j, :])

            xmpf = np.mean(x, axis=1)

            for j in range(Ny):
                hx[j, :] = MISC.kddm(hx[j, :], hxo[j, :], omega_y[j, :])

        # kddkm good!

        max_res = np.max(res)
        if niter == maxiter:
            break


    #var_infl = np.ones(Ny) / min_res
    # if you want to use an enkf for the last tempering step - not implemented/tested by me
    # if max(min_res) > 0:
    #xmpf, x = enkf_update_tempered(x, hx, y, var_y, HC, HCH, 0.6, var_infl)
    
    #return xmpf, x.T, e_flag
    return x, e_flag

def _pf_merge(x, xs, loc, Ne, xmpf, var_a, alpha):
    ''' See Poterjoy (2019) section 3b '''

    # if no localization is happening, use bootstrap PF moments
    if (loc == 1).all():
        xmpf = np.mean(xs, axis=1)
        var_a = np.var(xs, axis=1)
    
    c = (1 - loc) / loc

    
    xs = xs - xmpf[:, None]
    x = x - xmpf[:, None]

    v1 = np.sum(xs**2, axis=1) #Nx
    v2 = np.sum(x**2, axis=1) #Nx
    v3 = np.sum(x * xs, axis=1) #Nx
    
    c2 = c * c
    r1 = v1 + c2* v2 + 2 * c * v3
    r2 = c2 / r1

    r1 = alpha * np.sqrt((Ne - 1) * var_a / r1)
    r2 = np.sqrt((Ne - 1) * var_a * r2)

    
    if alpha < 1:
        m1 = np.sum(xs, axis=1) / Ne
        m2 = np.sum(x, axis=1) / Ne
        v1 -= Ne * m1**2
        v2 -= Ne * m2**2
        v3 -= Ne * m1 * m2
        
        T1 = v2
        T2 = 2 * (r1 * v3 + r2 * v2)
        T3 = v1 * r1**2 + v2 * r2**2 + 2 * v3 * r1 * r2 - (Ne - 1) * var_a
        alpha2 = (-T2 + np.sqrt(T2**2 - 4 * T1 * T3)) / (2 * T1)
        
        r2 += alpha2
    
    xa = np.zeros_like(xs)
    pfm = np.zeros(x.shape[0])
    
    for n in range(Ne):
        xa[:, n] = xmpf + r1 * xs[:, n] + r2 * x[:, n]
        pfm += xa[:, n]
    
    pfm /= Ne
    
    for n in range(Ne):
        xa[:, n] = xmpf + (xa[:, n] - pfm)
        nanind = np.isnan(xa[:, n])
        xa[nanind, n] = xmpf[nanind] + xs[nanind, n]
    
    return xa