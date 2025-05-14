import numpy as np
import copy
from scipy.optimize import fmin
from . import MISC

def basic_pf_update(xm, hx, Y, var_y):
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

def find_beta(sum_exp, Neff):
    #sum_exp is of size Ne
    Ne = sum_exp.shape[0]
    beta_max = np.max([1, 10*np.max(sum_exp)])
    w = np.exp(-sum_exp)
    ws = np.sum(w)
    if ws > 0:
        w = w/ws
        Neff_init = 1/sum(w**2)
    else:
        Neff_init = 1
    
    if Neff == 1:
        return

    if Neff_init < Neff or ws == 0:
        ks, ke = 1, beta_max
        tol = 1E-5
        #Start Bisection Method

        for i in range(1000):
            w = np.exp(-sum_exp/ks)
            w = w/np.sum(w)
            fks = Neff - 1/np.sum(w**2)
            if np.isnan(fks):
                fks = Neff-1
            
            w = np.exp(-sum_exp/ke)
            w = w/np.sum(w)
            fke = Neff - 1/np.sum(w**2)

            km = (ke + ks)/2
            w = np.exp(-sum_exp/km)
            w = w/np.sum(w)
            fkm = Neff - 1/np.sum(w**2)
            if np.isnan(fkm):
                fkm = Neff-1
            if (ke-ks)/2 < tol:
                break

            if fkm*fks > 0:
                ks = km
            else:
                ke = km
            
        beta = km
        w = np.exp(-sum_exp/beta)
        w = w/np.sum(w)
        Nf = 1/np.sum(w**2)
    else:
        beta = 1
    return beta


def get_reg(Nx, Ne, C, hw, Neff, res, beta_max):
    beta = np.zeros((Nx, ))
    res_ind = np.where(res > 0.0)[0]
    beta[res <= 0.0] = beta_max
    #hw is Ny x Ne
    for i in res_ind:
        wo = 0
        dum = (Ne*hw - 1)*C[:, i, None]
        #ind  = np.where(np.abs(dum) > 0.1)
        #dum[ind] = np.log(dum[ind] + 1 + 1E-10) #Avoid -inf because of np.log([0.0])
        dum = np.log(dum + 1)
        wo = wo - np.sum(dum, axis = 0)
        wo = wo - np.min(wo)
        beta[i] = MISC.find_beta(wo, Neff)
        if res[i] < 1/beta[i]:
            beta[i] = 1/res[i]
            res[i] = 0
        else:
            res[i] = res[i] - 1/beta[i]

        beta[i] = np.min([beta[i], beta_max])
    return beta, res


    #Loop through each state variable
    #If the residual has been reachs, set beta as just beta_max

def lpf_update(x, hx, Y, var_y, H, C_pf, N_eff, gamma, min_res, maxiter, kddm_flag,  e_flag, qcpass):


    #TODO Turn on qaqcpass
    if np.sum(qcpass) == len(Y):
        e_flag = 1
        return np.nan, e_flag
    
    Nx, Ne = x.shape
    HCH = np.matmul(C_pf, H.T)*(1 - 1e-5)

    Y = Y[qcpass == 0, :]
    hx = hx[qcpass == 0, :]
    C_pf = C_pf[qcpass == 0, :]* (1 - 1e-5)
    HCH = HCH[qcpass == 0, :]
    HCH = HCH[:, qcpass == 0]
    Ny = len(Y)
    #TODO Remove obs that don't pass QAQC Here

    max_res = 1.0
    beta = np.ones((Nx,))
    beta_y = np.ones((Ny,))
    beta_max = 1e100
    res = np.ones(beta.shape)
    res_y = np.ones(beta_y.shape)
    niter = 0
    pf_infl = np.ones((Ny,))
    res_infl = np.ones(pf_infl.shape)

    res = res- min_res
    res_y = res_y - min_res

    #Beta stuff begins

    hxo = copy.deepcopy(hx) #From previous assimilated obs
    xo = copy.deepcopy(x) #from previously assimilated obs cycle Nx x Ne
    
    omega = np.ones((Nx, Ne))*(1/Ne) #Nx x Ne
    omega_y = np.ones((Ny, Ne))*(1/Ne)
    lomega = np.zeros_like(omega)
    lomega_y = np.zeros_like(omega_y)

    d = (Y - hxo)**2/(2*var_y)
    #d = d - np.min(d, axis = -1)[:, None]
    #wo = np.exp(-d) + 1E-40
    wo = np.exp(-d)
    wo = wo/np.sum(wo, axis = -1)[:, None]

    if np.any(np.isnan(wo)):
        e_flag = 1
        return np.nan, e_flag

    beta_y, res_y = get_reg(Ny, Ne, HCH, wo, N_eff, res_y, beta_max)
    beta, res = get_reg(Nx, Ne, C_pf, wo, N_eff, res, beta_max)
    wo_ind = np.where(1 < 0.98*Ne*np.sum(wo**2, axis = -1))[0]

    #Obs loop
    for i in wo_ind:
        beta_ind = np.where(beta != beta_max)[0]
        wt = Ne*wo[i, :] - 1 #Ne Array
        C = C_pf[i, beta_ind] #Nxb array
        dum = np.zeros((len(beta_ind), Ne))
        if np.any(C == 1.0):
            dum[C==1.0, :] = np.log(Ne*wo[i, :]) 
        dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1)
        lomega[beta_ind, :] = lomega[beta_ind, :] - dum
        lomega[beta_ind, :] = lomega[beta_ind, :] - np.min(lomega[beta_ind, :], axis = -1)[:, None]

        beta_ind = np.where(beta_y != beta_max)[0]
        wt = Ne*wo[i, :] - 1 #Ne Array
        C = HCH[i, beta_ind] #Nxb array
        dum = np.zeros((len(beta_ind), Ne))
        if np.any(C == 1.0):
            dum[C==1.0, :] = np.log(Ne*wo[i, :]) 
        dum[C!= 1.0, :] = np.log(np.matmul(C[C!=1.0][:, None], wt[None, :]) + 1)
        lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - dum
        lomega_y[beta_ind, :] = lomega_y[beta_ind, :] - np.min(lomega_y[beta_ind, :], axis = -1)[:, None]

        #Normalize
        #lomega is Nx x Ne
        #omega needs to be Nx x Ne

        omega = np.exp(-lomega / beta[:, None])
        omega_y =  np.exp(-lomega_y / beta_y[:, None])

        omegas_y = np.sum(omega_y, axis = -1)[:, None] #Sum over Ensemble Members
        omegas = np.sum(omega, axis = -1)[:, None]

        omega = omega/omegas
        xmpf = np.sum(omega*xo, axis = -1)[:, None]
        omega_y = omega_y/ omegas_y
        hxmpf =np.sum(omega_y*hxo, axis = -1)[:, None]

        if (1 > 0.98*Ne*sum(omega_y[i, :]**2)):
            continue

        var_a = np.sum(omega*(xo - xmpf)**2, axis = -1)[:, None]
        var_a_y = np.sum(omega_y*(hxo - hxmpf)**2, axis = -1)[:, None]

        norm = (1 - np.sum(omega**2, axis = -1))[:, None]
        var_a = var_a/norm
        norm = (1 - np.sum(omega_y**2, axis = -1))[:, None]
        var_a_y = var_a_y/norm
        #ks = np.random.choice(Ne, Ne, p = omega_y[i, :], replace=True)
        ks = MISC.sampling(hxo[i, :], omega_y[i, :], Ne)
        x = pf_merge(x, xo[:, ks], C_pf[i, :], Ne, xmpf, var_a, gamma)
        hx = pf_merge(hx, hxo[:, ks], HCH[i, :], Ne, hxmpf, var_a_y, gamma)
    
    if kddm_flag == 1:
        pass

    return x, e_flag

def pf_merge(x, xs, loc, Ne, xmpf, var_a, alpha):
    if np.all(loc == 1):
        xmpf = np.mean(xs, axis = -1)[:, None]
        var_a = np.var(xs, axis = -1)[:, None]
    c = (1-loc)/loc
    xs = xs - xmpf
    x = x - xmpf
    var_a = var_a[:, 0]
    c2 = c**2
    v1 = np.sum(xs**2, axis = -1)
    v2 = np.sum(x**2, axis = -1)
    v3 = np.sum(x*xs, axis = -1)

    r1 = v1 + c2*v2 + 2*c*v3
    r2 = c2/r1

    r1 = alpha*np.sqrt((Ne-1)*var_a/r1)
    r2 = np.sqrt((Ne-1)*var_a*r2)

    if alpha < 1:
        m1 = np.mean(xs, axis = -1)
        m2 = np.mean(x, axis = -1)
        v1 = v1 - Ne*(m1**2)
        v2 = v2 - Ne*(m2**2)
        v3 = v3 - Ne*(m1*m2)
        T1 = v2
        T2 = 2*(r1*v3 + r2*v2)
        T3 = v1*(r1**2) + v2*(r2**2) + 2*v3*r1*r2 - (Ne-1)*var_a
        alpha2 = (-T2+np.sqrt((T2**2) - 4*T1*T3))/(2*T1)
        r2 = r2+alpha2

    xa = xmpf + r1[:, None]*xs + r2[:, None]*x
    pfm = (np.sum(xa, axis = -1)/Ne)[:, None]
    xa = xmpf + (xa - pfm)

    nanind = np.where(np.isnan(xa))
    xa[nanind] = xmpf[nanind[0], 0] + xs[nanind]

    return xa






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
    y = y[: , qcpass == 0]
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
        hxo = copy.deepcopy(hx)
        if len(hxo.shape) == 1:
             hxo = hxo[None, :]
        if len(hx.shape) == 1:
             hx = hx[None, :]

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