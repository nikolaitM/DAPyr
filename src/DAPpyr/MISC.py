import numpy as np
from scipy.special import erf
from scipy.interpolate import interp1d


def calc_SV(xa, xf):
      Nx, Ne = xa.shape
      sqrt_Ne = np.sqrt(1/(Ne - 1))
      xma = np.mean(xa, axis = 1)
      xmf = np.mean(xf, axis = 1)
      xpa, xpf = sqrt_Ne*(xa - xma[:, np.newaxis]), sqrt_Ne*(xf - xmf[:, np.newaxis])
      vals, vecs = np.linalg.eigh(np.matmul(xpf.T, xpf))
      val_sort = vals.argsort()[::-1]
      vals = vals[val_sort]
      vecs = vecs[:, val_sort]
      SVe = np.matmul(xpf, vecs)
      SVi = np.matmul(xpa, vecs)
      energy = np.sum(SVe*SVe, axis = 0)/np.sum(SVi*SVi, axis = 0)
      return SVi, SVe, energy, vals



def create_periodic(sigma, m, dx):
      if m % 2 == 0: #Even
            cx = m/2
            x = np.concatenate([np.arange(0, cx), np.arange(cx, 0, -1), np.arange(0, cx), np.arange(cx, 0, -1)])
      else: #Odd
            cx = np.floor(m/2)
            x = np.concatenate([np.arange(0, cx+1), np.arange(cx, 0, -1), np.arange(0, cx+1), np.arange(cx, 0, -1)])
      wlc = np.exp(-((dx*(x))**2)/(2*sigma*2))
      B = np.zeros((m, m))
      for i in range(m):
            B[i, :] = wlc[m - i:2*m - i]
      B = np.where(B < 0, 0, B)
      return B


def find_beta(sum_exp, Neff):
    ''' perform bisection method search for the tempering coefficient beta
    that yields a sufficiently large effective ensemble size'''
    Ne = len(sum_exp)
    beta_max = max(1, 20 * max(sum_exp))
    
    w = np.exp(-sum_exp)
    ws = np.sum(w)
    
    if ws > 0:
        w /= ws
        Neff_init = 1 / np.sum(w ** 2)
    else:
        Neff_init = 1
    
    if Neff == 1:
        return 1
    
    if Neff_init < Neff or ws == 0:
        ks, ke = 1, beta_max
        tol = 1e-3
        
        for _ in range(1000):
            w = np.exp(-sum_exp / ks)
            w /= np.sum(w)
            fks = Neff - 1 / np.sum(w ** 2)
            if np.isnan(fks):
                fks = Neff - 1
            
            w = np.exp(-sum_exp / ke)
            w /= np.sum(w)
            fke = Neff - 1 / np.sum(w ** 2)
            
            km = (ke + ks) / 2
            w = np.exp(-sum_exp / km)
            w /= np.sum(w)
            fkm = Neff - 1 / np.sum(w ** 2)
            if np.isnan(fkm):
                fkm = Neff - 1
            
            if abs(ke - ks) < tol:
                break
            
            if fkm * fks > 0:
                ks = km
            else:
                ke = km
        
        beta = km
        w = np.exp(-sum_exp / beta)
        w /= np.sum(w)
        Nf = 1 / np.sum(w ** 2)
        
        if Nf <= Neff - 1 or np.isnan(Nf):
            print(f'WARNING! Neff is {Nf} but target is {Neff}')
            beta = beta_max
        
    else:
        beta = 1
    
    return beta


'''def sampling(x, w, Ne):
    ind = np.zeros((Ne,))
    b = np.argsort(x)
    a = x[b]
    cum_weight = np.zeros((w.shape[0] + 1,))
    cum_weight[1:] = np.cumsum(w[b])
    offset = 0.0
    base = 1/(Ne - offset)/2

    k = 1
    for n in range(Ne):
        frac = base + (n)/(Ne - offset)

        flag = 0
        while flag==0:
            if (cum_weight[k-1] < frac) and (frac <= cum_weight[k]):
                ind[n] = k-1
                flag = 1
            else:
                k = k+1
    ind = ind.astype(np.int64)
    ind = b[ind]
    ind2 = ind*0
    for n in range(Ne):
        if sum(ind == n) != 0:
            ind2[n] = n
            dum = np.where(ind == n)[0]
            ind[dum[0]] = []
    
    ind0 = np.where(ind2 == 0)[0]
    ind2[ind0] = ind
    ind = ind2
    return ind
'''
def get_reg(Nx, Ny, Ne, C, hw, Neff, res, beta_max):
    # find next regularization coefficient for the current
    # tempering step; uses precomputed particle weights
    # and bisection method on beta (see above) to do so,
    # then reduces the residual term appropriately so we know
    # how much of the factored likelihood we have left to assimilate.
    beta = np.zeros(Nx)

    # print((Ne * hw[0, :] - 1) * C[0, 0])
    # print(np.log((Ne * hw[0, :] - 1) * C[0, 0] + 1))

    for j in range(Nx):
        if res[j] <= 0:
            beta[j] = beta_max
            continue
        
        wo = 0.0
        for i in range(Ny):
            dum = np.log((Ne * hw[i, :] - 1) * C[i, j] + 1)
            wo -= dum
            wo -= np.min(wo)

        # return

        
        beta[j] = find_beta(wo, Neff)
        
        
        if res[j] < 1 / beta[j]:
            beta[j] = 1 / res[j]
            res[j] = 0
        else:
            res[j] -= 1 / beta[j]
        
        beta[j] = min(beta[j], beta_max)
    
    
    # print(beta)
    return beta, res



# glue prior and resampled particles together given posterior moments
# that we're seeking to match and a localization length scale
# merging is done to match the vanilla pf solution when no localization is happening
# and to match the prior when we're at a state very far away from the current obs

def sampling(x, w, Ne):

    # Sort sample
    b = np.argsort(x)
    
    # Apply deterministic sampling by taking value at every 1/Ne quantile
    cum_weight = np.concatenate(([0], np.cumsum(w[b])))
    
    offset = 0.0
    base = 1 / (Ne - offset) / 2
    
    ind = np.zeros(Ne, dtype=int)
    k = 1
    for n in range(Ne):
        frac = base + (n / (Ne - offset))
        while cum_weight[k] < frac:
            k += 1
        ind[n] = k - 1
    ind = b[ind]

    # Replace removed particles with duplicated particles
    ind2 = -999*np.ones(Ne, dtype=int)
    for n in range(Ne):
        if np.sum(ind == n) != 0:
            ind2[n] = n
            dum = np.where(ind == n)[0]
            ind = np.delete(ind, dum[0])
    

    ind0 = np.where(ind2 == -999)[0]
    ind2[ind0] = ind
    ind = ind2
    
    return ind



def gaussian_L(x, y, r):
    return np.exp(-(y - x)**2 / (2 * r)).item()

def kddm(x, xo, w):
    Ne = len(w)
    sig = (max(x) - min(x)) / 6
    npoints = 300
    
    xmin = min(min(xo), min(x))
    xmax = max(max(xo), max(x))
    
    xd = np.linspace(xmin, xmax, npoints)
    qf = np.zeros_like(x)
    cdfxa = np.zeros_like(xd)
    
    for n in range(Ne):
        qf += (1 + erf((x - x[n]) / (np.sqrt(2) * sig))) / (2 * Ne)
        cdfxa += w[n] * (1 + erf((xd - xo[n]) / (np.sqrt(2) * sig))) / 2
    
    interp_func = interp1d(cdfxa, xd, bounds_error=False, fill_value="extrapolate")
    xa = interp_func(qf)
    
    if np.var(xa) < 1e-8:
        print("Warning: Low variance detected in xa")
    
    if np.isnan(qf).any():
        print("Warning: NaN values detected in qf")
    
    return xa
