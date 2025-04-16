import numpy as np

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