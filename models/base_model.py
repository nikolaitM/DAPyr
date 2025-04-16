from numbalsoda import lsoda
import numpy as np
import copy

def model(x, dt, T, funcptr):
      #funcptr = rhs.address
      #tspan = np.linspace(0, dt*T, 100)
      tspan = np.array([0, dt])
      usol = copy.deepcopy(x)
      for t in range(T):
            usol, success = lsoda(funcptr, usol, tspan)
            usol = usol[-1, :]
      #usol, success = dop853(funcptr, x, tspan)
      return usol 
