import sys
import unittest
import DAPyr as dap
import numpy as np
import pytest
import copy
import DAPyr.MISC
import DAPyr.Exceptions

class TestINF(unittest.TestCase):

      def setUp(self):
            self.xf = np.loadtxt('./tests/states/xf.txt', delimiter = ',')
            self.xa = np.loadtxt('./tests/states/xa.txt', delimiter = ',')
            self.Y = np.loadtxt('./tests/states/Y.txt', delimiter = ',')
            return super().setUp()

      def test_RTPS_gamma_0(self):
            gamma = 0.0
            xf = self.xf
            xa = self.xa
            xa_new = dap.INFLATION.do_RTPS(xf, xa, gamma)
            self.assertTrue(np.allclose(xa, xa_new))
      
      def test_RTPP_gamma_0(self):
            gamma = 0.0
            xf = self.xf
            xa = self.xa
            xa_new = dap.INFLATION.do_RTPP(xf, xa, gamma)
            self.assertTrue(np.allclose(xa, xa_new))

      def test_RTPS_preserve_xa(self):
            gamma = 0.0
            xf = self.xf
            xa = self.xa
            xa_copy = copy.deepcopy(xa)
            xf_copy = copy.deepcopy(xf)
            xa_new = dap.INFLATION.do_RTPS(xf, xa, gamma)
            self.assertTrue(np.allclose(xa, xa_copy))
            self.assertTrue(np.allclose(xf, xf_copy))

      def test_RTPP_preserve_xa(self):
            gamma = 0.0
            xf = self.xf
            xa = self.xa
            xa_copy = copy.deepcopy(xa)
            xf_copy = copy.deepcopy(xf)
            xa_new = dap.INFLATION.do_RTPP(xf, xa, gamma)
            self.assertTrue(np.allclose(xa, xa_copy))
            self.assertTrue(np.allclose(xf, xf_copy))

      def test_RTPS(self):
            xa_true = np.loadtxt('./tests/states/xa_rtps_gamma_3.txt', delimiter = ',')
            gamma = 0.3
            xf = self.xf
            xa = self.xa
            xa_new = dap.INFLATION.do_RTPS(xf, xa, gamma)
            self.assertTrue(np.allclose(xa_new, xa_true))
            
      def test_RTPP(self):
            xa_true = np.loadtxt('./tests/states/xa_rtpp_gamma_3.txt', delimiter = ',')
            gamma = 0.3
            xf = self.xf
            xa = self.xa
            xa_new = dap.INFLATION.do_RTPP(xf, xa, gamma)
            self.assertTrue(np.allclose(xa_new, xa_true))

      def test_adaptive_inf(self):
            xf_true = np.loadtxt('./tests/states/xa_inflation_init_8.txt', delimiter = ',')
            init_infs = 0.8
            xf = self.xf
            Y = self.Y
            Ny = len(Y)
            Nx, Ne = xf.shape
            infs = np.ones((Nx,))*init_infs
            infs_y = np.ones((Ny,))*init_infs
            var_infs = np.ones((Nx,))*init_infs
            var_infs_y= np.ones((Ny,))*init_infs
            H = np.eye(Nx)
            var_y = 1.0
            xf_new, infs_new, var_infs_new = dap.INFLATION.do_Anderson2009(xf, xf, Y[:, None], infs, var_infs, H, var_y)
            self.assertTrue(np.allclose(xf_new, xf_true))

if __name__ == '__main__':
    unittest.main()