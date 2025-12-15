import sys
import unittest
sys.path.append("src")
import DAPyr as dap
import numpy as np
import pytest
import copy
import DAPyr.MISC
import DAPyr.Exceptions
import DAPyr.OBS_ERRORS

class TestObsErrors(unittest.TestCase):

      def setUp(self):
            self.xf = np.loadtxt('./tests/states/xf.txt', delimiter = ',')
            self.Y = np.loadtxt('./tests/states/Y.txt', delimiter = ',')
            return super().setUp()

      def test_bad_true_obs_params(self):
            e1 = dap.Expt('test1')
            self.assertRaises(DAPyr.Exceptions.BadObsParams, e1.modExpt, {'true_obs_err_params': {'bad param': 4}})
      
      def test_bad_assumed_obs_params(self):
            e1 = dap.Expt('test1', {'expt_flag': 1, 'assumed_obs_err_params': {'bad param': 4}})
            self.assertRaises(DAPyr.Exceptions.BadObsParams, dap.runDA, e1)

      def test_enkf_no_variance(self):
            assumed_obs_params = {'mu1': 0, 'mu2': 0, 'sigma1':1, 'sigma2':1, 'threshold':0}
            assumed_obs_dist = 1
            e1 = dap.Expt('test1', {'expt_flag': 0,
                                    'assumed_obs_err_dist': assumed_obs_dist,
                                    'assumed_obs_err_params': assumed_obs_params})
            self.assertRaises(KeyError, dap.runDA, e1)

      def test_qaqc_no_variance(self):
            assumed_obs_params = {'mu1': 0, 'mu2': 0, 'sigma1':1, 'sigma2':1, 'threshold':0}
            assumed_obs_dist = 1
            e1 = dap.Expt('test1', {'expt_flag': 1,
                                    'qc_flag':1,
                                    'assumed_obs_err_dist': assumed_obs_dist,
                                    'assumed_obs_err_params': assumed_obs_params})
            self.assertRaises(KeyError, dap.runDA, e1)


      def test_assim_change_true_obs_err_dist(self):

            xa_true = np.loadtxt('./tests/states/xa_true_obs_error_sd_gaussian.txt', delimiter = ',')
            
            true_obs_dist = 1
            true_obs_params = {'mu1': -2.0, 'sigma1': 1, 'mu2': 3.0, 'sigma2':.5, 'threshold': 0}
            assumed_obs_params = {'mu' : 0, 'sigma': 1}
            assumed_obs_dist = 0

            rng = np.random.default_rng(58)
            xt = rng.normal(size=(40,20))
            xf = rng.normal(size=(40,20))

            Y = xt + dap.OBS_ERRORS.sample_errors(xt, true_obs_dist, true_obs_params, rng)
            Y = Y[:,0]

            L = dap.OBS_ERRORS.get_likelihood(assumed_obs_dist, assumed_obs_params)
            Nx, Ne = xf.shape
            Ny = Nx
            H = np.eye(Nx)
            C = np.ones((Nx, Nx))
            e_flag = 0
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]

            N_eff = 0.4
            kddm_flag = 0
            maxiter = 1
            alpha = 0.3
            min_res = 0.0
            xa, e_flag = dap.DA.lpf_update(xf, xf, Y[:, None], H, C, N_eff*Ne, alpha, 
                                           min_res, maxiter, kddm_flag, e_flag, qcpass, L)

            self.assertTrue(np.allclose(xa_true, xa))
            
      def test_assim_change_assumed_obs_err_dist(self):

            xa_true = np.loadtxt('./tests/states/xa_assumed_obs_error_sd_gaussian.txt', delimiter = ',')
            
            true_obs_params = {'mu' : 0, 'sigma': 1}
            true_obs_dist = 0
            assumed_obs_dist = 1
            assumed_obs_params = {'mu1': -2.0, 'sigma1': 1, 'mu2': 3.0, 'sigma2':.5, 'threshold': 0}

            rng = np.random.default_rng(58)
            xt = rng.normal(size=(40,20))
            xf = rng.normal(size=(40,20))

            Y = xt + dap.OBS_ERRORS.sample_errors(xt, true_obs_dist, true_obs_params, rng)
            Y = Y[:,0]

            L = dap.OBS_ERRORS.get_likelihood(assumed_obs_dist, assumed_obs_params)
            Nx, Ne = xf.shape
            Ny = Nx
            H = np.eye(Nx)
            C = np.ones((Nx, Nx))
            e_flag = 0
            qcpass = np.zeros((Ny,))
            xm = np.mean(xf, axis = -1)[:, None]

            N_eff = 0.4
            kddm_flag = 0
            maxiter = 1
            alpha = 0.3
            min_res = 0.0
            xa, e_flag = dap.DA.lpf_update(xf, xf, Y[:, None], H, C, N_eff*Ne, alpha, 
                                           min_res, maxiter, kddm_flag, e_flag, qcpass, L)

            self.assertTrue(np.allclose(xa_true, xa))

if __name__ == '__main__':
    unittest.main()
