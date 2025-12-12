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
            true_obs_params = {'mu1': -2.0, 'sigma1': 1, 'mu2': 3.0, 'sigma2':.5, 'threshold': 0}
            true_obs_dist = 1

            e_true_sdg = dap.Expt('test1', {'expt_flag': 1,
                                   'model_flag':1,
                                   'seed': 1,
                                   'true_obs_err_dist': true_obs_dist,
                                   'true_obs_err_params': true_obs_params,
                                   'Ne':20,
                                   'T': 5})
            dap.runDA(e_true_sdg)
            xa = e_true_sdg.x_ens.flatten()
            self.assertTrue(np.allclose(xa_true, xa))

      def test_assim_change_assumed_obs_err_dist(self):

            xa_true = np.loadtxt('./tests/states/xa_assumed_obs_error_sd_gaussian.txt', delimiter = ',')
            assumed_obs_params = {'mu1': -2.0, 'sigma1': 1, 'mu2': 3.0, 'sigma2':.5, 'threshold': 0}
            assumed_obs_dist = 1

            e_assumed_sdg = dap.Expt('test1', {'expt_flag': 1,
                                   'model_flag':1,
                                   'seed': 1,
                                   'assumed_obs_err_dist': assumed_obs_dist,
                                   'assumed_obs_err_params': assumed_obs_params,
                                   'Ne':20,
                                   'T': 5})
            dap.runDA(e_assumed_sdg)
            xa = e_assumed_sdg.x_ens.flatten()
            self.assertTrue(np.allclose(xa_true, xa))

if __name__ == '__main__':
    unittest.main()
