import sys
import unittest
import DAPyr as dap
from unittest.mock import patch
import numpy as np
import copy
import pytest

def loadStates(self, Nx, Ne, dt, T, tau, funcptr, NumPool, h_flag, H):
    '''A mock function that will replace the Expt._spinup method.
    
    The actual _spinup method incorporates a random number generator, and is therefore
    not reproducible for each initialization of an Expt. This mock function 
    instead loads in pre-saved xf_0, xt, and Y states. 
    All other parameters are the default parameters set by Expt
    
    NOTE: If you modify an experiment after initializing with this method, 
    it will not adjust xf_0, xt, or Y. (i.e. changing Ne will not change the shape of xf_0)
    '''
    match Nx:
        case 3: #L63
            xf_0 = np.load('./tests/states/L63_xf_0.npy')
            xt = np.load('./tests/states/L63_xt.npy')
            Y = np.load('./tests/states/L63_Y.npy')
        case 40: #L96
            xf_0 = np.load('./tests/states/L96_xf_0.npy')
            xt = np.load('./tests/states/L96_xt.npy')
            Y = np.load('./tests/states/L96_Y.npy')
        case 480: #L05
            xf_0 = np.load('./tests/states/L05_xf_0.npy')
            xt = np.load('./tests/states/L05_xt.npy')
            Y = np.load('./tests/states/L05_Y.npy')
    return xf_0, xt, Y


class TestExptMethod(unittest.TestCase):
    def setUp(self):
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        self.patcher = patch.object(dap.Expt, '_spinup', new = loadStates)
        #self.patcher.start()    
        self.expt = dap.Expt('test')

    def tearDown(self):
        self.patcher.stop()    

    def test_getParam(self):
        '''Test the getParam method of the Expt class'''
        self.assertEqual(self.expt.getParam('Ne'), 10)
        for par in ['xt', 'xf_0', 'Y']:
            tmp = self.expt.getParam(par)
            tmp[0] = 0
            self.assertFalse(np.allclose(tmp, self.expt.getParam(par)))
    
    def test_modParam(self):
        '''Test the modParam method of the Expt class'''
        expt = dap.Expt('test', {'dt':0.01})
        self.assertEqual(expt.getParam('dt'), 0.01)
        expt.modExpt({'dt':0.001})
        self.assertEqual(expt.getParam('dt'), 0.001)
        del(expt)
    
    def test_mock(self):
        '''Test the mock functionality for unit testing by comparing it the states imported manually'''
        self.patcher.start()
        model_flags = [0, 1, 2]
        Lstrings = ['L63', 'L96', 'L05']
        paramsList = ['xf_0', 'xt', 'Y']
        for flag, L in zip(model_flags, Lstrings):
            expt = dap.Expt('test', {'model_flag': flag})
            for par in paramsList:
                expt_par = expt.getParam(par)
                init_par = np.load('./tests/states/{}_{}.npy'.format(L, par))
                np.testing.assert_array_equal(expt_par, init_par)
            del(expt)
        self.patcher.stop()
        
    def test_copyStates(self):
        '''Test the copyStates method of the Expt class'''
        params = ['xf_0', 'xt', 'Y']
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'dt': 0.1})
        for par in params:
            self.assertRaises(AssertionError, np.testing.assert_array_equal, e1.getParam(par), e2.getParam(par))
        e2.copyStates(e1)
        for par in params:
            np.testing.assert_equal(e1.getParam(par), e2.getParam(par))
        
        #Test if changing one state does not change the other
        e1.states['xt'][:, :] = 0
        self.assertRaises(AssertionError, np.testing.assert_array_equal, e1.getParam('xt'), e2.getParam('xt'))
    
    def test_copyMismatchNx(self):
        '''Test if the copyStates method thrown the correct exception is the model_flags differ'''
        self.patcher.start()
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'model_flag':1})
        Nx1, Nx2 = e2.getParam('Nx'), e1.getParam('Nx')
        self.assertRaises(dap.Exceptions.MismatchModelSize, e2.copyStates, e1)
        self.patcher.stop()
    
    def test_copyMismatchT(self):
        '''Test if the copyStates method throws an exception if one tries to copy an experiment with less time steps than the what is being copied to'''
        self.patcher.start()
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'T': 200})
        self.assertRaises(dap.Exceptions.MismatchTimeSteps, e2.copyStates, e1)
        self.patcher.stop()

    def test_copyMismatchObs(self):
        '''Test if the copyStates method throwns an exception if differing number of observations are being assimilated'''
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'obf': 2})
        self.assertRaises(dap.Exceptions.MismatchObs, e2.copyStates, e1)
    
    @pytest.mark.filterwarnings("ignore:Time steps do not match")
    def test_copyExptDiffTime(self):
        '''Test if using copyStates method takes a subset of timesteps'''
        e1 = dap.Expt('test1') #Default timesteps of 100
        e2 = dap.Expt('test2', {'T': 50}) #Second experiment only runs out to 50
        e2.copyStates(e1)
        self.assertEqual(e2.getParam('xt').shape, (3, 50))

    def test_getBasicParams(self):
        '''Tests the getBasicParams method of the Expt class'''
        expt = dap.Expt('test')
        self.assertEqual((10, 3, 100, 0.01), expt.getBasicParams())
    
    def test_getStates(self):
        '''Tests the getStates method of the Expt class'''
        self.patcher.start()
        expt = dap.Expt('test')
        params = ['xf_0', 'xt', 'Y']
        xf_0, xt, Y = expt.getStates()
        for par, var in zip(params, [xf_0, xt, Y]):
            np.testing.assert_equal(var, np.load('./tests/states/L63_{}.npy'.format(par)))
        self.patcher.stop()
    
    def test_equality(self):
        '''Test the __eq__ method of the Expt class, making sure equality does hold.
        Equality is assessed if ALL stored parameters and states are the same.'''
        self.patcher.start()
        e1 = dap.Expt('test')
        e2 = dap.Expt('test2')
        self.assertTrue(e1 == e2)
        self.patcher.stop()
        e3 = dap.Expt('test3')
        self.assertTrue(e1 == e3)
        e4 = dap.Expt('test4', {'model_flag': 1})
        self.assertRaises(AssertionError, self.assertTrue, e1 == e4)

    def test_shallowcopy(self):
        '''Test the shallowcopy implementation'''
        e1 = dap.Expt('test')
        e2 = copy.copy(e1)
        e1.states['xf_0'] = np.zeros_like(e2.getParam('xf_0'))
        self.assertTrue(np.allclose(e1.states['xf_0'],  e2.getParam('xf_0')))

    def test_deepcopy(self):
        '''Test the deepcopy implementation'''
        e1 = dap.Expt('test')
        e2 = copy.deepcopy(e1)
        xf1 = e1.getParam('xf_0')
        xf2 = e2.getParam('xf_0')
        xf1 = np.zeros_like(xf1)
        self.assertFalse(np.allclose(xf1, xf2))

    def test_plotExpt(self):
        '''Test the thrown exceptions of the pltoExpt function'''
        e1 = dap.Expt('test')
        self.assertRaises(TypeError, dap.plotExpt, 10, 10)
        self.assertRaises(ValueError, dap.plotExpt, e1, 10)
        dap.runDA(e1)
        self.assertRaises(ValueError, dap.plotExpt, e1, -1)

    def test_default_dt(self):
        '''Test that the default dt varies based on model_flag'''
        e1 = dap.Expt('test1', {'model_flag': 0})
        e2 = dap.Expt('test2', {'T': 10, 'model_flag': 1})
        e3 = dap.Expt('test3', {'T': 10, 'model_flag': 2})
        self.assertEqual(e1.basicParams['dt'], 0.01)
        self.assertEqual(e2.basicParams['dt'], 0.05)
        self.assertEqual(e3.basicParams['dt'], 0.05)
        e4 = dap.Expt('test4', {'model_flag': 1, 'dt':0.1})
        self.assertEqual(e4.basicParams['dt'], 0.1)
        #Check that changing the model flag does not change dt to defuault
        e2.modExpt({'model_flag': 0})
        self.assertEqual(e2.basicParams['dt'], 0.05)
        self.assertEqual(e2.modelParams['model_flag'], 0)


if __name__ == '__main__':
    unittest.main()
