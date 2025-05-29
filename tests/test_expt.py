import sys
sys.path.insert(0,'../src/')
import unittest
import DAPpyr as dap
from unittest.mock import patch
import numpy as np
import copy
import DAPpyr.MISC
import DAPpyr.Exceptions



#This is a mock function that will replace the Expt._spinup method.
#The actual _spinup method incorporates random numbers, and is therefore
#not reproducible for each initialization of an Expt.
#This mock function instead loads in pre-saved xf_0, xt, and Y states
#All other parameters are the default parameters set by Expt

#Note: if you modify an experiment after initializing with this method, 
#it will not adjust xf_0, xt, or Y.
#(i.e. changing Ne will not change the shape of xf_0)
def loadStates(self, Nx, Ne, dt, T, tau, funcptr, NumPool, sig_y, h_flag, H):
    match Nx:
        case 3: #L63
            xf_0 = np.load('./states/L63_xf_0.npy')
            xt = np.load('./states/L63_xt.npy')
            Y = np.load('./states/L63_Y.npy')
        case 40: #L96
            xf_0 = np.load('./states/L96_xf_0.npy')
            xt = np.load('./states/L96_xt.npy')
            Y = np.load('./states/L96_Y.npy')
        case 480: #L05
            xf_0 = np.load('./states/L05_xf_0.npy')
            xt = np.load('./states/L05_xt.npy')
            Y = np.load('./states/L05_Y.npy')
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
        self.assertEqual(self.expt.getParam('Ne'), 10)
    
    def test_modParam(self):
        expt = dap.Expt('test', {'dt':0.01})
        self.assertEqual(expt.getParam('dt'), 0.01)
        expt.modExpt({'dt':0.001})
        self.assertEqual(expt.getParam('dt'), 0.001)
        del(expt)
    
    def test_mock(self):
        self.patcher.start()
        model_flags = [0, 1, 2]
        Lstrings = ['L63', 'L96', 'L05']
        paramsList = ['xf_0', 'xt', 'Y']
        for flag, L in zip(model_flags, Lstrings):
            expt = dap.Expt('test', {'model_flag': flag})
            for par in paramsList:
                expt_par = expt.getParam(par)
                init_par = np.load('./states/{}_{}.npy'.format(L, par))
                np.testing.assert_array_equal(expt_par, init_par)
            del(expt)
        self.patcher.stop()
        
    def test_copyStates(self):
        params = ['xf_0', 'xt', 'Y']
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'dt': 0.05})
        for par in params:
            self.assertRaises(AssertionError, np.testing.assert_array_equal, e1.getParam(par), e2.getParam(par))
        e2.copyStates(e1)
        for par in params:
            np.testing.assert_equal(e1.getParam(par), e2.getParam(par))
        
        #Test if changing one chages the other
        xt1 = e1.getParam('xt')
        xt1[:, :] = 0
        self.assertRaises(AssertionError, np.testing.assert_array_equal, e1.getParam('xt'), e2.getParam('xt'))
    def test_copyMismatchNx(self):
        self.patcher.start()
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'model_flag':1})
        Nx1, Nx2 = e2.getParam('Nx'), e1.getParam('Nx')
        self.assertRaises(DAPpyr.Exceptions.MismatchModelSize, e2.copyStates, e1)
        self.patcher.stop()
    
    def test_copyMismatchT(self):
        self.patcher.start()
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'T': 200})
        self.assertRaises(DAPpyr.Exceptions.MismatchTimeSteps, e2.copyStates, e1)
        self.patcher.stop()

    def test_copyMismatchObs(self):
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'obf': 2})
        self.assertRaises(DAPpyr.Exceptions.MismatchObs, e2.copyStates, e1)

    def test_copyExptDiffTime(self):
        e1 = dap.Expt('test1')
        e2 = dap.Expt('test2', {'T': 50})
        e2.copyStates(e1)
        self.assertEqual(e2.getParam('xt').shape, (3, 50))

    def test_getBasicParams(self):
        expt = dap.Expt('test')
        self.assertEqual((10, 3, 100, 0.01), expt.getBasicParams())
    
    def test_getStates(self):
        self.patcher.start()
        expt = dap.Expt('test')
        params = ['xf_0', 'xt', 'Y']
        xf_0, xt, Y = expt.getStates()
        for par, var in zip(params, [xf_0, xt, Y]):
            np.testing.assert_equal(var, np.load('./states/L63_{}.npy'.format(par)))
        self.patcher.stop()
    
    def test_equality(self):
        self.patcher.start()
        e1 = dap.Expt('test')
        e2 = dap.Expt('test2')
        self.assertTrue(e1 == e2)
        self.patcher.stop()
        e3 = dap.Expt('test3')
        self.assertTrue(e1 == e3)
        e4 = dap.Expt('test4', {'model_flag': 1})
        self.assertRaises(AssertionError, self.assertTrue, e1 == e4)

    #TODO Test Loading and Saving Expts
    def test_loadParamFile(self):
        pass

    def test_loadExpt(self):
        pass

    def test_saveExpt(self):
        pass

    def test_shallowcopy(self):
        e1 = dap.Expt('test')
        e2 = copy.copy(e1)
        e1.states['xf_0'] = np.zeros_like(e2.getParam('xf_0'))
        self.assertTrue(np.allclose(e1.states['xf_0'],  e2.getParam('xf_0')))

    def test_deepcopy(self):
        e1 = dap.Expt('test')
        e2 = copy.deepcopy(e1)
        xf1 = e1.getParam('xf_0')
        xf2 = e2.getParam('xf_0')
        xf1 = np.zeros_like(xf1)
        self.assertFalse(np.allclose(xf1, xf2))

class TestMisc(unittest.TestCase):

    def setUp(self):
        return super().setUp()
    def test_create_periodic(self):
        pass

    def test_SV(self):
        Nx, Ne = 40, 10 #Basic setup
        xa = np.ones((Nx, Ne))
        sv_xf = np.ones((Nx, Ne))
        

class TestDA(unittest.TestCase):
    def test_EnSRF(self):
        pass

    def test_LPF(self):
        pass


if __name__ == '__main__':
    unittest.main()