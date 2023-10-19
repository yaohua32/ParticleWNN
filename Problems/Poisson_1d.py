# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 15:30:28 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 15:30:28 
#  */
import numpy as np 
import torch 
from torch.autograd import Variable
#
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#
from Utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from Utils.GenData import GenData
import Problems.Module as Module

class Problem(Module.Problem):

    def __init__(self, dtype:np.dtype=np.float64,
                 testFun_type:str='Wendland', **args):
        '''
        The 1d poisson problem:
            - u''(x) = f(x)   in [-2,2]
            u(x)   = g(x)   in {-2,2}
        Input:
            dtype: np.float
            testFun_type: str='Wendland'
            args: {'freq':default=15*np.pi}
        '''
        self._dim = 1
        self._name = 'poisson_1d'
        self._dtype = dtype
        #
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
        try:
            self._freq = args['freq']
        except:
            self._freq = 15.*np.pi
        #
        self._testFun_particlewnn = TestFun_ParticleWNN(testFun_type, self.dim)
        self._gen_data = GenData(d=self._dim, x_lb=self._lb, 
                                 x_ub=self._ub, dtype=dtype)
    
    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim
    
    @property
    def lb(self):
        lb = self._lb.reshape(-1,self.dim)
        return torch.from_numpy(lb.astype(self._dtype))

    @property
    def ub(self):
        ub = self._ub.reshape(-1,self.dim)
        return torch.from_numpy(ub.astype(self._dtype))
    
    def get_test(self, x:torch.tensor=None)->torch.tensor:
        '''
        Input:
            x: size(?,d) 
                or
            None
        Output:
            u: size(?,1)
                or
            u: size(?,1)
            x: size(?,1)
        '''
        if x is not None:
            return x * torch.cos(self._freq * x)
        else:
            x_mesh = np.linspace(self._lb, self._ub, 10000)
            x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._dtype))
            #
            u = x * torch.cos(self._freq * x)
            return u, x
    
    def fun_bd(self, model:torch.nn.Module, x_list:list[torch.tensor]) -> torch.tensor:
        '''
        Input:
            model: 
            x_list: list= [size(n,d)]*2d     
        Output:
            cond_bd: size(n*2d,1)
        '''
        x_bd = torch.cat(x_list, dim=0)
        #
        cond_pred = model(x_bd)
        cond_true = self.get_test(x_bd)

        return cond_pred - cond_true
    
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        f = (self._freq**2 * x * torch.cos(self._freq * x) 
             + 2. * self._freq * torch.sin(self._freq * x))

        return f
    
    def strong_pinn(self, model:torch.nn.Module, x:torch.tensor)->torch.tensor:
        '''
        The strong form residual
        Input: 
            model:
            x:size(?,d)
        Output: 
            The residual: size(?,1)
        '''
        ############ 
        x = Variable(x, requires_grad=True)
        x_list = torch.split(x, split_size_or_sections=1, dim=1)
        ############# 
        u = model(torch.cat(x_list, dim=1))
        #
        du = self._grad_u(x, u)
        Lu = self._Laplace_u(x_list, du)
        ##############
        eq = - Lu - self.fun_f(x)
        
        return eq

    def weak_particlewnn(self, model:torch.nn.Module, xc:torch.tensor, R:torch.tensor,
                         x_mesh:torch.tensor, phi:torch.tensor, 
                         dphi_scaled:torch.tensor)->torch.tensor:
        '''
        The weak residual of ParticleWNN 
            (x = x_mesh * R + xc)
        Input:     
            model: the network model
            xc: particles                     (The centers of test functions)
            R: radius                         (The radius of compact support regions)
            x_mesh: size(m, d)                (Integration points; scaled in B(0,1))
            phi: size(m, 1)                   (Test function)
            dphi_scaled: size(m, d)           (1st derivative of test function; scaled by R)
        Output: 
            The weak residual: size(?, 1)
        '''
        ###############
        m = x_mesh.shape[0] 
        x = x_mesh * R +xc 
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        ################
        u = model(x)
        du = self._grad_u(x, u)
        #
        u, du = u.view(-1, m, 1), du.view(-1, m, self.dim)
        dphi = dphi_scaled/R 
        f = self.fun_f(x=x).view(-1, m, 1)
        ######
        eq = (torch.mean( torch.sum( du * dphi, dim=2, keepdims=True), dim=1) 
              - torch.mean( f * phi, dim=1))
        
        return eq
    
