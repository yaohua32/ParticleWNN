# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:05:31 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:05:31 
#  */
import numpy as np 
import torch 
import sys
import os
from torch.autograd import Variable
#
from Utils.TestFun import TestFun
import Problems.Module_Spatial as Module
#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class Problem(Module.Problem):
    '''
    The poisson problem:
        - u''(x) = f(x)   in [-2,2]
          u(x)   = g(x)   in {-2,2}
    #
          u(x) = x * cos( k * x )
          f(x) = k^2 * x * cos( k * x ) + 2 * k * sin( k * x)
    '''
    def __init__(self, test_type:str='Wendland', dtype:np.dtype=np.float32):
        #
        self._dim = 1
        self._name = 'poisson1d'
        #
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
        self._k = 15 * torch.pi
        #
        self._dtype = dtype
        self._test_fun = TestFun(f'{test_type}', self.dim)
    
    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim
    
    @property
    def lb(self):
        return self._lb.reshape(-1,self.dim).astype(self._dtype)

    @property
    def ub(self):
        return self._ub.reshape(-1,self.dim).astype(self._dtype)

    def _fun_u(self, x:torch.tensor=None)->torch.tensor:
        '''
        The solution u
        Input  x: size(?,d)
                    or
               None
        Output  u: size(?,1)
                    or
                u: size(?,1)
                x: size(?,d)
        '''
        if x is not None:
            assert len(x.shape)==2
            return x * torch.cos(self._k * x)
        else:
            x_mesh = np.linspace(self._lb, self._ub, 5000)
            x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._dtype))
            #
            u = x * torch.cos(self._k * x)

            return u, x
    
    def fun_u_bd(self, x_list:torch.tensor, model_u=None) -> torch.tensor:
        '''
        Input:   x_list: list= [size(n,d)]*2d
                 model_u:       
        Output:  u_bd: size(n*2d,1) if model_u=None
                    or
                 u_bd_nn: size(n*2d,1) if model_u is given.
        '''
        x_bd = torch.cat(x_list, dim=0)
        #
        if model_u is not None:
            u_bd_nn= model_u(x_bd)

            return u_bd_nn
        else:
            u_bd = self._fun_u(x_bd)

            return u_bd
    
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        The right hand side f
        Input:  x: size(?,d)
        Output: f: size(?,1)
        '''
        assert len(x.shape)==2
        f = self._k**2 * x * torch.cos(self._k * x) + 2. * self._k * torch.sin(self._k * x)

        return f
    
    def strong(self, model_u, x:torch.tensor)->torch.tensor:
        '''
        The strong form:
        Input: model_u
               x:size(?,d)
        Output: pde: size(?,1)
        '''
        ############ variables
        x = Variable(x, requires_grad=True)
        ############# grads 
        u = model_u(x)
        #
        du_dx = self._grad_u(x, u)
        du_d2x = self._Laplace_u([x], du_dx)
        ############## The pde
        left = - du_d2x
        #
        right = self.fun_f(x)
        
        return left - right
    
    def weak(self, model_u, x_scaled:torch.tensor, xc:torch.tensor, 
             R:torch.tensor)->torch.tensor:
        '''
        The weak form
             model_u: the network model
            x_scaled: size(m, d)
                  xc: size(?, 1, d)
                   R: size(?, 1, 1)
        Output: weak_form
        '''
        ###############
        m = x_scaled.shape[0] 
        x = x_scaled * R +xc 
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        ################
        u = model_u(x)
        #
        du_dx = self._grad_u(x, u)
        u, du_dx = u.view(-1, m, 1), du_dx.view(-1, m, self.dim)
        # 
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ###### weak form
        f = self.fun_f(x=x).view(-1, m, 1)
        #
        right = torch.mean( f * v, dim=1)
        left = torch.mean( torch.sum( du_dx * dv, dim=2, keepdims=True), dim=1)

        return (left-right)

    def energy(self, model_u, x:torch.tensor)->torch.tensor:
        '''
        The energy function
        Input: model_u: the network model
                     x: size(?,d)
        Output: the energy function
        '''
        ############ variables
        x = Variable(x, requires_grad=True)
        ############# grads 
        u = model_u(x)
        #
        du_dx = self._grad_u(x, u)
        f = self.fun_f(x)
        ############## The energy
        left = 0.5 * torch.mean(torch.sum(du_dx * du_dx, dim=1, keepdim=True))
        #
        right = torch.mean( u * f )
        
        return left - right