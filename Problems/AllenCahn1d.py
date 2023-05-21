# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:04:19 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:04:19 
#  */
import numpy as np 
import torch 
import sys
import os
import scipy.io
from torch.autograd import Variable
#
from Utils.TestFun import TestFun
import Problems.Module_Time as Module
#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class Problem(Module.Problem):
    '''
    The Allen-Cahn equation:
          u_t - lambda * u_xx + 5u^3 - 5u = f(t,x)  in [0,T]*[-1,1]
          u(0,x)   = g(x)   in [-1,1]
          u(t,-1) = u(t,1)
          u_x(t,-1) = u_x(t,1)
    where
          g(x) = x^2 * cos(pi * x)
    '''
    def __init__(self, test_type:str='Wendland', dtype:np.dtype=np.float32):
        #
        self._dim = 1
        self._name = 'allen_cahn'
        #
        self._t0 = 0.
        self._tT = 1.
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
        self._k = torch.pi
        self._lambda = 0.0001
        #
        self._dtype = dtype
        self._test_fun = TestFun(f'{test_type}', self.dim)
        #
        self._true_data = scipy.io.loadmat('./Problems/Data/allen-cahn1d.mat')

    @property
    def name(self):
        return self._name
    
    @property
    def dim(self):
        return self._dim

    @property
    def lb(self):
        return np.array([self._t0, self._lb[0]]).reshape(-1, 1+self.dim).astype(self._dtype)

    @property
    def ub(self):
        return np.array([self._tT, self._ub[0]]).reshape(-1, 1+self.dim).astype(self._dtype)
    
    def _fun_u(self, x:torch.tensor=None, t:torch.tensor=None):
        '''
        The ground truth of u 
        Input: x:size(?,d)
               t:size(?,1)
        Return: u: size(?,1)
                    or
                u: size(?,1)
                x: size(?,d)
                t: size(?,1)
        '''
        if x is not None:
            raise NotImplementedError('Explicit solution is not available.')
        else:
            t_mesh = self._true_data['tt'].flatten()
            x_mesh = self._true_data['x'].flatten()
            t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh)
            #
            t = torch.from_numpy(t_mesh.reshape(-1,1).astype(self._dtype))
            x = torch.from_numpy(x_mesh.reshape(-1,1).astype(self._dtype))
            #
            u = torch.from_numpy(self._true_data['uu'].reshape(-1,1).astype(self._dtype))
            
            return u, x, t

    def fun_u_bd(self, x_list:list[torch.tensor], t:torch.tensor, model_u=None)->torch.tensor:
        '''
        Input:   x_list: list= [size(n,1)]*2d
                 t: size(n,1)
                 model       
        Output:  u_lb, u_ub: size(n*2d,1) if model_u is None
                 u_lb_nn, u_ub_nn: size(n*2d,1) if model_u is given.
        '''
        if model_u is not None:
            #
            x_lb = Variable(x_list[0], requires_grad=True)
            x_ub = Variable(x_list[1], requires_grad=True)
            #
            u_lb_nn = model_u(torch.cat([t, x_lb], dim=1))
            u_ub_nn = model_u(torch.cat([t, x_ub], dim=1))
            #
            du_lb_nn = self._grad_u(x_lb, u_lb_nn) 
            du_ub_nn = self._grad_u(x_ub, u_ub_nn)

            return u_lb_nn - u_ub_nn, du_lb_nn - du_ub_nn
        else:
            return torch.zeros_like(t), torch.zeros_like(t)

    def fun_u_init(self, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        Input: x: size(?,d)
               t: size(?,1)
        Output: u: size(?,1)
        '''
        assert len(x.shape)==2 and len(t.shape)==2
        return x**2 * torch.cos(self._k * x)
    
    def fun_f(self, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        Input: x: size(?,d)
               t: size(?,1)
        Output: f: size(?,1)
        '''
        assert len(x.shape)==2 and len(t.shape)==2
        return torch.zeros_like(t)
    
    def strong(self, model_u, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        The strong form
        Input: model_u
                    x:size(?,d)
                    t:size(?,1)
        Output: pde: size(?,1)
        '''
        ############ variables
        x = Variable(x , requires_grad=True)
        t = Variable(t, requires_grad=True)
        ############# grads 
        u = model_u(torch.cat([t,x], dim=1))
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        du_d2x = self._Laplace_u([x], du_dx)
        ############# The pde
        left = du_dt+ 5 * (u**3 - u) - self._lambda * du_d2x
        ############## The right hand side
        right = self.fun_f(x, t)
        
        return left-right
    
    def weak(self, model_u, x_scaled:torch.tensor, xc:torch.tensor, t:torch.tensor, 
             R:torch.tensor)->torch.tensor:
        '''
        The weak form
        Input:  model: the network model
                x_scaled: size(m, d)
                  xc: size(?, 1, d)
                   t: size(?, 1, 1)
                   R: size(?, 1, 1)
        Output: weak_form
        '''
        ###############
        m = x_scaled.shape[0]
        x = x_scaled * R + xc  
        t = t.repeat(1,m,1)
        #
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        t = Variable(t.view(-1,1), requires_grad=True)
        ################
        u = model_u(torch.cat([t,x], dim=1))
        #
        du_dt, du_dx = self._grad_u(t, u), self._grad_u(x, u)
        u, du_dt, du_dx = u.view(-1, m, 1), du_dt.view(-1, m, 1), du_dx.view(-1, m, self.dim)
        #
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ################## weak form
        left = torch.mean(du_dt * v, dim=1)  
        left += 5. * torch.mean( (u**3 -u) * v, dim=1) 
        left += self._lambda * torch.mean(torch.sum( du_dx * dv, dim=2, keepdims=True), dim=1)
        # 
        f = self.fun_f(x=x, t=t).view(-1, m, 1)
        right = torch.mean( f * v, dim=1)

        return left-right