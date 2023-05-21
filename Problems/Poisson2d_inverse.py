# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:05:46 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:05:46 
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
    The 2d Poisson's problem:
             -div( k(x,y)*grad(u) )  = f(x,y)    in [-1,1]*[-1,1]
    where
          u(x,y) = sin( beta * x  ) * sin( beta * y )
          k(x,y) = 0.1 + exp(-(x-0.5)**2/0.04-(y-0.5)**2/0.04)
          f(x,y) = - 2 * lambda * beta^2 * u + k(x,y) * u
    '''
    def __init__(self, test_type:str='Wendland', 
                 dtype:np.dtype=np.float64,
                 k_center=None, k_sigma=None):
        #
        self._dim = 2
        self._name = 'poisson2d_inverse'
        #
        self._lb = np.array([-1., -1.])
        self._ub = np.array([1., 1.])
        self._beta = 1. * torch.pi
        #
        self._k_constant = 0.1
        self._k_scale = 1.
        if k_center is not None:
            self._k_center = k_center
            self._k_sigma = k_sigma
        else:
            self._k_center = np.random.uniform(low=self._lb+0.5, high=self._ub-0.5, size=(1,2))
            self._k_sigma = np.random.uniform(low=[0.01, 0.01], high=[0.5, 0.5], size=(1,2))
        #
        self._dtype = dtype
        self._test_fun = TestFun(f'{test_type}', self.dim)
        self._eps = sys.float_info.epsilon
    
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
        Input    x: size(?,d)
                    or
                   None
        Output  u: size(?,1)
                    or
                u: size(?,1)
                x: size(?,d)
        '''
        if x is not None:
            assert len(x.shape)==2
            return torch.sin(self._beta * x[:,0:1]) * torch.sin(self._beta * x[:,1:])
        else:
            x_mesh = np.linspace(self._lb, self._ub, 50)
            x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
            x = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
            x = torch.from_numpy(x.astype(self._dtype))
            #
            u = torch.sin(self._beta * x[:,0:1]) * torch.sin(self._beta * x[:,1:])

            return u, x

    def _fun_k(self, x:torch.tensor=None)->torch.tensor:
        '''
        Input      x: size(?,d)
                    or
                   None
        Output  k: size(?,1)
                    or
                k: size(?,1)
                x: size(?,d)
        '''
        if x is not None:
            assert len(x.shape)==2
            exp_term = 0.
            for i in range(self.dim):
                exp_term += (x[:,i:i+1]-self._k_center[0,i])**2 / self._k_sigma[0,i]
            #
            return self._k_constant + self._k_scale * torch.exp( - exp_term)
        else:
            x_mesh = np.linspace(self._lb, self._ub, 50)
            x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
            x = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
            x = torch.from_numpy(x.astype(self._dtype))
            #
            exp_term = 0.
            for i in range(self.dim):
                exp_term += (x[:,i:i+1]-self._k_center[0,i])**2 / self._k_sigma[0,i]
            #
            return self._k_constant + self._k_scale * torch.exp( - exp_term), x
        
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
        Input: x: size(?,d)
        Output: f: size(?,1)
        '''
        assert len(x.shape)==2
        #
        k = self._fun_k(x)
        u = self._fun_u(x)
        part1 = - 2*self._beta**2 * k * u 
        #
        du_x = self._beta * torch.cos(self._beta * x[:,0:1]) * torch.sin(self._beta * x[:,1:])
        du_y = self._beta * torch.sin(self._beta * x[:,0:1]) * torch.cos(self._beta * x[:,1:])
        dk_x = - (k-self._k_constant) * 2. * (x[:,0:1]-self._k_center[0,0]) / self._k_sigma[0,0]
        dk_y = - (k-self._k_constant) * 2. * (x[:,1:2]-self._k_center[0,1]) / self._k_sigma[0,1]
        part2 = du_x * dk_x + du_y * dk_y
        #
        return - part1 -part2
    
    def get_sensors(self, n_inside:int, n_bd_each_side:int)->torch.tensor:
        '''
        Get the sensors' position
            n_inside: size of inside sensors
            n_bd_each_side: size of sensors on one boundary side
        Output:
            x_in: positions of sensors inside the domain
            x_bd: positions of sensors on the boundary
        '''
        from scipy.stats import qmc
        lhs_x = qmc.LatinHypercube(self.dim)
        #  sensors inside
        x_in = qmc.scale(lhs_x.random(n_inside), self._lb, self._ub)
        # sensors on the boundary
        x_bd_list = []
        for d in range(self.dim):
            mesh = np.linspace(self._lb[d], self._ub[d], n_bd_each_side).reshape(-1,1)
            x_lb, x_ub= np.concatenate([mesh,mesh], axis=1), np.concatenate([mesh,mesh], axis=1)
            #
            x_lb[:,d:d+1], x_ub[:,d:d+1] = self._lb[d], self._ub[d]
            x_bd_list.extend([torch.from_numpy(x_lb.astype(self._dtype)), 
                              torch.from_numpy(x_ub.astype(self._dtype))])
        
        return torch.tensor(x_in.astype(self._dtype)), x_bd_list

    def strong(self, model_u, model_k, x:torch.tensor)->torch.tensor:
        '''
        The strong form
        Input: model_u
               model_k
               x:size(?,d)
        Output: pde: size(?,1)
        '''
        ############ variables
        x = Variable(x, requires_grad=True)
        x_list = torch.split(x, split_size_or_sections=1, dim=1)
        ############# grads 
        u = model_u(torch.cat(x_list, dim=1))
        k = model_k(torch.cat(x_list, dim=1))
        #
        du = self._grad_u(x, u)
        Lu = self._Laplace_u(x_list, du)
        dk = self._grad_u(x, k)
        ############## The pde
        left = - (k*Lu + torch.sum(du*dk, dim=1, keepdim=True))
        #
        right = self.fun_f(x)
        
        return left - right

    def weak(self, model_u, model_k, x_scaled:torch.tensor, xc:torch.tensor, 
             R:torch.tensor)->torch.tensor:
        '''
        The weak form
             model_u: the network model
             model_k: 
            x_scaled: size(m, d)
                  xc: size(?, 1, d)
                   R: size(?, 1, 1)
        Output: weak_form
        '''
        ###############
        m = x_scaled.shape[0] 
        x = x_scaled * R + xc 
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        ################
        u = model_u(x)
        k = model_k(x)
        #
        du = self._grad_u(x, u)
        u, du, k = u.view(-1, m, 1), du.view(-1, m, self.dim), k.view(-1, m, 1)
        #
        v, dv_scaled, _ = self._test_fun.get_value(x_scaled)
        dv = dv_scaled/R
        ###### weak form
        left = torch.mean( torch.sum( k * du * dv, dim=2, keepdims=True), dim=1) 
        # f
        f = self.fun_f(x=x).view(-1, m, 1) 
        right = torch.mean( f * v, dim=1)

        return left-right