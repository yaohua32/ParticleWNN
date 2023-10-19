# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-28 12:22:26 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-28 12:22:26 
#  */
import numpy as np 
import torch 
import scipy.io
from torch.autograd import Variable
#
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#
from Utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from Utils.GenData_Time import GenData
import Problems.Module_Time as Module

class Problem(Module.Problem):

    def __init__(self, Nt_slice:int=2, dtype:np.dtype=np.float64,
                 testFun_type:str='Wendland'):
        '''
        The 1d Burger's problem:
            u_t + uu_x - \beta*u_xx = f(t,x)  in [0,T]*[-1,1]
            u(x,0)   = g(x)   in [-1,1]
            u(t,-1) = u(t,1) = 0
        Input:
            dtype: np.float
            testFun_type: str='Wendland'
            args: {'freq': default=2*np.pi, 'Re': default=40}
        '''
        assert Nt_slice>=2
        self._dim = 1
        self._name = 'burgers_1d'
        self._dtype = dtype
        self._t_mesh = np.linspace(0., 1., Nt_slice)
        #
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
        #
        self._freq = np.pi 
        self._lambda = 0.01/np.pi
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
    def t_mesh(self):
        return self._t_mesh
    
    @property
    def lb(self):
        lb = self._lb.reshape(-1,self.dim)
        return torch.from_numpy(lb.astype(self._dtype))

    @property
    def ub(self):
        ub = self._ub.reshape(-1,self.dim)
        return torch.from_numpy(ub.astype(self._dtype))

    def get_test(self, t_start_loc:int=0, t_end_loc:int=100)->torch.tensor:
        '''
        Input:
            t_start_loc: the location of the start t
            t_end_loc: the location of the end t
        Output:
            u: size(?,1)
            x: size(?,d)
            t: size(?,1)
        '''
        if t_end_loc<=t_start_loc:
            raise ValueError('t_end_loc should be greater than t_start_loc')
        #
        True_data = scipy.io.loadmat('./Problems/data/burgers_1d.mat')
        t_mesh = True_data['t'].flatten()
        x_mesh = True_data['x'].flatten()
        t_mesh, x_mesh = np.meshgrid(t_mesh, x_mesh)
        u_all = True_data['usol']
        #
        t = t_mesh[:,t_start_loc:t_end_loc]
        x = x_mesh[:,t_start_loc:t_end_loc]
        u = u_all[:,t_start_loc:t_end_loc]
        #
        t = torch.from_numpy(t.reshape(-1,1).astype(self._dtype))
        x = torch.from_numpy(x.reshape(-1,1).astype(self._dtype))
        u = torch.from_numpy(u.reshape(-1,1).astype(self._dtype))

        return u, x, t
        
    def fun_bd(self, model:torch.nn.Module, x_list:list[torch.tensor], 
               t:torch.tensor) -> torch.tensor:
        '''
        Input:
            model:
            x_list: list= [size(n,d)]*2d  
            t: size(n,1)
        Output:  
            cond_bd: size(n*2d,1)
        '''
        t_list, x_lb_list, x_ub_list = [], [], []
        for d in range(self.dim):
            t_list.append(t)
            x_lb_list.append(x_list[2*d])
            x_ub_list.append(x_list[2*d+1])
        t = torch.cat(t_list, dim=0)
        x_lb = torch.cat(x_lb_list, dim=0)
        x_ub = torch.cat(x_ub_list, dim=0)
        #
        cond_list = []
        x_lb = Variable(x_lb, requires_grad=True)
        x_ub = Variable(x_ub, requires_grad=True)
        #
        u_lb_nn = model(x_lb, t)
        u_ub_nn = model(x_ub, t)
        #
        cond_list.append(u_lb_nn)
        cond_list.append(u_ub_nn)

        return torch.cat(cond_list, dim=0)
    
    def fun_init(self, model:torch.nn.Module, 
                 x:torch.tensor, t:torch.tensor) -> torch.tensor:
        '''
        Input:
            x: size(?,d)
            t: size(?,1)
        Output:  
            cond_init: size(?,1)
        '''
        u_init_nn = model(x, t)
        u_init = - torch.sin(self._freq * x)

        return u_init_nn - u_init

    def strong_pinn(self, model:torch.nn.Module, x:torch.tensor,
                    t:torch.tensor)->torch.tensor:
        '''
        The strong form residual
        Input: 
            model:
            x:size(?,d)
            t:size(?,1)
        Output: 
            The residual: size(?,1)
        '''
        ############ variables
        x = Variable(x, requires_grad=True)
        t = Variable(t, requires_grad=True)
        x_list = torch.split(x, split_size_or_sections=1, dim=1)
        #############  
        u = model(torch.cat(x_list, dim=1), t)
        #
        dux, dut = self._grad_u(x, u), self._grad_u(t, u)
        Lu = self._Laplace_u(x_list, dux)
        ############## The pde
        eq = dut + u * dux - self._lambda * Lu
        
        return eq

    def weak_particlewnn(self, model:torch.nn.Module, xc:torch.tensor, 
                         tc:torch.tensor, R:torch.tensor,
                         x_mesh:torch.tensor, phi:torch.tensor,
                         dphi_scaled:torch.tensor)->torch.tensor:
        '''
        The weak residual of ParticleWNN 
            (x = x_mesh * R + xc)
        Input:     
            model: the network model
            xc: particles           (The centers of test functions)
            tc: 
            R: radius               (The radius of compact support regions)
            x_mesh: size(m, d)      (Integration points; scaled in B(0,1))
            phi: size(m, 1)         (Test function)
            dphi_scaled: size(m, d) (1st derivative of test function; scaled by R)
        Output: 
            The weak residual: size(?, 1)
        '''
        ###############
        m = x_mesh.shape[0] 
        x = x_mesh * R + xc 
        t = tc.repeat(1, m, 1)
        #
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        t = Variable(t.view(-1,1), requires_grad=True)
        ################
        u = model(x, t)
        dux, dut = self._grad_u(x, u), self._grad_u(t, u)
        #
        u, dux, dut = u.view(-1,m,1), dux.view(-1,m,self.dim), dut.view(-1,m,1)
        #
        dphi = dphi_scaled / R 
        ###### 
        eq = (torch.mean(dut * phi, dim=1) 
              - torch.mean(u**2 * dphi / 2., dim=1) 
              + self._lambda * torch.mean(torch.sum( dux * dphi, dim=2, keepdims=True), dim=1))

        return eq
