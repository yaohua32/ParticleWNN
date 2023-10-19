# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-27 23:18:55 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-27 23:18:55 
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
                 testFun_type:str='Wendland'):
        '''
        The 2d Poisson's inverse problem:
                -div( k(x,y)*grad(u) )  = f(x,y)    in [-1,1]*[-1,1]
        Input:
            dtype: np.float
            testFun_type: str='Wendland'
        '''
        self._dim = 2
        self._name = 'inverse_poisson_2d'
        self._dtype = dtype
        #
        self._lb = np.array([-1., -1.])
        self._ub = np.array([1., 1.])
        self._freq = 1. * torch.pi
        #
        self._k_constant = 0.1
        self._k_scale = 1.
        try:
            path = './Problems/data/inverse_poisson_2d_sensors.npy'
            saved_set = np.load(path, allow_pickle=True)
            #
            self._k_center = saved_set.item().get('k_center')
            self._k_sigma = saved_set.item().get('k_sigma')
            print('**************** Load k_center and k_sigma *********************')
        except:
            self._k_center = np.random.uniform(
                low=self._lb+0.5, high=self._ub-0.5, size=(1,2))
            self._k_sigma = np.random.uniform(
                low=[0.01, 0.01], high=[0.5, 0.5], size=(1,2))
            print('**************** Generate k_center and k_sigma *********************')
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

    def get_observe(self, Nx_in:int=None, Nx_bd_each_side:int=None)->torch.tensor:
        '''
        Get the observation
            Nx_in: size of inside sensors
            Nx_bd_each_side: size of sensors on each boundary
                or 
               None (means loading from saved file.)
        Output:
            data_observe: dict={'x_ob_in', 'u_ob_in', 'x_ob_bd', 'u_ob_bd'}
        '''
        try:
            path = './Problems/data/inverse_poisson_2d_sensors.npy'
            saved_set = np.load(path, allow_pickle=True)
            x_in = saved_set.item().get('x_sensor_in')
            x_bd = saved_set.item().get('x_sensor_bd')
            print(f'*************** Sensors have been loaded. ******************')
        except:
            from scipy.stats import qmc
            lhs_x = qmc.LatinHypercube(self.dim)
            # sensors inside
            x_in = qmc.scale(lhs_x.random(Nx_in), self._lb, self._ub)
            # sensors on the boundary
            x_bd_list = []
            for d in range(self.dim):
                mesh = np.linspace(self._lb[d], self._ub[d], Nx_bd_each_side).reshape(-1,1)
                x_lb, x_ub= np.concatenate([mesh,mesh], axis=1), np.concatenate([mesh,mesh], axis=1)
                #
                x_lb[:,d:d+1], x_ub[:,d:d+1] = self._lb[d], self._ub[d]
                x_bd_list.extend([x_lb, x_ub])
            x_bd = np.concatenate(x_bd_list, axis=0)
            #
            x_in = torch.from_numpy(x_in.astype(self._dtype))
            x_bd = torch.from_numpy(x_bd.astype(self._dtype))
            print(f'*************** Sensors have been generated. ******************')
        ######################
        uk_in = self.get_test(x_in)
        uk_bd = self.get_test(x_bd)
        data_observe = {'x_ob_in':x_in, 'u_ob_in':uk_in[:,0:1], 
                        'x_ob_bd':x_bd, 'u_ob_bd':uk_bd[:,0:1]}
        
        return data_observe

    def get_test(self, x:torch.tensor=None)->torch.tensor:
        '''
        Input:
            x: size(?,d) 
                or
            None
        Output:
            u/k: size(?,2)
                or
            u/k: size(?,2)
            x: size(?,d)
        '''
        if x is not None:
            u = torch.sin(self._freq * x[:,0:1]) * torch.sin(self._freq * x[:,1:])
            #
            exp_term = 0.
            for i in range(self.dim):
                exp_term += (x[:,i:i+1]-self._k_center[0,i])**2 / self._k_sigma[0,i]
            k = self._k_constant + self._k_scale * torch.exp( - exp_term)

            return torch.cat([u,k], dim=1)
        else:
            x_mesh = np.linspace(self._lb, self._ub, 100)
            x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
            x = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
            x = torch.from_numpy(x.astype(self._dtype))
            #
            u = torch.sin(self._freq * x[:,0:1]) * torch.sin(self._freq * x[:,1:])
            #
            exp_term = 0.
            for i in range(self.dim):
                exp_term += (x[:,i:i+1]-self._k_center[0,i])**2 / self._k_sigma[0,i]
            k = self._k_constant + self._k_scale * torch.exp( - exp_term)

            return torch.cat([u,k], dim=1), x

    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        # The value of f(x)
        uk = self.get_test(x)
        u, k = uk[:,0:1], uk[:,1:]
        #
        part1 = - 2*self._freq**2 * k * u 
        #
        du_x = self._freq * torch.cos(self._freq * x[:,0:1]) * torch.sin(self._freq * x[:,1:])
        du_y = self._freq * torch.sin(self._freq * x[:,0:1]) * torch.cos(self._freq * x[:,1:])
        dk_x = - (k-self._k_constant) * 2. * (x[:,0:1]-self._k_center[0,0]) / self._k_sigma[0,0]
        dk_y = - (k-self._k_constant) * 2. * (x[:,1:2]-self._k_center[0,1]) / self._k_sigma[0,1]
        part2 = du_x * dk_x + du_y * dk_y
        #
        return - part1 -part2

    def strong_pinn(self, model:torch.nn.Module, x:torch.tensor)->torch.tensor:
        '''
        The strong form residual
        Input: 
            model:
            x: size(?,d)
        Output: 
            The residual: size(?,1)
        '''
        ############ variables
        x = Variable(x, requires_grad=True)
        x_list = torch.split(x, split_size_or_sections=1, dim=1)
        ############# grads 
        uk = model(torch.cat(x_list, dim=1))
        u, k = uk[:,0:1], uk[:,1:]
        #
        du, dk = self._grad_u(x, u), self._grad_u(x, k)
        Lu = self._Laplace_u(x_list, du)
        ############## The pde
        eq = - (k*Lu + torch.sum(du*dk, dim=1, keepdim=True)) -  self.fun_f(x)
        
        return eq

    def weak_particlewnn(self, model:torch.nn.Module,
                         xc:torch.tensor, R:torch.tensor, 
                         x_mesh:torch.tensor, phi:torch.tensor, 
                         dphi_scaled:torch.tensor)->torch.tensor:
        '''
        The weak residual of ParticleWNN 
            (x = x_mesh * R + xc)
        Input:     
            model:
            xc: particles           (The centers of test functions)
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
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        ################
        uk = model(x)
        u, k = uk[:,0:1], uk[:,1:]
        du = self._grad_u(x, u)
        #
        u, du, k = u.view(-1, m, 1), du.view(-1, m, self.dim), k.view(-1, m, 1)
        #
        dphi = dphi_scaled/R # (m, d) / (?, 1, 1) = (?, m, d)
        f = self.fun_f(x=x).view(-1, m, 1) 
        ###### weak form
        eq = (torch.mean( torch.sum( k * du * dphi, dim=2, keepdims=True), dim=1) 
              - torch.mean( f * phi, dim=1)) 
        
        return eq