# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-27 13:55:16 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-27 13:55:16 
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
        The Kovasznay flow problem (2d): https://en.wikipedia.org/wiki/Kovasznay_flow
        Input:
            dtype: np.float
            testFun_type: str='Wendland'
            args: {'freq': default=2*np.pi, 'Re': default=40}
        '''
        self._dim = 2
        self._name = 'NS_Kovasznay_2d'
        self._dtype = dtype
        try:
            self._freq = args['freq']
        except:
            self._freq = 2. * np.pi
        try:
            self._Re = args['Re']
        except:
            self._Re = 40
        #
        self._lb = np.array([-1., -1.])
        self._ub = np.array([1., 1.])
        #
        self._nu = 0.5*self._Re - np.sqrt(0.25*self._Re**2 + self._freq**2)
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
            uv: size(?,2)
                or
            uv: size(?,2)
            x: size(?,d)
        '''

        if x is not None:
            u = 1 - torch.exp(self._nu*x[:,0:1]) * torch.cos(self._freq*x[:,1:])
            v = self._nu * torch.exp(self._nu*x[:,0:1]) * torch.sin(self._freq*x[:,1:]) / self._freq

            return torch.cat([u,v], dim=1)
        else:
            x_mesh = np.linspace(self._lb, self._ub, 100)
            x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
            x = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
            x = torch.from_numpy(x.astype(self._dtype))
            #
            u = 1 - torch.exp(self._nu*x[:,0:1]) * torch.cos(self._freq*x[:,1:])
            v = self._nu * torch.exp(self._nu*x[:,0:1]) * torch.sin(self._freq*x[:,1:]) / self._freq

            return torch.cat([u,v], dim=1), x

    def get_test_p(self, x:torch.tensor=None)->torch.tensor:
        '''
        Input:
            x: size(?,d) 
                or
            None
        Output:
            p: size(?,1)
                or
            p: size(?,1)
            x: size(?,d)
        '''

        if x is not None:
            p = 0.5 * (1. - torch.exp(2*self._nu*x[:,0:1]))

            return p
        else:
            x_mesh = np.linspace(self._lb, self._ub, 100)
            x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
            x = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)], axis=1)
            x = torch.from_numpy(x.astype(self._dtype))
            #
            p = 0.5 * (1. - torch.exp(2*self._nu*x[:,0:1]))

            return p, x

    def get_observe(self, Nx_in:int, **args)->dict:
        '''
        Input: 
            Nx_int: the number of integration points for computing the integral of p
        Output: 
            data_observe: dict={'x_ob', 'p_int'} 
            args: {'int_method':default='hypercube'}
        '''
        try:
            x_in = self._gen_data.get_in(Nx_size=Nx_in, method=args['int_method'])
        except:
            x_in = self._gen_data.get_in(Nx_size=Nx_in)
        p_int = (2. - (np.exp(2*self._nu)-np.exp(-2*self._nu)) / (2*self._nu))/4.
        #
        data_dict = {}
        data_dict['x_ob'] = x_in
        data_dict['p_int'] = torch.tensor(p_int)

        return data_dict

        
    def fun_bd(self, model_uv:torch.nn.Module, model_p:torch.nn.Module, 
               x_list:list[torch.tensor]) -> torch.tensor:
        '''
        Input:
            model:
            x_list: list= [size(n,d)]*2d  
        Output:  
            cond_bd: size(n*2d,1)
        '''
        x_bd = torch.cat(x_list, dim=0)
        #
        uv_pred, p_pred= model_uv(x_bd), model_p(x_bd)
        u_pred, v_pred = uv_pred[:,0:1], uv_pred[:,1:2]
        uvp_true = self.get_test(x_bd)
        u_true, v_true, p_true = uvp_true[:,0:1], uvp_true[:,1:2], uvp_true[:,2:3]
        #
        cond_list = []
        cond_list.append(u_pred-u_true)
        cond_list.append(v_pred-v_true)

        return torch.cat(cond_list, dim=0)
    
    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        return torch.zeros((x.shape[0], 1))

    def strong_pinn(self, model_uv:torch.nn.Module, model_p:torch.nn.Module,
                    x:torch.tensor)->torch.tensor:
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
        uv, p = model_uv(torch.cat(x_list, dim=1)), model_p(torch.cat(x_list, dim=1))
        u, v= uv[:,0:1], uv[:,1:2]
        #
        du = self._grad_u(x, u)
        dv = self._grad_u(x, v)
        dp = self._grad_u(x, p)
        Lu = self._Laplace_u(x_list, du)
        Lv = self._Laplace_u(x_list, dv)
        ##############
        eq_u = u*du[:,0:1] + v*du[:,1:] + dp[:,0:1] - (1./self._Re) * Lu 
        eq_v = u*dv[:,0:1] + v*dv[:,1:] + dp[:,1:] - (1./self._Re) * Lv
        eq_div = du[:,0:1] + dv[:,1:]
        
        return eq_u, eq_v, eq_div

    def weak_particlewnn(self, model_uv:torch.nn.Module, model_p:torch.nn.Module, 
                         xc:torch.tensor, R:torch.tensor,
                         x_mesh:torch.tensor, phi:torch.tensor,
                         dphi_scaled:torch.tensor)->torch.tensor:
        '''
        The weak residual of ParticleWNN 
            (x = x_mesh * R + xc)
        Input:     
            model: the network model
            xc: particles                  (The centers of test functions)
            R: radius                      (The radius of compact support regions)
            x_mesh: size(m, d)             (Integration points; scaled in B(0,1))
            phi: size(m, 1)                (Test function)
            dphi_scaled: size(m, d)        (1st derivative of test function; scaled by R)
        Output: 
            The weak residual: size(?, 1)
        '''
        ###############
        m = x_mesh.shape[0] 
        x = x_mesh * R + xc 
        #
        x = Variable(x.view(-1, self.dim), requires_grad=True)
        ################
        uv, p = model_uv(x), model_p(x)
        u, v = uv[:,0:1], uv[:,1:2]
        u, v, p = u.view(-1,m,1), v.view(-1,m,1), p.view(-1,m,1)
        #
        du, dv, dp = self._grad_u(x, u), self._grad_u(x, v), self._grad_u(x, p)
        du, dv, dp = du.view(-1,m,self.dim), dv.view(-1,m,self.dim), dp.view(-1,m,self.dim)
        #
        dphi = dphi_scaled / R 
        ###### 
        eq_u = (torch.mean(torch.sum((u*du[:,:,0:1] + v*du[:,:,1:]) * phi, dim=2, keepdim=True), dim=1)
                + (1./self._Re) * torch.mean(torch.sum(du * dphi, dim=2, keepdim=True),dim=1) 
                + torch.mean(torch.sum( dp[:,:,0:1] * phi, dim=2, keepdim=True), dim=1))
                # - torch.mean(torch.sum( p * dphi[:,:,0:1], dim=2, keepdim=True), dim=1))
        eq_v = (torch.mean(torch.sum((u*dv[:,:,0:1] + v*dv[:,:,1:]) * phi, dim=2, keepdim=True), dim=1)
                + (1./self._Re) * torch.mean(torch.sum(dv * dphi, dim=2, keepdim=True),dim=1)
                + torch.mean(torch.sum( dp[:,:,1:2] * phi, dim=2, keepdim=True), dim=1))
                # - torch.mean(torch.sum( p * dphi[:,:,1:], dim=2, keepdim=True), dim=1))
        
        eq_div = torch.mean(torch.sum((du[:,:,0:1] + dv[:,:,1:]) * phi, dim=2, keepdim=True), dim=1)

        return eq_u, eq_v, eq_div