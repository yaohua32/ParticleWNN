# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 15:30:17 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 15:30:17 
#  */
import numpy as np 
import torch 
from torch.autograd import grad
#
from Utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from Utils.GenData import GenData

class Problem():

    def __init__(self, dtype:np.dtype=np.float64,
                 testFun_type:str='Wendland', **args):
        '''
        The definition of the Problem
        '''
        self._name = 'Problem_Module'
        self._dtype = dtype
        self._testFun_type = testFun_type
    
    @ property
    def name(self):
        return self._name

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def lb(self):
        raise NotImplementedError

    @property
    def ub(self):
        raise NotImplementedError


    def _grad_u(self, x:torch.tensor, u:torch.tensor)->torch.tensor:
        '''
        Input:  
            x: size(?,d)
            u: size(?,1)
        Output: 
            du: size(?,d)
        '''
        du = grad(inputs=x, outputs=u, grad_outputs=torch.ones_like(u), 
                  create_graph=True)[0]
        return du

    def _Laplace_u(self, x_list:list[torch.tensor], du:torch.tensor)->torch.tensor:
        '''
        Input: 
            x_list: [ size(?,1) ]*d
            du: size(?,d)
        Output: 
            Lu: size(?,1)
        '''
        assert len(x_list)==du.shape[1]
        #
        Lu = torch.zeros_like(du[:,0:1])
        for d in range(len(x_list)):
            Lu += grad(inputs=x_list[d], outputs=du[:,d:d+1], 
                       grad_outputs=torch.ones_like(du[:,d:d+1]), 
                       create_graph=True)[0]
        return Lu
    
    def get_observe(self, Nx_in:int=None, Nx_bd_each_side:int=None)->torch.tensor:
        '''
        Get the observation
            Nx_in: size of inside sensors
            Nx_bd_each_side: size of sensors on each boundary
                or 
               None (means loading from saved file.)
        Output:
            data_observe: dict={}
        '''
        raise NotImplementedError
    
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
        raise NotImplementedError
    
    def fun_bd(self, model:torch.nn.Module, x_list:list)->torch.tensor:
        '''
        The boundary conditions
        Input: 
            model:
            x_list: list= [size(n,1)]*2d    
        Output:  
            cond: size(n*2d*?,1)
        '''
        raise NotImplementedError

    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        Input: 
            x: size(?,d)
        Output: 
            f: size(?,1)
        '''
        raise NotImplementedError

    #####################################################################  Test Functions
    def get_testFun_particlewnn(self, GenData:GenData,
                                TestFun:TestFun_ParticleWNN,
                                Nx_mesh:int, int_method:str='mesh'):
        '''
        Input:
            GenData:
            TestFun:
            Nx_mesh: the number of integration points or meshsize (if the int_method is 'mesh')
            int_method: 'mesh'
        Output:
            x_mesh: size(?,d)
            phi: size(?,1)
            dphi_scaled: size(?,d)
        '''
        x_mesh, _ = GenData.get_x_mesh(Nx_mesh=Nx_mesh, method=int_method)
        #
        phi, dphi_scaled, _ = TestFun.get_value(x_mesh=x_mesh)

        return x_mesh, phi, dphi_scaled
    
    ############################################################ Collocation Points
    def get_point_pinn(self, GenData:GenData, N_xin: int, 
                       N_xbd_each_face: int=None) -> dict:
        '''
        Input:
            GenData:
            N_xin: the number of points in the domain
            N_xbd_each_face: the number of points on the boundary (each face)
        Output:
            data_point: {'x_in', 'x_bd_list'}
        '''
        data_point = {}
        #
        x_in = GenData.get_in(Nx_size=N_xin)
        data_point['x_in'] = x_in 
        #
        if N_xbd_each_face is not None:
            x_bd_list = GenData.get_bd(N_bd_each_face=N_xbd_each_face)
            data_point['x_bd_list'] = x_bd_list

        return data_point

    def get_point_particlewnn(self, GenData:GenData, N_xin: int, 
                              N_xbd_each_face: int=None, **args) -> dict:
        '''
        Input:
            GenData:
            N_xin: the number of points in the domain
            N_xbd_each_face: the number of points on the boundary (each face)

            args: 'Rmax', 'Rmin'
        Output:
            data_point: {'xc', 'R', 'x_bd_list'}
        '''
        data_point = {}
        #
        xc, R = GenData.get_particle(N_xc=N_xin, 
                                     R_max=args['Rmax'],
                                     R_min=args['Rmin'])
        data_point['xc'] = xc 
        data_point['R'] = R
        #
        if N_xbd_each_face is not None:
            x_bd_list = GenData.get_bd(N_bd_each_face=N_xbd_each_face)
            data_point['x_bd_list'] = x_bd_list

        return data_point
    
    ###################################################################  
    def strong_pinn(self, model:torch.nn.Module, x:torch.tensor)->torch.tensor:
        '''
        The strong residual of PINN
        Input: 
            model:
            x:size(?,d)
        Output: 
            The strong residual: size(?,1)
        '''
        raise NotImplementedError
    
    def weak_particlewnn(self, model:torch.nn.Module, xc:torch.tensor, R:torch.tensor, 
                         x_mesh:torch.tensor, phi:torch.tensor, 
                         dphi_scaled:torch.tensor)->torch.tensor:
        '''
        The weak residual of ParticleWNN 
            (x = x_mesh * R + xc)
        Input:     
            model: the network model
            xc: particles (The centers of test functions)
            R: radius (The radius of compact support regions)
            x_mesh: size(m, d) 
            phi: size(m, 1)
            dphi_scaled: size(m, d)
        Output: 
            The weak residual: size(?, 1)
        '''
        raise NotImplementedError