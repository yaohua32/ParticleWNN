# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-28 17:28:18 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-28 17:28:18 
#  */
import numpy as np 
import torch 
from torch.autograd import grad
#
from Utils.TestFun_ParticleWNN import TestFun_ParticleWNN
from Utils.GenData_Time import GenData

class Problem():

    def __init__(self, dtype:np.dtype=np.float64,
                 testFun_type:str='Wendland', **args):
        '''
        The definition of the Problem
        '''
        self._name = 'Problem_Module_Time'
        self._dtype = dtype
        self._testFun_type = testFun_type
    
    @ property
    def name(self):
        return self._name

    @property
    def dim(self):
        raise NotImplementedError
    
    @property 
    def t_mesh(self):
        raise NotImplementedError

    @property
    def lb(self):
        raise NotImplementedError

    @property
    def ub(self):
        raise NotImplementedError

    def _grad_u(self, x_or_t:torch.tensor, u:torch.tensor)->torch.tensor:
        '''
        Input:  
            x or t: size(?,d) or size(?,1)
            u: size(?,1)
        Output: 
            du/dx or du/dt: size(?,d) or size(?,1)
        '''
        du = grad(inputs=x_or_t, outputs=u, grad_outputs=torch.ones_like(u), 
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
    
    def get_observe(self, Nt_in:int, Nx_in:int, **args)->dict:
        '''
        Input: 
            Nt_in: the mesh_size of the t-axis
            Nx_in: the mesh_size of the x-axis
        Output: 
            data_observe: dict={} 
            args: dict={}
        '''
        raise NotImplementedError
    
    def get_test(self, t_start_loc:int, t_end_loc:int)->torch.tensor:
        '''
        Input:
            t_start_loc: the location of the start t
            t_end_loc: the location of the end t
        Output:
            u: size(?,1)
            x: size(?,d)
            t: size(?,1)
        '''
        raise NotImplementedError
    
    def fun_bd(self, model:torch.nn.Module, x_list:list[torch.tensor], 
               t:torch.tensor)->torch.tensor:
        '''
        The boundary conditions
        Input:  
            model:
            x_list: list= [size(n,1)]*2d
            t: size(n,1)
            model:       
        Output:  
            cond_bd: size(?,1)
        '''
        raise NotImplementedError
    
    def fun_init(self, model:torch.nn.Module, x:torch.tensor, 
                 t:torch.tensor)->torch.tensor:
        '''
        The initial conditions
        Input:
            model:
            x: size(?,d)
            t: size(?,1)
        Output:
            cond_init: size(?,1)
        '''
        raise NotImplementedError

    def fun_f(self, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        Input: 
            x: size(?,d)
            t: size(?,1)
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
        x_mesh = GenData.get_x_mesh(Nx_mesh=Nx_mesh, method=int_method)
        #
        phi, dphi_scaled, _ = TestFun.get_value(x_mesh=x_mesh)

        return x_mesh, phi, dphi_scaled

    ############################################################ Collocation Points
    def get_point_pinn(self, GenData:GenData, 
                       Nt_in:int, Nx_in:int, 
                       Nt_bd:int, Nx_bd_each_face:int,
                       Nx_init:int, **args)->list[dict]:
        '''
        Input:
            GenData:
            Nt_in: the number of time slices (for points in the domain)
            Nx_in: the number of points inside the domain (each time slice)
            Nt_bd: the number of time slices (for points on the boundary)
            Nx_bd_each_face: the number of points on the boundary (each face and each time slice)
            Nx_init: the number of points for initial condition
            args:     
                tloc_list: [(tloc_0, tloc_1), (tloc_1, tloc_2), ......]
        Output:
            data_point: {'x_init', 't_init', 'x_bd_list', 't_bd', 
                        'x_in_list', 't_in_list'}
        '''
        ################### points for initial condition
        x_init, t_init = GenData.get_init(t0=self.t_mesh[0], Nx_init=Nx_init)
        ################### points for bounday condition 
        x_bd_list, t_bd = GenData.get_bd(t0=self.t_mesh[0], tT=self.t_mesh[-1], 
                                         Nt_size= Nt_bd, Nx_bd_each_face=Nx_bd_each_face)
        ################### Points for PDE residual
        x_in, t_in = GenData.get_in(
            t0=self.t_mesh[0], tT=self.t_mesh[-1], Nt_size=Nt_in, Nx_size=Nx_in)
        x_in_list, t_in_list = [x_in], [t_in]
        #
        data_point = {'x_init':x_init, 't_init':t_init, 
                      'x_bd_list':x_bd_list, 't_bd':t_bd,
                      'x_in_list':x_in_list, 't_in_list':t_in_list}
        
        return data_point
    
    def get_point_particlewnn(self, GenData:GenData,
                              Nt_in:int, Nx_in:int,
                              Nt_bd:int, Nx_bd_each_face:int, 
                              Nx_init:int, **args)->dict:
        '''
        Input:
            GenData:
            Nt_in: the number of time slices (for points in the domain)
            Nx_in: the number of points inside the domain (each time slice)
            Nt_bd: the number of time slices (for points on the boundary)
            Nx_bd_each_face: the number of points on the boundary (each face and each time slice)
            Nx_init: the number of points for initial condition
            args:
                Rmax, Rmin:           
                tloc_list: [(tloc_0, tloc_1), (tloc_1, tloc_2), ......]
        Output:
            data_point: {'x_init', 't_init', 'x_bd_list', 't_bd', 
                        'xc_list', 'tc_list', 'R_list'}
        '''
        ################### points for initial condition
        x_init, t_init = GenData.get_init(t0=self.t_mesh[0], Nx_init=Nx_init)
        ################### points for bounday condition 
        x_bd_list, t_bd = GenData.get_bd(t0=self.t_mesh[0], tT=self.t_mesh[-1], 
                                         Nt_size= Nt_bd, Nx_bd_each_face=Nx_bd_each_face)
        ################### particles for PDE residual
        xc, tc, R= GenData.get_particle(
            t0=self.t_mesh[0], tT=self.t_mesh[-1], N_tc=Nt_in, N_xc=Nx_in, 
            R_max=args['Rmax'], R_min=args['Rmin'])
        xc_list, tc_list, R_list = [xc], [tc], [R]
        ###############
        data_point = {'x_init':x_init, 't_init':t_init, 
                      'x_bd_list':x_bd_list, 't_bd':t_bd,
                      'xc_list': xc_list, 'tc_list': tc_list, 'R_list': R_list}
     
        return data_point
    
    def strong_pinn(self, model:torch.nn.Module, x:torch.tensor, 
                    t:torch.tensor)->torch.tensor:
        '''
        The strong residual of PINN
        Input: 
            model:
            x:size(?,d)
            t:size(?,1)
        Output: 
            The strong residual: size(?,1)
        '''
        raise NotImplementedError
    
    def weak_particlewnn(self, model:torch.nn.Module, xc:torch.tensor, 
                         tc:torch.tensor, R:torch.tensor, 
                         x_mesh:torch.tensor, phi:torch.tensor, 
                         dphi_scaled:torch.tensor)->torch.tensor:
        '''
        The weak residual of ParticleWNN 
            (x = x_mesh * R + xc)
        Input:     
            model: the network model
            xc: size(?, 1, 1) particles (The centers of test functions)
            tc: size(?, 1, 1) timestamps
            R: size(?, 1, 1) radius (The radius of compact support regions)
            x_mesh: size(m, d) 
            phi: size(m, 1)
            dphi_scaled: size(m, d)
        Output: 
            The weak residual: size(?, 1)
        '''
        raise NotImplementedError