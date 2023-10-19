# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 11:36:42 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 11:36:42 
#  */
import numpy as np 
import torch 
from scipy.stats import qmc

class GenData():

    def __init__(self, d:int, x_lb:np.array, x_ub:np.array, 
                 dtype:np.dtype=np.float64):
        '''
        Generate collocation points:
            d: the dim of x-axis
            x_lb: the lower bound of x
            x_ub: the upper bound of x
            dtype: the datatype
        '''
        self.d = d
        self.x_lb = x_lb 
        self.x_ub = x_ub
        self.dtype = dtype
        self.lhs_t = qmc.LatinHypercube(1)
        self.lhs_x = qmc.LatinHypercube(d)

    def get_in(self, t0:float, tT:float, Nt_size:int, Nx_size:int,
               t_method:str='hypercube'):
        '''
        Input:
            t0, tT :
            Nx_size: meshsize for x-axis
            Nt_size: meshsize for t-axis
            t_method: 
        Return:
             x: size(Nx_size * Nt_size, d)
             t: size(Nx_size * Nt_size, 1)
        '''
        if t_method=='mesh':
            t = np.linspace(t0, tT, Nt_size).reshape(-1,1)  
        elif t_method=='random':
            t = np.random.uniform(t0, tT, [Nt_size,1])  
        elif t_method=='hypercube':
            t = qmc.scale(self.lhs_t.random(Nt_size), t0, tT)
        else:
            raise NotImplementedError
        #（size = Nx_size * Nt_size)
        x = qmc.scale(self.lhs_x.random(Nx_size*Nt_size), self.x_lb, self.x_ub)
        t = t.repeat(Nx_size, axis=0)
        
        return torch.tensor(x.astype(self.dtype)), torch.tensor(t.astype(self.dtype))
    
    def get_bd(self, t0:float, tT:float, Nt_size:int, Nx_bd_each_face:int, 
               t_method='hypercube'):
        '''
        Input:
                    t0, tT:
            Nx_bd_each_face: mesh-size in the x-axis
                   Nt_size: mesh-size in the t-axis
                   t_methd: 'mesh' or 'hypercube'
        Return:
             x_list : [ size(N_bd_each_face * Nt_size, 1) ] * 2d
                      where x_list has the form [lb_d1, ub_d1, lb_d2, ub_d2, ......]
             t: size(N_bd_each_face * Nt_size, 1)
        '''
        x_list = []
        if t_method=='mesh':
            t = np.linspace(t0, tT, Nt_size).reshape(-1,1)
        elif t_method=='random':
            t = np.random.uniform(t0, tT, [Nt_size,1])  
        elif t_method=='hypercube':
            t = qmc.scale(self.lhs_t.random(Nt_size), t0, tT)
        else:
            raise NotImplementedError
        # size = [N_bd_each_face * Nt_size] * 2 * d
        x = qmc.scale(self.lhs_x.random(Nx_bd_each_face * Nt_size), self.x_lb, self.x_ub)
        for d in range(self.d):
            x_lb, x_ub= np.copy(x), np.copy(x)
            x_lb[:,d:d+1], x_ub[:,d:d+1] = self.x_lb[d], self.x_ub[d]
            x_list.extend([torch.from_numpy(x_lb.astype(self.dtype)), 
                           torch.from_numpy(x_ub.astype(self.dtype))])
        t = t.repeat(Nx_bd_each_face, axis=0)

        return x_list, torch.from_numpy(t.astype(self.dtype))

    def get_init(self, t0:float, Nx_init:int):
        '''
        Input:
            t0:
            N_init:
        Output:    
            x:size(?,d)
            t:size(?,1)
        '''
        t = t0 * np.ones([Nx_init, 1])
        x = qmc.scale(self.lhs_x.random(Nx_init), self.x_lb, self.x_ub)

        return torch.from_numpy(x.astype(self.dtype)), torch.from_numpy(t.astype(self.dtype))
    
    def get_particle(self, t0:float, tT:float, N_tc:int, N_xc:int, 
                     R_max:float=1e-4, R_min:float=1e-4,
                     t_method:str='hypercube', R_first=True):
        '''
        Input: 
               t0, tT:
               N_xc: The number of particles (each time slice)
               N_tc: The number of time slices
               R_max: The maximum of Radius 
               R_min: The minimum of Radius
               t_method: 'mesh' or 'hypercube' or ...
               R_first:  True or False
        Output: 
                xc: size(?, 1, d)
                tc: size(?, 1, 1)
                R: size(?, 1, 1)
        '''
        if R_max<R_min:
            raise ValueError('R_max should be greater than R_min.')
        elif (2.*R_max)>np.min(self.x_ub - self.x_lb):
            raise ValueError('R_max is too large.')
        elif (R_min)<1e-4 and self.dtype is np.float32:
            raise ValueError('R_min<1e-4 when data_type is np.float32!')
        elif (R_min)<1e-10 and self.dtype is np.float64:
            raise ValueError('R_min<1e-10 when data_type is np.float64!')
        #
        if R_first:
            R = np.random.uniform(R_min, R_max, [N_xc*N_tc, 1])
            lb, ub = self.x_lb + R, self.x_ub - R 
            #
            if t_method=='hypercube':
                tc = qmc.scale(self.lhs_t.random(N_tc), t0, tT)
            elif t_method=='mesh':
                tc = np.linspace(t0, tT, N_tc).reshape(-1,1)
            elif t_method=='random':
                tc = np.random.uniform(t0, tT, [N_tc,1]) 
            else:
                raise NotImplementedError
            # N_xc * Nt_mesh
            xc = self.lhs_x.random(N_xc*N_tc) * (ub - lb) + lb 
            tc = tc.repeat(N_xc, axis=0)
        else:
            raise NotImplementedError
        
        return torch.tensor(xc.astype(self.dtype)).view(-1,1,self.d),\
               torch.tensor(tc.astype(self.dtype)).view(-1, 1, 1),\
               torch.tensor(R.astype(self.dtype)).view(-1, 1, 1),
                
    def get_x_mesh(self, Nx_mesh:int, method:str='mesh')->torch.tensor:
        '''
        Input: 
            N_xmesh: the number of meshgrids or meshsize
            method: 'random' or 'hypercube' or 'mesh'
        Output: 
            x_mesh: size(?,d)
        '''
        if method=='random':
            if self.d==1:
                x_mesh = np.random.uniform(-1., 1., size=(Nx_mesh, self.d))
            else:
                X_d = np.random.normal(size=(Nx_mesh, self.d+2))               
                X_d = X_d / np.sqrt(np.sum(X_d**2, axis=1, keepdims=True)) 
                #                        
                x_mesh = X_d[:,0:self.d].reshape([-1, self.d])
        elif method=='hypercube':
            if self.d==1:
                x_mesh = qmc.scale(self.lhs_x.random(Nx_mesh), -1., 1.)
            else:
                X_d = qmc.scale(self.lhs_x.random(Nx_mesh), -1., 1.)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) < 1.)[0]
                x_mesh = X_d[index,:]
        elif method=='mesh':
            if self.d==1:
                x_mesh = np.linspace(-1., 1., Nx_mesh).reshape(-1, self.d)
            elif self.d==2:
                x, y = np.meshgrid(np.linspace(-1., 1., Nx_mesh), np.linspace(-1., 1., Nx_mesh))
                X_d = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) <1.)[0]
                x_mesh = X_d[index,:]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
            
        return torch.tensor(x_mesh.astype(self.dtype))

    def get_t_mesh(self, N_tmesh:int, method:str='mesh')->torch.tensor:
        '''
        Input:
            N_tmesh: the number of meshgrids in t-axis
            method: 'mesh' or 'hypercube' or 'random'
        '''
        if method=='random':
            t_mesh = np.random.uniform(-1., 1., size=(N_tmesh, 1))
        elif method=='hypercube':
            t_mesh = qmc.scale(self.lhs_t.random(N_tmesh), -1., 1.)
        elif method=='mesh':
            t_mesh = np.linspace(-1., 1., N_tmesh).reshape(-1, 1)
        else:
            raise NotImplementedError
            
        return torch.tensor(t_mesh.astype(self.dtype))