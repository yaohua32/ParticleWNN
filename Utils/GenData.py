# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 11:36:56 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 11:36:56 
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
        self.lhs_x = qmc.LatinHypercube(d)

    def get_in(self, Nx_size:int, method:str='hypercube')->torch.tensor:
        '''
        Input:
            Nx_size: meshsize for x-axis
            method:
        Return:
             x
        '''
        if method=='mesh':
            if self.d==1:
                x_scaled = np.linspace(-1., 1., Nx_size).reshape(-1, self.d)
            elif self.d==2:
                xx, yy = np.meshgrid(np.linspace(-1., 1., Nx_size), np.linspace(-1., 1., Nx_size))
                x_scaled = np.concatenate([xx.reshape(-1,1), yy.reshape(-1,1)], axis=1)
            elif self.d==5:
                x_mesh = np.linspace(-1, 1, Nx_size)
                mesh_list = np.meshgrid(x_mesh, x_mesh, x_mesh, x_mesh, x_mesh)
                mesh_list = [mesh.reshape(-1,1) for mesh in mesh_list]
                x_scaled = np.concatenate(mesh_list, axis=1)
            elif self.d==10:
                x_mesh = np.linspace(-1, 1, Nx_size)
                mesh_list = np.meshgrid(x_mesh, x_mesh, x_mesh, x_mesh, x_mesh,
                                        x_mesh, x_mesh, x_mesh, x_mesh, x_mesh)
                mesh_list = [mesh.reshape(-1,1) for mesh in mesh_list]
                x_scaled = np.concatenate(mesh_list, axis=1)
            else:
                raise NotImplementedError(f'Not availabel for d={self.d}.')
            x = (self.x_ub-self.x_lb)*x_scaled/2 + (self.x_ub + self.x_lb)/2
        elif method=='hypercube':
            x = qmc.scale(self.lhs_x.random(Nx_size), self.x_lb, self.x_ub)
        else:
            raise NotImplementedError
        
        return torch.tensor(x.astype(self.dtype))
    
    def get_bd(self, N_bd_each_face:int)->list[torch.tensor]:
        '''
        Input:
            N_bd_each_face: mesh-size in the x-axis
        Return:
             x_list: x_list has the form [lb_d1, ub_d1, lb_d2, ub_d2, ......]
        '''
        x_list = []
        # [N_bd_each_face ] * 2 * d
        x = qmc.scale(self.lhs_x.random(N_bd_each_face), self.x_lb, self.x_ub)
        for d in range(self.d):
            x_lb, x_ub= np.copy(x), np.copy(x)
            x_lb[:,d:d+1], x_ub[:,d:d+1] = self.x_lb[d], self.x_ub[d]
            x_list.extend([torch.from_numpy(x_lb.astype(self.dtype)), 
                            torch.from_numpy(x_ub.astype(self.dtype))])

        return x_list
    
    def get_particle(self, N_xc:int, R_max:float=1e-4, R_min:float=1e-4,
                     R_first=True)->torch.tensor:
        '''
        Input: 
               N_xc: The number of particles (each time slice)
               R_max: The maximum of Radius 
               R_min: The minimum of Radius
               R_first:  True or False
        Output: 
                xc: size(?, 1, d)
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
            R = np.random.uniform(R_min, R_max, [N_xc, 1])
            lb, ub = self.x_lb + R, self.x_ub - R 
            # 
            xc = self.lhs_x.random(N_xc) * (ub - lb) + lb 
        else:
            xc = qmc.scale(self.lhs_x.random(N_xc), self.x_lb+R_max, self.x_ub-R_max)
            #
            R = R_max * np.ones([N_xc, 1])
        
        return torch.tensor(xc.astype(self.dtype)).view(-1,1,self.d),\
            torch.tensor(R.astype(self.dtype)).view(-1, 1, 1)
            
    
    def get_x_mesh(self, Nx_mesh:int, method:str='mesh')->torch.tensor:
        '''
        Input: 
            N_xmesh: the number of meshgrids or meshsize
            method: 'mesh' or 'gaussian' or 'random' or 'hypercube'
        Output: 
            x_mesh: size(?,d)
            w_weight: 
        '''
        if method=='random':
            if self.d==1:
                x_scaled = np.random.uniform(-1., 1., size=(Nx_mesh, self.d))
            else:
                X_d = np.random.normal(size=(Nx_mesh, self.d+2))               
                X_d = X_d / np.sqrt(np.sum(X_d**2, axis=1, keepdims=True))
                #                        
                x_scaled = X_d[:,0:self.d].reshape([-1, self.d])
        elif method=='hypercube':
            if self.d==1:
                x_scaled = qmc.scale(self.lhs_x.random(Nx_mesh), -1., 1.)
            else:
                X_d = qmc.scale(self.lhs_x.random(Nx_mesh), -1., 1.)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) < 1.)[0]
                x_scaled = X_d[index,:]
        elif method=='mesh':
            if self.d==1:
                x_scaled = np.linspace(-1., 1., Nx_mesh).reshape(-1, self.d)
            elif self.d==2:
                x, y = np.meshgrid(np.linspace(-1., 1., Nx_mesh), np.linspace(-1., 1., Nx_mesh))
                X_d = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) <1.)[0]
                x_scaled = X_d[index,:]
            elif self.d==5:
                x = np.linspace(-1, 1, Nx_mesh)
                mesh_list = np.meshgrid(x,x,x,x,x)
                mesh_list = [mesh.reshape(-1,1) for mesh in mesh_list]
                X_d = np.concatenate(mesh_list, axis=1)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) <1.)[0]
                x_scaled = X_d[index,:]
            elif self.d==10:
                x = np.linspace(-1, 1, Nx_mesh)
                mesh_list = np.meshgrid(x,x,x,x,x,x,x,x,x,x)
                mesh_list = [mesh.reshape(-1,1) for mesh in mesh_list]
                X_d = np.concatenate(mesh_list, axis=1)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) <1.)[0]
                x_scaled = X_d[index,:]   
            else:
                raise NotImplementedError(f'The mesh method is not availabel for d={self.d}.')
        elif method=='gaussian':
            if self.d==1:
                x, w = np.polynomial.legendre.leggauss(Nx_mesh)
                x_scaled = x.reshape(-1,1)
                w_scaled = w.reshape(-1,1)
            else:
                raise NotImplementedError(f'The gaussian method is not availabel for d={self.d}.')
        else:
            raise NotImplementedError
        #
        try:
            return torch.tensor(x_scaled.astype(self.dtype)), \
                torch.tensor(w_scaled.astype(self.dtype))
        except:
            return torch.tensor(x_scaled.astype(self.dtype)), None