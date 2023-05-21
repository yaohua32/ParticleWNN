# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:07:20 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:07:20 
#  */
import numpy as np 
import torch 
import os 
import sys 
from scipy.stats import qmc
#
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
#
from Problems.Module_Spatial import Problem

class GenData():
    '''
    '''
    def __init__(self, problem:Problem, dtype:np.dtype=np.float64):
        self.problem = problem
        self.dtype = dtype
        self.lhs_x = qmc.LatinHypercube(self.problem.dim)

    def get_in(self, Nx_size:int)->torch.tensor:
        '''
        Input:
             Nx_size
        Output:
             x
        '''
        x = qmc.scale(self.lhs_x.random(Nx_size), self.problem._lb, self.problem._ub)
        
        return torch.tensor(x.astype(self.dtype))
    
    def get_bd(self, N_bd_each_face:int)->torch.tensor:
        '''
        Input:
            N_bd_each_face
        Output:
            x_list: if the problem is time-dependent
            where x_list takes the form [lb_d1, ub_d1, lb_d2, ub_d2, ......]
        '''
        x_list = []
        # ([N_bd_each_face ] * 2 * d)
        x = qmc.scale(self.lhs_x.random(N_bd_each_face), self.problem._lb, self.problem._ub)
        for d in range(self.problem.dim):
            x_lb, x_ub= np.copy(x), np.copy(x)
            x_lb[:,d:d+1], x_ub[:,d:d+1] = self.problem._lb[d], self.problem._ub[d]
            x_list.extend([torch.from_numpy(x_lb.astype(self.dtype)), 
                            torch.from_numpy(x_ub.astype(self.dtype))])

        return x_list
    
    def get_xc(self, N_xc:int, R_max:float=1e-3, R_min:float=1e-8, 
               R_method:str='R_first')->torch.tensor:
        '''
        Input: N_xc: particles
               R_max: 
               R_min: 
               R_method: 
        Output: R, xc
        '''
        if R_max<R_min:
            raise ValueError('R_max should be large than R_min.')
        elif (2.*R_max)>np.min(self.problem._ub - self.problem._lb):
            raise ValueError('R_max is too large.')
        elif (R_max)<(1e-7+1e-8) and self.dtype is np.float32:
            raise ValueError('R_max is too small.')
        elif (R_max)<(1e-15+1e-16) and self.dtype is np.float64:
            raise ValueError('R_max is too small.')
        #  N_xc
        if R_method=='R_first':
            R = np.random.uniform(R_min, R_max, [N_xc, 1])
            lb, ub = self.problem._lb + R, self.problem._ub - R
            # 
            xc = self.lhs_x.random(N_xc) * (ub - lb) + lb 
        elif R_method=='xc_first':
            xc = qmc.scale(self.lhs_x.random(N_xc), self.problem._lb+R_max, self.problem._ub-R_max)
            #
            R = R_max * np.ones([N_xc, 1])
        else:
            raise NotImplementedError
        
        return torch.tensor(R.astype(self.dtype)).view(-1, 1, 1),\
            torch.tensor(xc.astype(self.dtype)).view(-1,1,self.problem.dim)
    
    def get_x_scaled(self, Nx_scaled, method:str='mesh')->torch.tensor:
        '''
        Input: Nx_scaled:
               method: 'mesh'
        Output: x_scaled
        '''
        if method=='mesh':
            if self.problem.dim==1:
                x_scaled = np.linspace(-1., 1., Nx_scaled).reshape(-1, self.problem.dim)
            elif self.problem.dim==2:
                x, y = np.meshgrid(np.linspace(-1., 1., Nx_scaled), np.linspace(-1., 1., Nx_scaled))
                X_d = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
                #
                index = np.where(np.linalg.norm(X_d, axis=1, keepdims=True) <1.)[0]
                x_scaled = X_d[index,:]
            else:
                raise NotImplementedError('The mesh method is not availabel for d>3.')
        elif method=='gaussian':
            if self.problem.dim==1:
                x, w = np.polynomial.legendre.leggauss(Nx_scaled)
                x_scaled = x.reshape(-1,1)
                w_scaled = w.reshape(-1,1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        #
        try:
            return torch.tensor(x_scaled.astype(self.dtype)), \
                torch.tensor(w_scaled.astype(self.dtype))
        except:
            return torch.tensor(x_scaled.astype(self.dtype))