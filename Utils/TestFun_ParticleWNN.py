# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 15:41:31 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 15:41:31 
#  */
import numpy as np 
import torch 
import math
#
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

class TestFun_ParticleWNN():
    '''
    '''
    def __init__(self, type:str='Cosin', dim:int=1):
        '''
        Input:
            type: the test function type
            dim: the dimension of the problem
        '''
        self._type = type
        self._dim = dim
        self._eps = sys.float_info.epsilon

    def _dist(self, x:torch.tensor)->torch.tensor:
        '''
        Input:
          x: (?,d) or (?, m, d)
        Output:
          y: the norm of x
        '''
        return torch.linalg.norm(x, dim=1, keepdims=True)

    def _grad(self, x:torch.tensor, y:torch.tensor)->torch.tensor:
        '''
        Input:
            x: the variables
            y: the function values
        Output:
            dy: the grad dy/dx
        '''
        dy = torch.autograd.grad(inputs=x, outputs=y, 
                                 grad_outputs=torch.ones_like(y), 
                                 create_graph=True)[0]
        
        return dy

    def _Laplace(self, x_list:list[torch.tensor], dy_list:torch.tensor)->torch.tensor:
        '''
        Input:
            x_list: the variables list
            dy_list: the list of gradients dy/dx 
        Output:
            Laplace_y: the sum of d^2y/d^2x 
        '''
        Laplace_y = torch.zeros_like(x_list[0])
        for xi, dyi in zip(x_list, dy_list):
            Laplace_y += torch.autograd.grad(outputs=dyi, inputs=xi, 
                                             grad_outputs=torch.ones_like(dyi), 
                                             create_graph=True)[0]
        
        return Laplace_y 
    
    def _Bump(self, x_mesh:torch.tensor, dim:int=1)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function
            dv: the grad dv/dx_mesh
            Lv: the Laplacian 
        '''
        ############ 
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        r_list = [r]
        for _ in range(3):
            r_list.append(r*r_list[-1])
        #
        v = torch.exp(1. - 1. / (1. - r_list[1] + self._eps))
        ########## 
        dv_dr_divide_by_r = v * (-2.) / ((1.-r_list[1])**2 + self._eps)
        ddv_dr = v * (6.*r_list[3]- 2.) / ((1.-r_list[1])**4 + self._eps)
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
            Lv = ddv_dr 
        else:
            dv =  dv_dr_divide_by_r * x_mesh
            Lv = dv_dr_divide_by_r * (dim-1) + ddv_dr
        
        return v.detach(), dv.detach(), Lv.detach()
    
    def _Wendland(self, x_mesh:torch.tensor, dim:int=1)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function
            dv: the grad dv/dx_mesh
            Lv: the Laplacian
        '''
        ############
        l = math.floor(dim / 2) + 3
        #
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        r_list = [r]
        for _ in range(1):
            r_list.append(r*r_list[-1])
        #
        v = (1-r) ** (l+2) * ( (l**2+4.*l+3.) * r_list[1] + (3.*l+6.) * r + 3.) / 3.
        #
        dv_dr_divide_by_r = (1-r)**(l+1) * (- (l**3+8.*l**2+19.*l+12) * r - (l**2+7.*l+12)) / 3.
        ddv_dr = (1-r)**l * ( (l**4+11.*l**3+43.*l**2+69.*l+36)*r_list[1] - (l**3+7.*l**2+12*l)*r - (l**2+7.*l+12)) / 3.
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
            Lv = ddv_dr 
        else:
            dv =  dv_dr_divide_by_r * x_mesh
            Lv = dv_dr_divide_by_r * (dim-1) + ddv_dr
        
        return v.detach(), dv.detach(), Lv.detach()

    def _Cosin(self, x_mesh:torch.tensor, dim:int=1)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function values
            dv: the grad dv/dx_mesh
            Lv: the Laplacian of d^2v/d^2x_mesh
        '''
        ############
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        v = (1. - torch.cos(torch.pi * (r + 1.))) / torch.pi
        #
        dv_dr_divide_by_r = torch.sin(torch.pi * (r+1.)) / (r + self._eps)
        ddv_dr = torch.pi * torch.cos(torch.pi * (r+1.))
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
            Lv = torch.where(r<1., ddv_dr, torch.zeros_like(r))
        else:
            dv = dv_dr_divide_by_r * x_mesh 
            Lv = torch.where(r<1., dv_dr_divide_by_r * (dim-1) + ddv_dr, torch.zeros_like(r))
        
        return v.detach(), dv.detach(), Lv.detach()

    def _Wend_powerK(self, x_mesh:torch.tensor, dim:int=1, k:int=4)->torch.tensor:
        '''
        Input:
            x_mesh: (?, d) or (?, m, d)
        Output:
            v: the test function
            dv: the grad dv/dx_mesh
            Lv: the Laplacian
        '''
        ############
        l = math.floor(dim / 2) + 3
        #
        r = 1. - torch.relu(1. - self._dist(x_mesh))
        r_list = [r]
        for _ in range(1):
            r_list.append(r*r_list[-1])
        #
        v_wend = (1-r) ** (l+2) * ( (l**2+4.*l+3.) * r_list[1] + (3.*l+6.) * r + 3.) / 3.
        dv_dr_divide_by_r_wend = (1-r)**(l+1) * (- (l**3+8.*l**2+19.*l+12) * r - (l**2+7.*l+12)) / 3.
        ddv_dr_wend = (1-r)**l * ( (l**4+11.*l**3+43.*l**2+69.*l+36)*r_list[1] - (l**3+7.*l**2+12*l)*r - (l**2+7.*l+12)) / 3.
        #
        v = v_wend ** k
        dv_dr_divide_by_r = k * v_wend**(k-1) * dv_dr_divide_by_r_wend
        ddv_dr = k*(k-1)* v_wend**(k-2) * dv_dr_divide_by_r_wend**2 * r_list[1] + k * v_wend**(k-1) * ddv_dr_wend
        #
        if dim==1:
            dv = dv_dr_divide_by_r * r * torch.sign(x_mesh)
            Lv = ddv_dr 
        else:
            dv =  dv_dr_divide_by_r * x_mesh
            Lv = dv_dr_divide_by_r * (dim-1) + ddv_dr
        
        return v.detach(), dv.detach(), Lv.detach()
    
    def get_value(self, x_mesh:torch.tensor)->torch.tensor:
        '''
        Input:
            x_mesh: size(?,d)
        Output:
            phi, dphi_scaled, Lphi_scaled
        '''
        if self._type=='Cosin':
            return self._Cosin(x_mesh=x_mesh, dim=self._dim)
        elif self._type=='Bump':
            return self._Bump(x_mesh=x_mesh, dim=self._dim)
        elif self._type=='Wendland':
            return self._Wendland(x_mesh=x_mesh, dim=self._dim)
        elif self._type=='Wendland_k':
            return self._Wend_powerK(x_mesh=x_mesh, dim=self._dim)
        else:
            raise NotImplementedError