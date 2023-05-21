# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:04:30 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:04:30 
#  */
import numpy as np 
import torch 
from torch.autograd import grad

class Problem():
    '''
    '''
    def __init__(self, test_type:str=None):
        #
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
    
    @ property
    def name(self):
        '''
        The problem name.
        '''
        raise NotImplementedError

    @property
    def dim(self):
        '''
        The dimension of spatial-space.
        '''
        raise NotImplementedError
    
    @property 
    def tRely(self):
        '''
        The problem relies on time or not (False/True).
        '''
        return False

    def _grad_u(self, x:torch.tensor, u_pred:torch.tensor)->torch.tensor:
        '''
        '''
        du = grad(inputs=x, outputs=u_pred, 
                  grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        return du

    def _Laplace_u(self, x_list:list[torch.tensor], du_pred:torch.tensor)->torch.tensor:
        '''
        '''
        assert len(x_list)==du_pred.shape[1]
        #
        Lu = torch.zeros_like(du_pred[:,0:1])
        for d in range(len(x_list)):
            Lu += grad(inputs=x_list[d], outputs=du_pred[:,d:d+1], 
                       grad_outputs=torch.ones_like(du_pred[:,d:d+1]), create_graph=True)[0]
        return Lu

    def fun_u_bd(self, x_list:list, model_u=None)->torch.tensor:
        '''
        '''
        raise NotImplementedError

    def fun_f(self, x:torch.tensor)->torch.tensor:
        '''
        The right hand side f
        '''
        raise NotImplementedError
    
    def strong(self, model_u, xs:torch.tensor)->torch.tensor:
        '''
        The strong form
        '''
        raise NotImplementedError
    
    def weak(self, model_u, x_scaled:torch.tensor, xc:torch.tensor,
             R:torch.tensor, stretch:float=1.)->torch.tensor:
        '''
        The weak form (wehre x_in = x_scaled * R + xc)
        Input:     model: the network model
                x_scaled: size(m, d)
                  xc: size(?, 1, d)
                   R: size(?, 1, d)
        Output: weak_form: size(?, 1)
        '''
        raise NotImplementedError