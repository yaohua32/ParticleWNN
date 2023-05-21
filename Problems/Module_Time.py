# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:05:08 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:05:08 
#  */
import numpy as np 
import torch 
from torch.autograd import grad

class Problem():
    '''
    '''
    def __init__(self, test_type:str=None):
        #
        self._t0 = 0.
        self._tT = 1.
        self._lb = np.array([-1.])
        self._ub = np.array([1.])
    
    @ property
    def name(self):
        '''
        The problem name
        '''
        raise NotImplementedError

    @property
    def dim(self):
        '''
        The dimension of spatial-space
        '''
        raise NotImplementedError
    
    @property 
    def tRely(self):
        '''
        The problem relies on time or not (False/True).
        '''
        return True

    def _grad_u(self, x_or_t:torch.tensor, u_pred:torch.tensor)->torch.tensor:
        '''
        Input: x_or_t:size(?,d)
               u_pred:size(?,1)
        Output: du: size(?,d)
        '''
        du = grad(inputs=x_or_t, outputs=u_pred, 
                  grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        return du

    def _Laplace_u(self, x_list:list[torch.tensor], du_pred:torch.tensor)->torch.tensor:
        '''
        Input: x_list: [ size(?,1) ]*d
               du_pred: size(?,d)
        Output: Lu: size(?,1)
        '''
        assert len(x_list)==du_pred.shape[1]
        #
        Lu = torch.zeros_like(du_pred[:,0:1])
        for d in range(len(x_list)):
            Lu += grad(inputs=x_list[d], outputs=du_pred[:,d:d+1], 
                       grad_outputs=torch.ones_like(du_pred[:,d:d+1]), create_graph=True)[0]
        return Lu

    def fun_u_init(self, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        Input: x: size(?,d)
               t: size(?,1)
        Output: u: size(?,1)
        '''
        raise NotImplementedError

    def fun_u_bd(self, x_list:list, t:torch.tensor, model_u=None)->torch.tensor:
        '''
        Input:   x_list: list= [size(n,1)]*2d
                 t: size(n,1)
                 model:       
        Output:  u_lb_pred: size(n*d,1)
                 u_ub_pred: size(n*d,1)
        '''
        raise NotImplementedError

    def fun_f(self, x:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        The right hand side f
        Input: x: size(?,d)
               t: size(?,1)
        Output: f: size(?,1)
        '''
        raise NotImplementedError
    
    def strong(self, model_u, xs:torch.tensor, t:torch.tensor)->torch.tensor:
        '''
        The strong form:
        Input: model_u
                    x:size(?,d)
                    t:size(?,1)
        Output: pde: size(?,1)
        '''
        raise NotImplementedError
    
    def weak(self, model_u, x_scaled:torch.tensor, xc:torch.tensor, t:torch.tensor,
              R:torch.tensor)->torch.tensor:
        '''
        The weak form
        Input:     model: the network model
                x_scaled: size(m, d)
                  xc: size(?, 1, d)
                   t: size(?, 1, 1)
                   R: size(?, 1, d)
        Output: weak_form: size(?, 1)
        '''
        raise NotImplementedError