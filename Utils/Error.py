# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 20:35:43 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 20:35:43 
#  */
import numpy as np 
import torch
import sys

class Error():

    def __init__(self):
        pass 

    def L2_error(self, model:torch.nn.Module, x:torch.tensor, u:torch.tensor)->torch.tensor:
        '''
        Input: 
            model: network model
            x: size(?,d)
            u: size(?,k)
        Output: 
            err:    size(1,k)
            u_pred: size(?,k)
        '''
        with torch.no_grad():
            u_pred = model(x)
            error = torch.mean( (u_pred - u)**2, dim=0) \
                / (torch.mean(u**2, dim=0) + sys.float_info.epsilon)
        
        return torch.sqrt(error), u_pred

    def L2_error_Time(self, model:torch.nn.Module, x:torch.tensor, t:torch.tensor, 
                      u:torch.tensor)->torch.tensor:
        '''
        Input: 
            model: network model
            x: size(?,d)
            t: size(?,1)
            u: size(?,k)
        Output: 
            err:    size(1,k)
            u_pred: size(?,k)
        '''
        with torch.no_grad():
            u_pred = model(x, t)
            error = torch.mean( (u_pred - u)**2, dim=0) \
                / (torch.mean(u**2, dim=0) + sys.float_info.epsilon)
        
        return torch.sqrt(error), u_pred

