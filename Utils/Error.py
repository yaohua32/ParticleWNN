# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:07:13 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:07:13 
#  */
import numpy as np 
import torch
import sys

class Error():

    def __init__(self):
        pass 

    def L2_error(self, u_pred:torch.tensor, u:torch.tensor)->torch.tensor:
        '''
        Input: 
                u_pred: size(?,1)
                u: size(?,1)
        Output: err: float
        '''
        # Error 
        err = torch.mean( (u_pred - u)**2 ) \
            / (torch.mean(u**2) + sys.float_info.epsilon)
        
        return torch.sqrt(err).item()