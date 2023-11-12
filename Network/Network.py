# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 15:40:04 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 15:40:04 
#  */
import numpy as np
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=3, 
                 activation:str='tanh', **kwargs):
        super(FeedForward, self).__init__()
        # Activation
        if activation=='relu':
            self.activation = torch.nn.ReLU()
        elif activation=='elu':
            self.activation = torch.nn.ELU()
        elif activation=='softplus':
            self.activation = torch.nn.Softplus()
        elif activation=='sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation=='tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise NotImplementedError
        # Network Sequential
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def forward(self, x):
        #
        x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        x = self.activation(self.fc_in(x))
        ############################
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x)) + x

        return self.fc_out(x)

class FeedForward_Sin(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=3, 
                 activation:str='tanh', **kwargs):
        super(FeedForward_Sin, self).__init__()
        # Activation
        if activation=='relu':
            self.activation = torch.nn.ReLU()
        elif activation=='elu':
            self.activation = torch.nn.ELU()
        elif activation=='softplus':
            self.activation = torch.nn.Softplus()
        elif activation=='sigmoid':
            self.activation = torch.nn.Sigmoid()
        elif activation=='tanh':
            self.activation = torch.nn.Tanh()
        else:
            raise NotImplementedError
        # Network Sequential
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        self.fc_hidden_list = nn.ModuleList() 
        for _ in range(hidden_layers):
            self.fc_hidden_list.append(nn.Linear(hidden_size, hidden_size))
        #
        try:
            assert kwargs['lb'].shape==(1,d_in)
            self.lb = kwargs['lb']
            self.ub = kwargs['ub']
        except:
            self.lb = - torch.ones(1, d_in)
            self.ub = torch.ones(1, d_in)

    def fun_sin(self, x):
        '''
        '''
        return torch.sin(torch.pi * (x+1.))

    def forward(self, x):
        #
        # x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        x = self.fc_in(x)
        ############################
        for fc_hidden in self.fc_hidden_list:
            x = self.fun_sin(x)
            x = self.activation(fc_hidden(x)) + x

        return self.fc_out(x)
    
class Model():
    '''
    '''
    def __init__(self, model_type:str, device=None, 
                 dtype:torch.dtype=torch.float64):
        self.model_type = model_type
        self.device = device
        torch.set_default_dtype(dtype)
    
    def get_model(self, d_in:int=1, d_out:int=1, 
                  h_size:int=100, l_size:int=3, activation:str='tanh', 
                  **kwargs):
        if self.model_type=='FeedForward':
            return FeedForward(d_in=d_in, d_out=d_out, 
                                hidden_size=h_size, 
                                hidden_layers=l_size,
                                activation=activation,
                                **kwargs).to(self.device)
        elif self.model_type=='FeedForward_Sin':
            return FeedForward_Sin(d_in=d_in, d_out=d_out, 
                                    hidden_size=h_size, 
                                    hidden_layers=l_size,
                                    activation=activation,
                                    **kwargs).to(self.device)
        else:
            raise NotImplementedError