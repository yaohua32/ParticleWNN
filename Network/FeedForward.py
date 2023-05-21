# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:01:30 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:01:30 
#  */
import numpy as np
import torch
import torch.nn as nn

class Network_tanh_sin(nn.Module):

    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh_sin, self).__init__()
        #
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
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
        return torch.sin(np.pi * (x+1.))

    def forward(self, x):
        #
        x = self.fc_in(x)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(self.fun_sin(fc_hidden(x))) + x
        x_out = x

        return self.fc_out(x_out)

class Network_tanh(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_tanh, self).__init__()
        #
        self.activation = torch.nn.Tanh()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
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
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation(fc_hidden(x)) + x
        x_out = x

        return self.fc_out(x_out)

class Network_sin(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_sin, self).__init__()
        #
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
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
        return torch.sin(np.pi * x)

    def forward(self, x):
        #
        x = 2. * (x-self.lb) / (self.ub- self.lb) - 1.
        x = self.fun_sin(self.fc_in(x))
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.fun_sin(fc_hidden(x)) + x
        x_out = x

        return self.fc_out(x_out)

class Network_relu(nn.Module):
    '''
    '''
    def __init__(self, d_in:int=1, d_out:int=1, 
                 hidden_size:int=100, hidden_layers=4, **kwargs):
        super(Network_relu, self).__init__()
        #
        self.activation = torch.nn.ReLU()
        self.fc_in = nn.Linear(d_in, hidden_size)
        self.fc_out = nn.Linear(hidden_size, d_out)
        #
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
        x = self.fc_in(x)
        #
        for fc_hidden in self.fc_hidden_list:
            x = self.activation( fc_hidden(torch.sin(np.pi * x)) ) + x

        return self.fc_out(x)
    
class Model():
    '''
    '''
    def __init__(self, model_type:str, device=None, dtype:torch.dtype=torch.float32):
        self.model_type = model_type
        self.device = device
        torch.set_default_dtype(dtype)
    
    def get_model(self, d_in:int=1, d_out:int=1, h_size:int=200, h_layers:int=3, **kwargs):
        if self.model_type=='sin':
            return Network_sin(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                               hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='relu':
            return Network_relu(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh':
            return Network_tanh(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                hidden_layers=h_layers, **kwargs).to(self.device)
        elif self.model_type=='tanh_sin':
            return Network_tanh_sin(d_in=d_in, d_out=d_out, hidden_size=h_size, 
                                    hidden_layers=h_layers, **kwargs).to(self.device)
        else:
            raise NotImplementedError(f'No network model {self.model_type}.')