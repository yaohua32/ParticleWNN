# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:06:22 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:06:22 
#  */
import numpy as np 
import scipy.io
import time
import os
import torch
#
from Network.FeedForward import Model
from Utils.Error import Error
# 
from Utils.GenData_Spatial import GenData
from Problems.Module_Spatial import Problem
import Solvers.Module as Module
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
try:
    print(f'{torch.cuda.get_device_name(0)}')
except:
    pass

class ParticleWNN(Module.Solver):
    '''
    '''
    def __init__(self, problem:Problem(),
                 Num_particles:int, Num_integral:int, 
                 x_sensor_in:int, x_sensor_bd:int,
                 R_max:float, maxIter:int, lr:float, net_type:str, 
                 **kwargs):
        #
        self.Num_particles = Num_particles
        self.Num_integral = Num_integral
        self.x_sensor_in = x_sensor_in 
        self.x_sensor_bd= x_sensor_bd
        self.Rmax = R_max
        self.iters = maxIter
        self.lr = lr
        self.net_type = net_type
        # Other settings
        self.noise_level = kwargs['noise_level']
        self.w_measure = kwargs['w_measure']
        self.Rway = kwargs['R_way']
        self.Rmin = kwargs['R_min']
        self.w_in = kwargs['w_in']
        self.w_bd = kwargs['w_bd']
        self.topk = kwargs['topk']
        self.int_method = kwargs['int_method']
        self.hidden_n = kwargs['hidden_width']
        self.hidden_l = kwargs['hidden_layer']
        self.dtype = kwargs['dtype']
        self.lrDecay = 2.
        #
        self.problem = problem
        self.data = GenData(self.problem, dtype=self.dtype['numpy'])

    def _save(self, save_path:str, model_type:str)->None:
        '''
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 
        if model_type=='model_final':
            dict_loss = {}
            dict_loss['loss_u'] = self.loss_train_list
            dict_loss['error_u'] = self.error_u 
            dict_loss['error_k'] = self.error_k
            dict_loss['time'] = self.time_list
            scipy.io.savemat(save_path+f'loss_error_saved.mat', dict_loss)
        # 
        model_dict = {'model_u':self.model_u.state_dict(), 
                      'model_k':self.model_k.state_dict()}
        torch.save(model_dict, save_path+f'trained_{model_type}.pth')

    def _load(self, load_path:str, model_type:str='model_best_loss')->None:
        '''
        '''
        model_dict = torch.load(load_path+f'trained_{model_type}.pth')
        try:
            self.model_u.load_state_dict(model_dict['model_u'])
            self.model_k.load_state_dict(model_dict['model_k'])
        except:
            self.get_net() 
            self.model_u.load_state_dict(model_dict['model_u'])
            self.model_k.load_state_dict(model_dict['model_k'])

    def test(self, save_path:str, model_type='model_best_loss')->None:
        '''
        '''
        # load the trained model
        self._load(save_path, model_type)
        #
        u_test, x_test = self.problem._fun_u()
        k_test = self.problem._fun_k(x_test)
        with torch.no_grad():
            u_pred = self.model_u(x_test.to(device))
            k_pred = self.model_k(x_test.to(device))
        #
        dict_test = {}
        dict_test['x_test'] = x_test.detach().cpu().numpy()
        dict_test['u_test'] = u_test.detach().cpu().numpy()
        dict_test['u_pred'] = u_pred.detach().cpu().numpy()
        dict_test['k_test'] = k_test.detach().cpu().numpy()
        dict_test['k_pred'] = k_pred.detach().cpu().numpy()
        scipy.io.savemat(save_path+f'test_saved.mat', dict_test)

    def get_net(self)->None:
        '''
        '''
        kwargs = {'d_in':self.problem.dim,
                  'h_size': self.hidden_n,
                  'h_layers': self.hidden_l,
                  'lb':torch.from_numpy(self.problem.lb).to(device), 
                  'ub':torch.from_numpy(self.problem.ub).to(device)}
        self.model_u = Model(self.net_type, device, dtype=self.dtype['torch']).get_model(**kwargs)
        self.model_k = Model(self.net_type, device, dtype=self.dtype['torch']).get_model(**kwargs)
        # 
        self.optimizer = torch.optim.Adam([
            {'params': self.model_u.parameters(), 'lr': self.lr}, 
            {'params': self.model_k.parameters(), 'lr': self.lr}
            ])
        #
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=(1. - self.lrDecay/self.iters), last_epoch=-1)

    def get_loss(self, **kwargs):
        '''
        '''
        ########### Residual inside the domain
        x_scaled = self.data.get_x_scaled(Nx_scaled=self.Num_integral, method=self.int_method)
        #
        R, xc= self.data.get_xc(N_xc=self.Num_particles, R_max=kwargs['Rmax'], R_min=kwargs['Rmin'])
        #
        weak = self.problem.weak(self.model_u, self.model_k, x_scaled.to(device), xc.to(device), R.to(device)) 
        weak_form = weak ** 2
        #
        weak_topk, _ = torch.topk(weak_form, k=self.topk, dim=0)
        loss = torch.mean(weak_topk) * self.w_in
        ############ mismatch inside the domain
        u_pred_in = self.model_u(self.x_sensor_in.to(device))
        u_mea_in = (self.problem._fun_u(self.x_sensor_in.to(device)) 
                    + torch.randn_like(u_pred_in) * self.noise_level)
        #
        loss += torch.mean( (u_pred_in - u_mea_in)**2 ) * self.w_measure
        ############ mismatch on the boundary
        u_pred_bd = self.model_u(self.x_sensor_bd.to(device))
        u_mea_bd = (self.problem._fun_u(self.x_sensor_bd.to(device)) 
                    + torch.randn_like(u_pred_bd) * self.noise_level)
        #
        loss += torch.mean( (u_pred_bd - u_mea_bd)**2 ) * self.w_bd

        return loss

    def train(self, save_path:str)->None:
        '''
        '''
        t_start = time.time()
        self.get_net()
        #
        u_valid, x_valid = self.problem._fun_u()
        k_valid = self.problem._fun_k(x_valid)
        # 
        iter = 0
        best_loss = 1e10
        self.time_list = []
        self.loss_train_list = []
        self.error_u, self.error_k = [], []
        for iter in range(self.iters):
            if self.Rway=='Rfix':
                R_adaptive = self.Rmax 
            elif self.Rway=='Rascend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * iter/self.iters
            elif self.Rway=='Rdescend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * (1-iter/self.iters)
            loss_train = self.get_loss(**{'Rmax':R_adaptive, 'Rmin':self.Rmin})
            # Train the network
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            self.scheduler.step()
            # Save loss and error
            iter += 1
            self.loss_train_list.append(loss_train.item())
            self.time_list.append(time.time()-t_start)
            with torch.no_grad():
                u_pred_valid = self.model_u(x_valid.to(device))
                k_pred_valid = self.model_k(x_valid.to(device))
                error_u_valid = Error().L2_error(u_pred_valid, u_valid.to(device))
                error_k_valid = Error().L2_error(k_pred_valid, k_valid.to(device))
                self.error_u.append(error_u_valid)
                self.error_k.append(error_k_valid)
                # Save network model (best loss)
                if loss_train.item() < best_loss:
                    best_loss = loss_train.item()  
                    self._save(save_path, model_type='model_best_loss')
                # 
                if iter%100 == 0:
                    print(f"At iter: {iter+1}, error_u:{self.error_u[-1]:.4f},\
                          error_k:{self.error_k[-1]:.4f}, \
                          loss:{np.mean(self.loss_train_list[-50:]):.4f}")
        # Save network model (final)
        self._save(save_path, model_type='model_final')
        print(f'The total time is {time.time()-t_start:.4f}.')