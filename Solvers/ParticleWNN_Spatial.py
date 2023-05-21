# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:06:35 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:06:35 
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
    def __init__(self, Problem:Problem,
                 Num_particles:int, Num_integral:int, 
                 train_bd_size_each_face:int, R_max:float, 
                 maxIter:int, lr:float, net_type:str, 
                 **kwargs):
        #
        self.Num_particles = Num_particles
        self.Num_integral = Num_integral
        self.train_bd_size_each_face = train_bd_size_each_face
        self.Rmax = R_max
        self.iters = maxIter
        self.lr_u = lr
        self.net_type = net_type
        # Other settings
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
        self.problem = Problem(test_type=kwargs['test_fun'], dtype=self.dtype['numpy'])
        self.data = GenData(self.problem, dtype=self.dtype['numpy'])

    def _save(self, save_path:str, model_type:str)->None:
        '''
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # 
        if model_type=='model_final':
            dict_loss = {}
            dict_loss['loss_u'] = self.loss_u_list
            dict_loss['error'] = self.error
            dict_loss['time'] = self.time_list
            scipy.io.savemat(save_path+f'loss_error_saved.mat', dict_loss)
        # 
        torch.save(self.model_u.state_dict(), save_path+f'trained_{model_type}.pth')

    def _load(self, load_path:str, model_type:str='model_best_loss')->None:
        '''
        '''
        try:
            self.model_u.load_state_dict(torch.load(load_path+f'trained_{model_type}.pth'))
        except:
            self.get_net()
            self.model_u.load_state_dict(torch.load(load_path+f'trained_{model_type}.pth'))

    def test(self, save_path:str, model_type='model_best_loss')->None:
        '''
        '''
        # load the trained model
        self._load(save_path, model_type)
        #
        u_test, x_test= self.problem._fun_u()
        with torch.no_grad():
            u_pred = self.model_u(x_test.to(device))
        # save test data
        dict_test = {}
        dict_test['x_test'] = x_test.detach().cpu().numpy()
        dict_test['u_test'] = u_test.detach().cpu().numpy()
        dict_test['u_pred'] = u_pred.detach().cpu().numpy()
        scipy.io.savemat(save_path+f'test_saved.mat', dict_test)

    def get_net(self)->None:
        '''
        '''
        # 
        kwargs = {'d_in':self.problem.dim,
                  'h_size': self.hidden_n,
                  'h_layers': self.hidden_l,
                  'lb':torch.from_numpy(self.problem.lb).to(device), 
                  'ub':torch.from_numpy(self.problem.ub).to(device)}
        self.model_u = Model(self.net_type, device, dtype=self.dtype['torch']).get_model(**kwargs)
        # 
        self.optimizer_u = torch.optim.Adam(self.model_u.parameters(), lr=self.lr_u)
        #
        self.scheduler_u = torch.optim.lr_scheduler.StepLR(
            self.optimizer_u, 1, gamma=(1.-self.lrDecay/self.iters), last_epoch=-1)

    def get_loss(self, **args):
        '''
        '''
        ########### Residual inside the domain
        x_scaled = self.data.get_x_scaled(Nx_scaled=self.Num_integral, method=self.int_method)
        #
        R, xc= self.data.get_xc(N_xc=self.Num_particles, R_max=args['Rmax'], R_min=args['Rmin'])
        #
        weak = self.problem.weak(self.model_u, x_scaled.to(device), xc.to(device), R.to(device)) 
        weak_form = weak ** 2
        #
        weak_topk, _ = torch.topk(weak_form, k=self.topk, dim=0)
        loss = torch.mean( weak_topk ) * self.w_in
        ############ mismatch on the boundary
        x_bd_list = self.data.get_bd(N_bd_each_face=self.train_bd_size_each_face)
        self.x_bd = torch.cat(x_bd_list, dim=0)
        #
        x_bd_list = [item.to(device) for item in x_bd_list]
        u_bd_pred = self.problem.fun_u_bd(x_bd_list, model_u=self.model_u)
        u_bd_true = self.problem.fun_u_bd(x_bd_list)
        loss += torch.mean( (u_bd_pred - u_bd_true)**2 ) * self.w_bd

        return loss

    def train(self, save_path:str)->None:
        '''
        '''
        t_start = time.time()
        self.get_net()
        # 
        u_valid, x_valid = self.problem._fun_u()
        # 
        iter = 0
        best_err, best_loss = 1e10, 1e10
        self.time_list = []
        self.loss_u_list = []
        self.error = []
        for iter in range(self.iters):
            if self.Rway=='Rfix':
                R_adaptive = self.Rmax 
            elif self.Rway=='Rascend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * iter/self.iters
            elif self.Rway=='Rdescend':
                R_adaptive = self.Rmin  + (self.Rmax-self.Rmin) * (1-iter/self.iters)
            loss_u_train = self.get_loss(**{'Rmax':R_adaptive, 'Rmin':self.Rmin})
            # Train the network
            self.optimizer_u.zero_grad()
            loss_u_train.backward()
            self.optimizer_u.step()
            self.scheduler_u.step()
            # Save loss and error
            iter += 1
            self.loss_u_list.append(loss_u_train.item())
            self.time_list.append(time.time()-t_start)
            with torch.no_grad():
                u_pred_valid = self.model_u(x_valid.to(device))
                error_valid = Error().L2_error(u_pred_valid, u_valid.to(device))
                self.error.append(error_valid)
                # Save network model (best loss)
                if (loss_u_train.item()) < best_loss:
                    best_loss = loss_u_train.item()
                    self._save(save_path, model_type='model_best_loss')
                # 
                if iter%100 == 0:
                    print(f"At iter: {iter+1}, error:{self.error[-1]:.4f}, \
                        loss_u:{np.mean(self.loss_u_list[-50:]):.4f}, R:{R_adaptive:.6f}")
        # Save network model (final)
        self._save(save_path, model_type='model_final')
        print(f'The total time is {time.time()-t_start:.4f}.')