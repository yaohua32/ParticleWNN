# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-28 01:24:49 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-28 01:24:49 
#  */
import numpy as np 
import torch
import time
import scipy.io
import os
#
from Network.Network import Model
from Problems.Module import Problem
from Utils.Error import Error
#
import Solvers.Module as Module
#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))
try:
    print(f'{torch.cuda.get_device_name(0)}')
except:
    pass

class Solver(Module.Solver):

    def __init__(self, problem:Problem,
                 N_particle:int,
                 N_int:int, R_max:int, R_min:int,
                 maxIter:int, lr:float, model_type:str,
                 data_type:dict, **kwargs):
        '''
        Input:
            problem: 
            N_particle: the number of particles
            N_int: the number of meshgrids (or meshsize) for computing integration 
            R_max: the maximum of the compact supports' Radius
            R_min: the minimum of the compact supports' Radius
            maxIter: the maximum of iterations
            lr: the learning rate
            model_type: 'FeedForward' or 'FeedForward_Sin'
            data_type: {'numpy', 'torch'}
            kwargs: 
        '''
        #
        self.problem = problem
        #
        self.N_particle = N_particle
        self.N_int = N_int
        self.R_max = R_max
        self.R_min = R_min
        self.maxIter = maxIter
        self.lr = lr
        self.model_type = model_type
        self.data_type = data_type
        # Other parameter setting
        self.lr_Decay = kwargs['lr_Decay']
        self.loss_weight = kwargs['loss_weight']
        self.topK = kwargs['topK']
        self.int_method = kwargs['int_method']
        self.hidden_n = kwargs['hidden_width']
        self.hidden_l = kwargs['hidden_layer']
        self.activation = kwargs['activation']
        self.noise_level = kwargs['noise_level']

    def _load(self, model:torch.nn.Module, load_path:str, load_type:str)->None:
        '''
        Input:
            model: torch.nn.Module
            load_path: the path of the trained model to be loaded
            load_type: 'model_best_error', 'model_best_loss', 'model_final'
        '''
        model_dict = torch.load(load_path+f'{load_type}.pth', 
                                map_location=torch.device(device))
        model.load_state_dict(model_dict['model'])

    def _save(self, model:torch.nn.Module, save_path:str, save_type:str)->None:
        '''
        Input: 
            model: torch.nn.Module
            save_path: the path for the trained model to be saved
            save_type: 'model_final', 'model_best_error', 'model_best_loss'
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save the loss and error if save_type=='model_final'
        if save_type=='model_final':
            dict_loss = {}
            dict_loss['loss'] = self.loss_list
            dict_loss['error_u'] = self.error_u
            dict_loss['error_k'] = self.error_k
            dict_loss['time'] = self.time_list
            scipy.io.savemat(save_path+f'loss_error_saved.mat', dict_loss)
        # save the trained model
        model_dict = {'model':model.state_dict()}
        torch.save(model_dict, save_path+f'{save_type}.pth')

    def predict(self, load_path:str, load_type:str)->None:
        '''
        Input:
            load_path: the path of the trained model
            load_type: 'model_best_error', 'model_best_loss', 'model_final'
        '''
        # load the trained model
        self.get_net()
        self._load(self.model, load_path, load_type)
        # prediction
        uk_test, x_test = self.problem.get_test()
        u_test, k_test = uk_test[:,0:1], uk_test[:,1:2]
        with torch.no_grad():
            uk_pred = self.model(x_test.to(device))
            u_pred, k_pred = uk_pred[:,0:1], uk_pred[:,1:2]
        # save result
        dict_test = {}
        dict_test['x_test'] = x_test.detach().cpu().numpy()
        dict_test['u_test'] = u_test.detach().cpu().numpy()
        dict_test['u_pred'] = u_pred.detach().cpu().numpy()
        dict_test['k_test'] = k_test.detach().cpu().numpy()
        dict_test['k_pred'] = k_pred.detach().cpu().numpy()
        scipy.io.savemat(load_path+f'test_saved.mat', dict_test)
        #
        print(f'Results have been saved in  {load_path}!')

    def get_net(self)->None:
        '''
        '''
        ##### The model structure
        kwargs = {'d_in':self.problem.dim,
                  'd_out': 2,
                  'h_size': self.hidden_n,
                  'l_size': self.hidden_l,
                  'activation': self.activation,
                  'lb': self.problem.lb.to(device), 
                  'ub': self.problem.ub.to(device)}
        self.model = Model(self.model_type, device, dtype=self.data_type['torch']).get_model(**kwargs)
        ###### The optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.lr}])
        ####### The scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=(1. - self.lr_Decay/self.maxIter), 
            last_epoch=-1)

    def get_loss(self, data_point:dict, data_observe:dict, **args):
        '''
        Input:
            data_point: dict={'xc', 'R'}
            data_observe: dict={'x_ob_in', 'u_ob_in', 'x_ob_bd', 'u_ob_bd'}
            args: 'x_mesh', 'phi', 'dphi_scaled'
        Output:
            loss_train:
            loss_all: [loss_in, loss_bd]
        '''
        loss_all = []
        ############ mismatch in the domain
        uk_ob_in_pred = self.model(data_observe['x_ob_in'].to(device))
        u_ob_in_pred = uk_ob_in_pred[:,0:1]
        u_ob_in_noise = (data_observe['u_ob_in'].to(device) 
                         + torch.randn_like(u_ob_in_pred) * self.noise_level)
        #
        loss_ob_in = torch.mean( (u_ob_in_pred - u_ob_in_noise)**2 )
        ############ mismatch on the boundary
        uk_ob_bd_pred = self.model(data_observe['x_ob_bd'].to(device))
        u_ob_bd_pred = uk_ob_bd_pred[:,0:1]
        u_ob_bd_noise = (data_observe['u_ob_bd'].to(device) 
                         + torch.randn_like(u_ob_bd_pred) * self.noise_level)
        #
        loss_ob_bd = torch.mean( (u_ob_bd_pred - u_ob_bd_noise)**2 )
        ########### Residual in the domain
        eq = self.problem.weak_particlewnn(model=self.model, 
                                           xc=data_point['xc'].to(device),
                                           R=data_point['R'].to(device), 
                                           x_mesh=args['x_mesh'].to(device),
                                           phi=args['phi'].to(device), 
                                           dphi_scaled=args['dphi_scaled'].to(device))
        eq = eq**2 
        try:
            eq_topk, index = torch.topk(eq, k=self.topK, dim=0)
        except:
            eq_topk = eq
        #
        loss_eq = torch.mean( eq_topk )
        ##############
        loss_all.append([loss_eq.detach(), loss_ob_in.detach(), loss_ob_bd.detach()])
        #
        loss_train = (loss_eq * self.loss_weight['eq'] 
                      + loss_ob_in * self.loss_weight['ob_in']
                      + loss_ob_bd * self.loss_weight['ob_bd'])
        
        return loss_train, loss_all

    def train(self,save_path:str, load_path:str=None, 
              load_type:str=None)->None:
        '''
        Input: 
            save_path: 
            load_path: path for loading trained model
            load_type: 'model_final', 'model_best_loss', 'model_best_error'
        '''
        t_start = time.time()
        ##############################################
        self.get_net()
        try:
            self._load(self.model, load_path=load_path, load_type=load_type)
            print('***********  A trained model has been loaded ...... ***************')
        except:
            print('*********** Started with a new model ...... ***************')
        ##############################################
        uk_test, x_test = self.problem.get_test()
        #
        x_mesh, phi, dphi_scaled = self.problem.get_testFun_particlewnn(
            GenData=self.problem._gen_data, TestFun=self.problem._testFun_particlewnn,
            Nx_mesh=self.N_int, int_method=self.int_method)
        #
        data_observe = self.problem.get_observe()
        ##############################################
        iter = 0
        best_err, best_loss = 1e10, 1e10
        self.time_list, self.loss_list = [], []
        self.error_u, self.error_k = [], []
        for iter in range(self.maxIter):
            #
            R_adaptive = self.R_min  + (self.R_max-self.R_min) * iter/self.maxIter
            data_train = self.problem.get_point_particlewnn(
                GenData=self.problem._gen_data, 
                N_xin=self.N_particle, N_xbd_each_face=None,
                **{'Rmax': R_adaptive, 'Rmin':self.R_min})
            #
            loss_train, loss_all = self.get_loss(
                data_train, data_observe, 
                **{'x_mesh': x_mesh, 'phi':phi, 'dphi_scaled':dphi_scaled})
            ###########################################
            self.loss_list.append(loss_train.item())
            self.time_list.append(time.time()-t_start)
            #
            err, _ = Error().L2_error(
                model=self.model, x=x_test.to(device), u=uk_test.to(device))
            self.error_u.append(err[0].item())
            self.error_k.append(err[1].item())
            #
            err_avg = torch.mean(err)
            if err_avg.item() < best_err:
                best_err = err_avg.item()
                self._save(self.model, save_path, save_type='model_best_error')
            if loss_train.item() < best_loss:
                best_loss = loss_train.item()
                self._save(self.model, save_path, save_type='model_best_loss')
            #
            if iter%100 == 0:
                print(f"At iter: {iter}, loss:{loss_train.item():.4f}, err_u:{err[0].item():.4f}, err_k:{err[1].item():.4f}")
                # print(f'---loss:', torch.mean(torch.tensor(loss_all), dim=0))
            ############################################# 
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            self.scheduler.step()
            iter += 1
        ##############################################
        self._save(self.model, save_path, save_type='model_final')
        print(f'The total training time is {time.time()-t_start:.4f}')