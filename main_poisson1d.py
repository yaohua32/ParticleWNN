# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:08:41 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:08:41 
#  */
import numpy as np
import torch
import scipy.io
from Utils.Draw import Draw

class Example():

    def __init__(self, np_type=np.float32, torch_type=torch.float32, train:bool=True):
        '''
        ''' 
        self.np_type = np_type
        self.torch_type = torch_type
        self.train = train

    def get_error(self, net_type:str, names:list, num:int=5):
        '''
        '''
        l2_error_best = np.empty((len(names), num))
        time = np.empty((len(names), num))
        abs_error_best = np.empty((len(names), num))
        #
        times_list = []
        l2_errs_list = []
        x_test_list, abs_list = [], []
        for i in range(len(names)):
            times, l2_errs = [], []
            x_test, abs = 0., 0.
            for j in range(num):
                save_path = f"./savedModel/{Problem().name}_{net_type}/{names[i]}_{j}/"
                loss_err_save = scipy.io.loadmat(save_path+'loss_error_saved.mat')
                data_test = scipy.io.loadmat(save_path + 'test_saved.mat')
                #
                times.append(loss_err_save['time'])
                l2_errs.append(loss_err_save['error'])
                x_test += data_test['x_test']
                abs += np.abs(data_test['u_pred'] - data_test['u_test'])
                #
                l2_error_best[i,j] =  min(loss_err_save['error'][0])
                #
                abs_error_best[i,j] = np.max(np.abs(data_test['u_pred'][:,0] - data_test['u_test'][:,0]))
                #
                time[i,j] = loss_err_save['time'][0][-1]
            ####
            times_list.append(np.concatenate(times, axis=0))
            l2_errs_list.append(np.concatenate(l2_errs, axis=0))
            x_test_list.append(x_test/num)
            abs_list.append(abs/num)
        ##################
        print('abs err (avg):', np.mean(abs_error_best, axis=1), 
              'abs err (std):', np.std(abs_error_best, axis=1))
        print('l2 err (avg):', np.mean(l2_error_best, axis=1), 
              'l2 err (std):', np.std(l2_error_best, axis=1))
        print('Time (avg):', np.mean(time, axis=1), 
              'Time (std):', np.std(time, axis=1))
        ################ Relative error vs. time
        Draw().show_confid(times_list, l2_errs_list, names, 
                           x_name='time(s)', y_name=r'Relative error (avg)',
                           confid=True,
                           save_path='./results/poisson1d_l2err.pdf')
        ############### point-wise error (avg)
        Draw().show_1d_list(x_test_list, abs_list, label_list=names,
                            save_path='./results/poisson1d_point_wise.pdf')

    def solve(self, solver, save_path:str, train:bool=True):
        '''
        '''
        if train:
            solver.train(save_path)
            solver.test(save_path)

    def WNN(self, inx:int, name:str='ParticleWNN'):
        '''
        '''
        from Solvers.ParticleWNN_Spatial import ParticleWNN
        ############ 
        kwargs = {'Num_particles': 200,
                  'Num_integral': 49,
                  'topk': 150,
                  'train_bd_size_each_face': 1,
                  'maxIter': 20000,
                  'lr': 1e-3,
                  'w_in': 1.,
                  'w_bd': 5.,
                  'R_way': 'Rdescend',
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  'net_type': 'sin',
                  'test_fun': 'Wendland',
                  'int_method': 'mesh',
                  'hidden_width': 50,
                  'hidden_layer': 3,
                  'dtype':{'numpy':self.np_type, 'torch':self.torch_type}
                  }
        save_path = f"./savedModel/{Problem().name}_{kwargs['net_type']}/{name}_{inx}/"
        #
        print(f'{name}********:{inx}\n', kwargs)
        ##############
        solver = ParticleWNN(Problem=Problem, **kwargs)
        self.solve(solver=solver, save_path=save_path, train=self.train)
        np.save(save_path+'settings.npy', kwargs)

if __name__=='__main__':
    from Problems.Poisson1d import Problem
    demo = Example(np_type=np.float64, torch_type=torch.float64, train=True)
    ############################################
    for inx in range(5):
        print(f'***************inx:{inx}*************')
        demo.WNN(inx=inx)
    ############################################
    demo.get_error(net_type='sin', names=['ParticleWNN'], num=5)