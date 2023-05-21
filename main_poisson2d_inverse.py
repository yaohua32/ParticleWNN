# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:08:52 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:08:52 
#  */
import numpy as np
import torch
import scipy.io
from Utils.Draw import Draw

class Example():

    def __init__(self, np_type=np.float32, torch_type=torch.float32):
        '''
        ''' 
        self.np_type = np_type
        self.torch_type = torch_type

    def show_setting(self, save_path):
        '''
        '''
        from Problems.Poisson2d_inverse import Problem
        settings = np.load(save_path+'problem_settings.npy', allow_pickle=True)
        ############### 测试 problem
        x_sensor_in = settings.item().get('x_sensor_in')
        x_sensor_bd = settings.item().get('x_sensor_bd')
        k_center = settings.item().get('k_center')
        k_sigma = settings.item().get('k_sigma')
        #
        problem = Problem(k_center=k_center, k_sigma=k_sigma)
        u_test, x_test = problem._fun_u()
        k_test = problem._fun_k(x_test)
        #
        Draw().show_2d(x_test, u_test, title='u(x,y)', 
                       points_list=[x_sensor_in, x_sensor_bd],
                       save_path='./results/poisson2d_inverse_u.pdf')
        #
        Draw().show_2d(x_test, k_test, title=r'a(x,y)',
                       save_path='./results/poisson2d_inverse_a.pdf')

    def get_error(self, noise:float, net_type:str, names:list, num:int=5):
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
                save_path = f"./savedModel/{Problem().name}_{net_type}/noise_{noise}/{names[i]}_{j}/"
                loss_err_save = scipy.io.loadmat(save_path+'loss_error_saved.mat')
                data_test = scipy.io.loadmat(save_path + 'test_saved.mat')
                #
                times.append(loss_err_save['time'])
                l2_errs.append(loss_err_save['error_k'])
                x_test += data_test['x_test']
                abs += np.abs(data_test['k_pred'] - data_test['k_test'])
                #
                l2_error_best[i,j] =  min(loss_err_save['error_k'][0])
                #
                abs_error_best[i,j] = np.max(np.abs(data_test['k_pred'][:,0] - data_test['k_test'][:,0]))
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
        Draw().show_confid(times_list, l2_errs_list, 
                           name_list=names, 
                           x_name='time(s)', 
                           y_name=r'Relative error (avg)',
                           confid=True,
                           save_path='./results/poisson2d_inverse_l2err.pdf')
        ############### point-wise err
        Draw().show_error_2d(x_test_list, abs_list, 
                             label_list=names,
                             title_list=[r'$avg. |a_{ParticleWNN} - a|$'],
                             save_path='./results/poisson2d_inverse_mae.pdf')

    def solve(self, solver, save_path:str, train:bool=True):
        '''
        '''
        if train:
            solver.train(save_path)
            solver.test(save_path)

    def get_measuremets(self, n_inside:int=None, n_bd_each_side:int=None, 
                        noise_level:float=None, save_path=None):
        '''
        '''
        try:
            # load saved data
            settings = np.load(save_path+'problem_settings.npy', allow_pickle=True)
            x_sensor_in = settings.item().get('x_sensor_in')
            x_sensor_bd = settings.item().get('x_sensor_bd')
            k_center = settings.item().get('k_center')
            k_sigma = settings.item().get('k_sigma')
            #
            print('***************Load saved sensors.******************')
            #
        except:
            problem = Problem(dtype=self.np_type)
            # generate new data
            x_sensor_in, x_sensor_bd = problem.get_sensors(n_inside=n_inside, 
                                                           n_bd_each_side=n_bd_each_side)
            x_sensor_bd = torch.cat(x_sensor_bd, dim=0)
            k_center = problem._k_center
            k_sigma = problem._k_sigma
            #
            print('*****************Generate new sensors.*****************')
        ############
        self.noise_level = noise_level
        self.problem_kwargs ={
                  'x_sensor_in': x_sensor_in,
                  'x_sensor_bd': x_sensor_bd,
                  'k_center': k_center, 
                  'k_sigma': k_sigma, 
                  'noise_level': self.noise_level
                  }

    def WNN(self, inx:int, name:str='ParticleWNN'):
        '''
        ParticleWNN
        '''
        from Solvers.ParticleWNN_Inverse import ParticleWNN
        ############  
        kwargs = {
                  'Num_particles': 200,
                  'Num_integral': 10,
                  'topk': 150,
                  'maxIter': 20000,
                  'lr': 1e-3,
                  'w_in': 1.,
                  'w_measure': 5.,
                  'w_bd': 5.,
                  'R_way': 'Rdescend',
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  'net_type': 'tanh',
                  'test_fun': 'Wendland',
                  'int_method': 'mesh', 
                  'hidden_width': 50,
                  'hidden_layer': 3,
                  'dtype':{'numpy':self.np_type, 'torch':self.torch_type},
                  }
        problem = Problem(test_type=kwargs['test_fun'], dtype=self.np_type,
                          k_center=self.problem_kwargs['k_center'], 
                          k_sigma=self.problem_kwargs['k_sigma'])
        #
        print(f'{name}******** noise_level:', self.noise_level, kwargs)
        save_path = f"./savedModel/{problem.name}_{kwargs['net_type']}/noise_{self.noise_level}/{name}_{inx}/"
        ############## 
        solver = ParticleWNN(problem=problem, **self.problem_kwargs, **kwargs)
        self.solve(solver=solver, save_path=save_path, train=True)
        ############## 
        np.save(save_path+'settings.npy', kwargs)
        np.save(save_path+'problem_settings.npy', self.problem_kwargs)

if __name__=='__main__':
    from Problems.Poisson2d_inverse import Problem
    ############################################
    demo = Example(np_type=np.float64, torch_type=torch.float64)
    save_path = f'./savedModel/poisson2d_inverse_tanh/settings/'
    #
    noise_level = [0.01, 0.1]
    for noise in noise_level:
        demo.get_measuremets(n_inside=None, n_bd_each_side=None, 
                             noise_level=noise, save_path=save_path)
        print(f'****************noise:{noise}******************************')
        for inx in range(5):
            demo.WNN(inx=inx)
    ########################## u & a
    demo.show_setting(save_path)
    #########################
    demo.get_error(noise=0.1, net_type='tanh', names=['ParticleWNN'], num=5)
            