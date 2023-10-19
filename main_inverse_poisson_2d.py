# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-28 02:00:59 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-28 02:00:59 
#  */
import numpy as np
import torch

class Example():

    def __init__(self, np_type, torch_type):
        '''
        ''' 
        self.np_type = np_type
        self.torch_type = torch_type

    def Solver_ParticleWNN(self, save_path:str, load_path:str, 
                           load_type:str, action:str, **args):
        '''
        '''
        try:
            lr = args['lr']
        except:
            lr = 1e-3 
        try:
            noise_level = args['noise_level']
        except:
            noise_level = 0.01
        ############ 
        kwargs = {'N_particle': 200,
                  'N_int': 10, 
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  'maxIter': 20001,
                  'lr': lr,
                  'model_type': 'FeedForward',
                  'data_type': {'numpy':self.np_type, 'torch':self.torch_type},
                  'lr_Decay': 1.,
                  'loss_weight': {'eq':1., 'ob_in':5., 'ob_bd':5.},
                  'topK': 150,
                  'int_method': 'mesh', 
                  'hidden_width': 50,
                  'hidden_layer': 3,
                  'activation': 'tanh',
                  'noise_level': noise_level,
                  }
        #########
        from Solvers.ParticleWNN_Inverse import Solver
        problem = Problem(dtype=self.np_type, testFun_type='Wendland')
        solver = Solver(problem=problem, **kwargs)
        if action=='train':
            solver.train(save_path, load_path, load_type)
        else:
            solver.predict(load_path, load_type)

    def train(self, inx:int, save_path:str, load_path:str, solver_name:str,
              **args):
        '''
        '''
        if solver_name=='ParticleWNN':
            demo.Solver_ParticleWNN(save_path=save_path+f'{solver_name}_{inx}/', 
                                    load_path=load_path+f'{solver_name}_{inx}/', 
                                    load_type='model_best_error', 
                                    action='train', **args)

    def pred(self, inx:int, load_path:str, solver_name:str, **args):
        '''
        '''
        if solver_name=='ParticleWNN':
            demo.Solver_ParticleWNN(save_path=None, 
                                    load_path=load_path+f'{solver_name}_{inx}/', 
                                    load_type= 'model_best_error', 
                                    action='predict', **args)

if __name__=='__main__':
    from Problems.Inverse_Poisson_2d import Problem
    demo = Example(np_type=np.float64, torch_type=torch.float64)
    #
    solver_names = ['ParticleWNN']
    path = f"./savedModel/{Problem().name}/"
    args = {'lr':1e-3, 'noise_level': 0.01}
    ###########################################
    for inx in range(5):
        for name in solver_names:
            print(f'************************ Method: {name} *************************************')
            demo.train(inx=inx, save_path=path, load_path=path, solver_name=name, **args)
            demo.pred(inx=inx, load_path=path, solver_name=name, **args)

