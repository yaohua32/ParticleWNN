# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-27 20:06:17 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-27 20:06:17 
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
            freq = args['freq']
        except:
            freq = 3.*np.pi
        ############ 
        kwargs = {'N_particle': 250,
                  'N_int': 14, 
                  'N_bd_each_face': 250,
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  'maxIter': 100001,
                  'lr': lr,
                  'model_type': 'FeedForward',
                  'data_type': {'numpy':self.np_type, 'torch':self.torch_type},
                  'lr_Decay': 1.,
                  'loss_weight': {'eq':1., 'bd':5., 'ob':1.},
                  'topK': 250,
                  'int_method': 'mesh',
                  'hidden_width': 50,
                  'hidden_layer': 3,
                  'activation': 'tanh',
                  }
        ######### 
        from Solvers.ParticleWNN_NS_Steady import Solver
        problem = Problem(dtype=self.np_type, testFun_type='Wendland',
                          **{'freq':freq})
        solver = Solver(problem=problem, **kwargs)
        if action=='train':
            solver.train(save_path, load_path, load_type)
        else:
            solver.predict(load_path, load_type)

    def train(self, inx:int, save_path:str, load_path:str, solver_name:str, **args):
        '''
        '''
        if solver_name=='ParticleWNN':
            demo.Solver_ParticleWNN(save_path=save_path+f'{solver_name}_{inx}/', 
                                    load_path=load_path+f'{solver_name}_{inx}/', 
                                    load_type='model_best_error', 
                                    action='train', 
                                    **{'lr':args['lr'],'freq':args['freq']})

    def pred(self, inx:int, load_path:str, solver_name:str, **args):
        '''
        '''
        if solver_name=='ParticleWNN':
            demo.Solver_ParticleWNN(save_path=None, 
                                    load_path=load_path+f'{solver_name}_{inx}/', 
                                    load_type= 'model_best_error', 
                                    action='predict',
                                    **{'lr':args['lr'],'freq':args['freq']})

if __name__=='__main__':
    from Problems.NS_Kovasznay_2d import Problem
    demo = Example(np_type=np.float64, torch_type=torch.float64)
    #
    solver_names = ['ParticleWNN']
    path = f"./savedModel/{Problem().name}/"
    args = {'lr':1e-3, 'freq':3.*np.pi}
    ###########################################
    for inx in range(5):
        for name in solver_names:
            print(f'************************ Method: {name} *************************************')
            demo.train(inx=inx, save_path=path, load_path=path, solver_name=name, **args)
            demo.pred(inx=inx, load_path=path, solver_name=name, **args)