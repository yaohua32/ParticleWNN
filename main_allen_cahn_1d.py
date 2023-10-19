# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-29 17:00:07 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-29 17:00:07 
#  */
import numpy as np
import torch

class Example():

    def __init__(self, np_type, torch_type):
        '''
        ''' 
        self.np_type = np_type
        self.torch_type = torch_type

    def Solver_ParticleWNN(self, tloc_list:list, save_path:str, 
                           load_path:str, load_type:str, action:str, 
                           **args):
        '''
        '''
        try:
            lr = args['lr']
        except:
            lr = 1e-3 
        #
        Nt_slice = tloc_list[-1][1]
        ############ 
        kwargs = {'Nt_particle': 1000,
                  'Nx_particle': 5,
                  'Nt_bd': 200,
                  'Nx_bd_each_face': 1,
                  'Nx_init': 200,
                  'Nx_int': 25,
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  'maxIter': 50000,
                  'lr': lr,
                  'model_type': 'FeedForward_Sin',
                  'data_type': {'numpy':self.np_type, 'torch':self.torch_type},
                  'lr_Decay': 2.,
                  'loss_weight': {'eq':100., 'bd':5., 'init':50.},
                  'topK': 4000,
                  'int_method': 'mesh',
                  'hidden_width': 100,
                  'hidden_layer': 3,
                  'activation': 'tanh',
                  'tloc_list': tloc_list,
                  }
        #########
        from Solvers.PartivleWNN_Time import Solver
        problem = Problem(Nt_slice=Nt_slice, dtype=self.np_type, testFun_type='Wendland')
        solver = Solver(problem=problem, **kwargs)
        if action=='train':
            solver.train(save_path, load_path, load_type)
        else:
            solver.predict(load_path, load_type)

    def train(self, inx:int, tloc_list:list, save_path:str, load_path:str, 
              solver_name:str, **args):
        '''
        '''
        if solver_name=='ParticleWNN':
            demo.Solver_ParticleWNN(tloc_list=tloc_list,
                                    save_path=save_path+f'{solver_name}_{inx}/', 
                                    load_path=load_path+f'{solver_name}_{inx}/', 
                                    load_type='model_best_error', 
                                    action='train', **{'lr':args['lr']})

    def pred(self, inx:int, tloc_list:list, load_path:str, solver_name:str, **args):
        '''
        '''
        if solver_name=='ParticleWNN':
            demo.Solver_ParticleWNN(tloc_list=tloc_list,
                                    save_path=None, 
                                    load_path=load_path+f'{solver_name}_{inx}/', 
                                    load_type= 'model_best_error', 
                                    action='predict', **{'lr':args['lr']})
if __name__=='__main__':
    from Problems.Allen_Cahn_1d import Problem
    demo = Example(np_type=np.float64, torch_type=torch.float64)
    solver_names = ['ParticleWNN']
    path = f"./savedModel/{Problem().name}/"
    args = {'lr':1e-3}
    #
    tloc_list = [(0,101)]
    ###########################################
    for inx in range(5):
        for name in solver_names:
            print(f'************************ Method: {name} *************************************')
            demo.train(inx=inx, tloc_list=tloc_list, save_path=path, load_path=path, 
                       solver_name=name, **args)
            demo.pred(inx=inx, tloc_list=tloc_list, load_path=path, 
                      solver_name=name, **args)
