# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-27 12:54:35 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-27 12:54:35 
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
            dim = args['dim']
        except:
            dim = 5
        ############ 
        kwargs = {'N_particle': 80,
                  'N_bd_each_face': 100,
                  'N_int': 7,
                  'R_max': 1e-4,
                  'R_min': 1e-6,
                  'maxIter': 20001,
                  'lr': lr,
                  'model_type': 'FeedForward',
                  'data_type': {'numpy':self.np_type, 'torch':self.torch_type},
                  'lr_Decay': 2.,
                  'loss_weight': {'eq':1., 'bd':5.},
                  'topK': 80,
                  'int_method': 'mesh',
                  'hidden_width': 50,
                  'hidden_layer': 3,
                  'activation': 'tanh',
                  }
        #########
        from Solvers.ParticleWNN import Solver
        problem = Problem(dtype=self.np_type, testFun_type='Wendland', 
                          **{'dim':dim})
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

    def pred(self, inx:int, load_path:str, solver_name:str, 
             **args):
        '''
        '''
        if solver_name=='ParticleWNN':
            demo.Solver_ParticleWNN(save_path=None, 
                                    load_path=load_path+f'{solver_name}_{inx}/', 
                                    load_type= 'model_best_error', 
                                    action='predict', **args)

if __name__=='__main__':
    from Problems.Poisson_Nd import Problem
    demo = Example(np_type=np.float64, torch_type=torch.float64)
    #
    solver_names = ['ParticleWNN']
    path = f"./savedModel/{Problem().name}/"
    args = {'lr':1e-3, 'dim':5}
    # ###########################################
    for inx in range(5):
        for name in solver_names:
            print(f'************************ Method: {name} *************************************')
            demo.train(inx=inx, save_path=path, load_path=path, solver_name=name, **args)
            demo.pred(inx=inx, load_path=path, solver_name=name, **args)
