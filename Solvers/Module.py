# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-05-21 20:06:13 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-05-21 20:06:13 
#  */
class Solver():

    def __init__(self):
        #
        pass

    def _save(self, save_path:str, model_type:str)->None:
        '''
        '''
        raise NotImplementedError

    def _load(self, save_path:str, model_type:str)->None:
        '''
        '''
        raise NotImplementedError

    def test(self)->None:
        '''
        '''
        raise NotImplementedError
    
    def get_net(self):
        '''
        '''
        raise NotImplementedError

    def get_loss(self):
        '''
        '''
        raise NotImplementedError

    def train(self)->None:
        '''
        '''
        raise NotImplementedError