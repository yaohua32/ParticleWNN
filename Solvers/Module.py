# /*
#  * @Author: yaohua.zang 
#  * @Date: 2023-09-26 15:47:24 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2023-09-26 15:47:24 
#  */
class Solver():

    def __init__(self):
        '''
        DNN-based PDE Solver
        '''

    def _save(self, save_path:str, model_type:str)->None:
        '''
        Save the model
        '''
        raise NotImplementedError

    def _load(self, load_path:str, model_type:str)->None:
        '''
        Load the model
        '''
        raise NotImplementedError

    def predict(self, load_path:str, model_type:str)->None:
        '''
        Prediction with the trained model
        '''
        raise NotImplementedError
    
    def get_net(self):
        '''
        Get the DNN structure
        '''
        raise NotImplementedError

    def get_loss(self):
        '''
        Get the loss 
        '''
        raise NotImplementedError

    def train(self)->None:
        '''
        Train the DNN model
        '''
        raise NotImplementedError