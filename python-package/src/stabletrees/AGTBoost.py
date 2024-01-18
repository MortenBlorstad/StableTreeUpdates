from _stabletrees import agtboost 
import numpy as np
class AGTBoost():

    def __init__(self,loss_function : str = "mse", nrounds:int = 5000, learning_rate:float = 0.01, gamma = 0.5) -> None:
        self.loss_function = loss_function
        self.nrounds = nrounds
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = agtboost()
        self.model.set_param(self.nrounds,self.learning_rate,0,self.loss_function,self.gamma)
    
    def inverse_function(self, pred:np.ndarray ):
        if  self.loss_function  == "mse":
            return pred
        if self.loss_function  =="poisson":
            return np.exp(pred)
        
        raise Exception("error")

    def fit(self,X : np.ndarray ,y : np.ndarray,verbose: int = 0, sample_weight: np.ndarray = None, offset: np.ndarray = None):
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        if offset is None:
            offset = np.zeros(shape=(len(y),))
        self.model.learn(y,X,verbose,False,False, sample_weight, offset)
        return 
    

    def update(self, X : np.ndarray ,y : np.ndarray,verbose: int = 0, sample_weight: np.ndarray = None, offset: np.ndarray = None):
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        if offset is None:
            offset = np.zeros(shape=(len(y),))
        prev_pred = self.predict(X,offset)
        self.model.update(y,prev_pred,X,verbose,False,False, sample_weight,offset)
        return 
    
    def predict(self,X : np.ndarray,offset: np.ndarray = None):
        if offset is None:
            offset = np.zeros(shape=(X.shape[0],))
        return self.inverse_function(self.model.predict(X,offset))
    