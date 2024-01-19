
from _stabletrees import Node, Tree
from _stabletrees import AbuTree as atree
from _stabletrees import StableLossTree as SLTree


from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import BaseEstimator 
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
import numpy as np

criterions = {"mse":0, "poisson":1}



class BaseRegressionTree(BaseEstimator, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                  max_features:int = None, random_state : int = None) -> None:
        criterion = str(criterion).lower()
        if criterion not in criterions.keys():
            raise ValueError("Possible criterions are 'mse' and 'poisson'.")
        self.criterion = criterion

        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)
        if max_features is None:
            max_features = 2147483647
        self.max_features = int(max_features)

        self.min_samples_split = float(min_samples_split)

        if random_state is None:
            random_state = 0
        self.random_state = int(random_state)

        self.adaptive_complexity = adaptive_complexity
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = 1


    def check_input(self,  X : np.ndarray ,y : np.ndarray):
        if X.ndim <2:
            X = np.atleast_2d(X)
        if np.issubdtype(X.dtype, np.number):
            X = X.astype("double")
        else:
            raise ValueError("X needs to be numeric")
        
        if y.ndim >1:
            raise ValueError("y needs to be 1-d")

        if np.issubdtype(y.dtype, np.number):
            y = y.astype("double")
        else:
            raise ValueError("y needs to be numeric")
        return X,y

    @abstractmethod
    def update(self,X : np.ndarray ,y : np.ndarray,sample_weight: np.ndarray = None):
        pass

    def fit(self,X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None): 
        X,y = self.check_input(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.tree.learn(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict(X)
    
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        return self.tree.predict_uncertainty(X)

    def plot(self):
        '''
        plots the tree. A visualisation of the tree
        '''
        plt.rcParams["figure.figsize"] = (20,10)
        self.__plot(self.root)
        plt.plot(0, 0, alpha=1) 
        plt.axis("off")
        

    def __plot(self,node: Node,x=0,y=-10,off_x = 100000,off_y = 15):
        '''
        a helper method to plot the tree. 
        '''
        

        # No child.
        if node is None:
            return

        if node.is_leaf():
            plt.plot(x+10, y-5, alpha=1) 
            plt.plot(x-10, y-5, alpha=1) 
            plt.text(x, y,f"{node.predict():.2f}", fontsize=8,ha='center') 
            plt.text(x, y-2,f"{node.nsamples():.2f}", fontsize=8,ha='center') 
            return
        
        
    
        x_left, y_left = x-off_x,y-off_y
        plt.text(x, y,f"X_{node.get_split_feature()}<={node.get_split_value():.4f}\n", fontsize=8,ha='center')
        plt.text(x, y-2,f"impurity: {node.get_impurity():.3f}", fontsize=8,ha='center')
        plt.text(x, y-4,f"nsamples: {node.get_features_indices()}", fontsize=8,ha='center')
        plt.annotate("", xy=(x_left, y_left+4), xytext=(x-2, y-4),
        arrowprops=dict(arrowstyle="->"))

        x_right, y_right = x+off_x,y-off_y
        plt.annotate("", xy=(x_right , y_right+4), xytext=(x+2, y-4),
        arrowprops=dict(arrowstyle="->"))
        self.__plot(node.get_left_node(),x_left, y_left, off_x*0.5)
        self.__plot(node.get_right_node() ,x_right, y_right,off_x*0.5)


class BaseLineTree(BaseRegressionTree):
    """
        Baseline: update method - same as the fit method. 
        Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.
    min_samples_leaf : int,  default = 5.
                Hyperparameter to determine the minimum number of samples required in a leaf node.
    adaptive_complexity : bool,  default = False.
                Hyperparameter to determine wheter find the tree complexity adaptively.
    max_features : bool,  default = None.
                The number of features to consider when looking for the best split.
    random_state : bool,  default = None.
                Controls the randomness of the tree.
    """

    def __init__(self, *,criterion : str = "mse", max_depth : int = None, min_samples_split : int = 2,min_samples_leaf:int = 5,
                    adaptive_complexity : bool = False, max_features:int = None, random_state : int = None) -> None:
        
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf, adaptive_complexity,max_features, random_state)
        self.tree = Tree(criterions[self.criterion], self.max_depth,self.min_samples_split,self.min_samples_leaf, self.adaptive_complexity,self.max_features,self.learning_rate,self.random_state)
    
    def update(self,X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None):
        return self.fit(X,y,sample_weight)
    # def update(self,X : np.ndarray ,y : np.ndarray):
    #     X,y = self.check_input(X,y)
    #     self.tree.update(X,y)
    #     self.root = self.tree.get_root()
    #     return self
    
    
    def fit_difference(self, X : np.ndarray ,y : np.ndarray,y_pred : np.ndarray ):
        X,y = self.check_input(X,y)
        self.tree.learn_difference(X,y,y_pred)
        self.root = self.tree.get_root()
        return self
    
    
class SklearnTree(DecisionTreeRegressor):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.
    min_samples_leaf : int,  default = 5.
                Hyperparameter to determine the minimum number of samples required in a leaf node.
    adaptive_complexity : bool,  default = False.
                Hyperparameter to determine wheter find the tree complexity adaptively.
    max_features : bool,  default = None.
                The number of features to consider when looking for the best split.
    random_state : bool,  default = None.
                Controls the randomness of the tree.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 1, adaptive_complexity : bool = False, random_state = None):
        
        super().__init__(criterion = criterion,max_depth= max_depth, min_samples_split= min_samples_split,min_samples_leaf = min_samples_leaf, random_state = random_state)

    def update(self, X,y):
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.fit(X,y)
        return self
    
  
class StableLossTree(BaseRegressionTree):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.
    min_samples_leaf : int,  default = 5.
                Hyperparameter to determine the minimum number of samples required in a leaf node.
    adaptive_complexity : bool,  default = False.
                Hyperparameter to determine wheter find the tree complexity adaptively.
    max_features : bool,  default = None.
                The number of features to consider when looking for the best split.
    random_state : int,  default = None.
                Controls the randomness of the tree.
    gamma : float,  default = 0.5.
                Determines the strength of the stability regularization.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                 max_features:int = None, random_state = None, gamma :float= 0.5):
        self.root = None
        self.gamma = gamma
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity,max_features,random_state)
        self.tree = SLTree(self.gamma, criterions[self.criterion], self.max_depth,self.min_samples_split,self.min_samples_leaf, self.adaptive_complexity,self.max_features,self.learning_rate,self.random_state)
    
    def update(self, X,y,sample_weight=None):
        X,y = self.check_input(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.tree.update(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self  



class AbuTree(BaseRegressionTree):
    """
    A regression tree that uses stability regularization when updating the tree. Method 2: update method build a new tree using the prediction from the previous tree as regularization.
    
    Parameters
    ----------
    criterion : string, {'mse', 'poisson'}, default = 'mse'
                Function to optimize when selecting split feature and value.
    max_depth : int, default = None.
                Hyperparameter to determine the max depth of the tree.
                If None, then nodes are expanded until all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    min_samples_split : int,  default = 2.
                Hyperparameter to determine the minimum number of samples required in order to split a internel node.
    min_samples_leaf : int,  default = 5.
                Hyperparameter to determine the minimum number of samples required in a leaf node.
    adaptive_complexity : bool,  default = False.
                Hyperparameter to determine wheter find the tree complexity adaptively.
    max_features : bool,  default = None.
                The number of features to consider when looking for the best split.
    random_state : int,  default = None.
                Controls the randomness of the tree.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                 max_features:int = None, random_state = 0, alpha:float = 0, beta:float = 1):
        
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity,max_features,random_state)
        self.alpha = alpha
        self.beta = beta
        self.tree = atree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,self.random_state,self.alpha,self.beta)
    
    def get_params(self, deep=True):
        # Get parameters for this estimator.
        return {"criterion": self.criterion,"max_depth": self.max_depth,"min_samples_split": self.min_samples_split, "min_samples_leaf": self.min_samples_leaf, 
                  "adaptive_complexity":self.adaptive_complexity,"max_features": self.max_features, "random_state": self.random_state, "alpha": self.alpha, "beta": self.beta}
    
    def set_params(self, **parameters):
        # Set the parameters of this estimator.
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    
    def predict_info(self, X):
        return self.tree.predict_info(X)
    
    def predict(self, X):
        return self.tree.predict(X)

    def update(self, X,y,sample_weight=None):
        X,y = self.check_input(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.tree.update(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self  
