
from _stabletrees import Node, Tree, NewTree
from _stabletrees import AbuTree as atree
from _stabletrees import AbuTree2 as atree2
from _stabletrees import NaiveUpdate as NuTree
from _stabletrees import StabilityRegularization as SrTree
from _stabletrees import TreeReevaluation as TrTree
from _stabletrees import STTree as sttree



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
    

class NaiveUpdate(BaseRegressionTree):
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
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                 max_features:int = None, random_state = None):
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity, max_features)
        self.tree = NuTree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,self.random_state)
    
    def update(self, X,y, sample_weight: np.ndarray = None):
        X,y = self.check_input(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.tree.update(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self 
    

    
class TreeReevaluation(BaseRegressionTree):
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
    delta : float,  default = 0.1.
                Determines the confidence level, 1-delta.
    alpha : float,  default = None.
                Determines the minimum improvements of replace update subtree.

    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 2,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                 max_features:int = None, random_state = None,delta : float=0.1, alpha:float = 0.05):
        self.root = None
        self.delta = delta
        self.alpha = alpha
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity,max_features)
        self.tree = TrTree(self.alpha,self.delta ,criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,self.random_state)
        
    
    
    def update(self, X,y,sample_weight: np.ndarray = None):
        X,y = self.check_input(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.tree.update(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self  
    
class StabilityRegularization(BaseRegressionTree):
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
        self.tree = SrTree(self.gamma, criterions[self.criterion], self.max_depth,self.min_samples_split,self.min_samples_leaf, self.adaptive_complexity,self.max_features,self.learning_rate,self.random_state)
    
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
                 max_features:int = None, random_state = 0):
        
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity,max_features,random_state)
        self.tree = atree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,self.random_state)
    
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



class AbuTree2(BaseRegressionTree):
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
        self.tree = atree2(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,self.random_state,self.alpha,self.beta)
    
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

methods = {"baseline":BaseLineTree, "NU":NaiveUpdate , "TR":TreeReevaluation, "SR":StabilityRegularization, "ABU":AbuTree}


class StableTree(BaseRegressionTree):
    def __init__(self, method:str = "baseline", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None) -> None:
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity)
        

class BABUTree(BaseRegressionTree):
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
    bumping_iterations : int,  default = 5.
                Determines the number of bumping interations.
    """
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 5,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                 max_features:int = None, random_state = 0, bumping_iterations:int = 5):
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity,max_features,random_state)
        self.tree =atree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,self.random_state)
        self.bumping_iterations = bumping_iterations


    def _fit(self, X: np.ndarray, y: np.ndarray,sample_weight):
        X,y = self.check_input(X,y)
        self.tree.learn(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self
    
    def _update(self, X: np.ndarray, y: np.ndarray,sample_weight):
        X,y = self.check_input(X,y)
        self.tree.update(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self  
        

    def fit(self, X: np.ndarray, y: np.ndarray,sample_weight=None):
        np.random.seed(0)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self._fit(X,y,sample_weight)
        n = X.shape[0]
        # X_ = X
        # y_ = y
        for b in range(self.bumping_iterations):
            # ind_b = np.random.randint(0,n,size=n)
            # # X_b = X[ind_b,:]
            # y_b = y[ind_b]
            # X_ = np.vstack((X_,X_b))
            # y_ = np.concatenate((y_,y_b),axis=0)
            self._update(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self

    def update(self, X,y,sample_weight=None): 
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self._update(X,y,sample_weight)
        return self
 
from sklearn.model_selection import KFold

class BABUTreeI(BaseRegressionTree):
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
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 5,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                 max_features:int = None, random_state = None, bumping_iterations:int = 5):
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity,random_state,max_features)
        self.tree = atree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,0)
        self.bumping_iterations = bumping_iterations

    def _fit(self, X: np.ndarray, y: np.ndarray):
       
        X,y = self.check_input(X,y)
        self.tree.learn(X,y)
        self.root = self.tree.get_root()
        return self
    
    def _update(self, X: np.ndarray, y: np.ndarray):
        X,y = self.check_input(X,y)
        self.tree.update(X,y)
        self.root = self.tree.get_root()
        return self  
        

    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(0)
        self._fit(X,y)
        n = X.shape[0]
        kf = KFold(n_splits=self.bumping_iterations)
        # X_ = X
        # y_ = y
        for train_index, test_index in kf.split(X):   
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self._update(X_train,y_train)
        self._update(X,y)
        return self

    # def update(self, X,y): 
    #     kf = KFold(n_splits=self.bumping_iterations)
    #     # X_ = X
    #     # y_ = y
    #     for train_index, test_index in kf.split(X):   
    #         X_train, X_test = X[train_index], X[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         self._update(X_train,y_train)
    #     self._update(X,y)
    def update(self, X,y): 
        self._update(X,y)
        return self



class STTree(BaseRegressionTree):
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
    
    def __init__(self, *,criterion = "mse", max_depth = None, min_samples_split = 5,min_samples_leaf:int = 5, adaptive_complexity : bool = False,
                 max_features:int = None, random_state = None, bumping_iterations:int = 5):
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf,adaptive_complexity,random_state,max_features)
        self.tree = sttree(criterions[self.criterion], self.max_depth, self.min_samples_split,self.min_samples_leaf,adaptive_complexity,self.max_features,self.learning_rate,0)
        self.bumping_iterations = bumping_iterations



    # def fit(self, X: np.ndarray, y: np.ndarray):
    #     np.random.seed(0)
    #     self._fit(X,y)
    #     n = X.shape[0]
    #     # X_ = X
    #     # y_ = y
    #     for b in range(self.bumping_iterations):
    #         # ind_b = np.random.randint(0,n,size=n)
    #         # # X_b = X[ind_b,:]
    #         # y_b = y[ind_b]
    #         # X_ = np.vstack((X_,X_b))
    #         # y_ = np.concatenate((y_,y_b),axis=0)
    #         self._update(X,y)
    #     return self
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(0)
        self.tree.learn(X,y)
        self.root = self.tree.get_root()
        n = X.shape[0]
        ind = np.random.randint(low=0, high = n, size = 10*n)
        X_u = X[ind]
        print(X_u.shape)
        increase = len(ind)//10
        n_tilde = increase
        X_tilde = X_u[0:n_tilde]
        np.random.normal(0, np.std(X, axis=0),10*n)
        for b in range(10):  

            self.tree.update(X,y,X_tilde)
   
            self.root = self.tree.get_root()
            n_tilde += increase
            X_tilde = X_u[0:n_tilde]
        
        self.root = self.tree.get_root()
        return self

    def _fit(self, X: np.ndarray, y: np.ndarray):
        X,y = self.check_input(X,y)
        self.tree.learn(X,y)
        self.root = self.tree.get_root()
        return self
    
    def _update(self, X: np.ndarray, y: np.ndarray):
        X,y = self.check_input(X,y)
        self.tree.update(X,y,X)
        self.root = self.tree.get_root()
        return self  
        

    

    def update(self, X,y): 
        self._update(X,y)
        return self



if __name__ =="__main__":
    np.random.seed(0)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    n = 200

    X = np.random.uniform(size=(n,1),low=0,high=4)
    y = np.random.normal(loc=X.ravel(),scale=1,size = n)
    X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)
    X_test = np.random.uniform(size=(n,1),low=0,high=4)
    y_test = np.random.normal(loc=X_test.ravel(),scale=1,size = n)

    t1 = BaseLineTree(adaptive_complexity=True).fit(X1,y1)
    t2 = AbuTree(adaptive_complexity=True).fit(X1,y1)
    t = STTree(adaptive_complexity=True).fit(X1,y1)
    t1_pred1 = t1.predict(X_test)
    t2_pred1 = t2.predict(X_test)
    t_pred1 = t.predict(X_test)


    print("initial")
    print(mean_squared_error(y_test,t1_pred1))
    print(mean_squared_error(y_test,t2_pred1))
    print(mean_squared_error(y_test,t_pred1))
    t.update(X,y)
    t1.update(X,y)
    t2.update(X,y)
    t1_pred2 = t1.predict(X_test)
    t2_pred2 = t2.predict(X_test)
    t_pred2 = t.predict(X_test)
    print("update")
    print(mean_squared_error(y_test,t1_pred2))
    print(mean_squared_error(y_test,t2_pred2))
    print("t: ", mean_squared_error(y_test,t_pred2))

    def S2(pred1, pred2):
        return np.mean((pred1- pred2)**2)

    print("stability")
    print("base ",S2(t1_pred1,t1_pred2))
    print("AbuTree: ",S2(t2_pred1,t2_pred2))
    print("t: ",S2(t_pred1,t_pred2))
