
from stabletrees.tree import BaseRegressionTree,BaseLineTree,NaiveUpdate,AbuTree,TreeReevaluation,StabilityRegularization,BABUTree
from _stabletrees import RandomForest as rf
from _stabletrees import RandomForestSL as rfsl
from _stabletrees import RandomForestNU as rfnu
from _stabletrees import RandomForestTR as rftr
from _stabletrees import RandomForestABU as rfabu
from _stabletrees import RandomForestBABU as rfbabu



import numpy as np

max_features_to_int = {"all": lambda x :x,
                       "third": lambda x :max(1, int(x/3)),
                       None : lambda x :x}

method_to_int = {"base":0,
                 "nu":1,
                 "tr":2,
                 "sl":3,
                 "abu":4,
                 "babu":5}
criterions = {"mse":0, "poisson":1}


class RF(BaseRegressionTree):
    """
        Baseline: update method - same as the fit method. 
        Parameters
    ----------
    method : str, {'base', 'nu', 'tr', 'sl', 'abu', 'babu'}, default = 'base'.
                The update method.
    n_estimators : int, default = 100.
                Number of trees in the forest.
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
    delta : float,  default = 0.1.
                Determines the confidence level, 1-delta (only applicable if method = 'tr').
    alpha : float,  default = None.
                Determines the minimum improvements of replace update subtree (only applicable if method = 'tr').
    gamma : float,  default = 0.5.
                Determines the strength of the stability regularization (only applicable if method = 'sl').
    bumping_iterations : int,  default = 5.
                Determines the number of bumping interations (only applicable if method = 'babu').
    """
    def __init__(self,method:str = "base",n_estimators:int = 100,max_features:str = "all", criterion: str = "mse", max_depth: int = None,
                  min_samples_split: int = 2, min_samples_leaf: int = 5, adaptive_complexity: bool = False, random_state: int = None, gamma:float = 0.5, delta:float = 0.05,alpha:float = 0.0, bumping_iterations =5) -> None:
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.bumping_iterations = bumping_iterations
        assert n_estimators>=1
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.method = method
        self.criterion =criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        if self.max_features not in max_features_to_int.keys():
            self.max_features = "all"

        if max_depth is None:
            max_depth = 2147483647
        self.max_depth = int(max_depth)
        self.forest = None


    
        

    def fit(self,X,y,sample_weight=None):
        max_feature = max_features_to_int[self.max_features](X.shape[1])
        #print(method_to_int[self.method])
        if method_to_int[self.method] ==0:
            self.forest = rf(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature)
        if method_to_int[self.method] ==1:
            self.forest = rfnu(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature)
        if method_to_int[self.method] ==2:
            self.forest = rftr(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature,self.delta,self.alpha)
        if method_to_int[self.method] ==3:
            self.forest = rfsl(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature,self.gamma)
        if method_to_int[self.method] ==4:
            self.forest = rfabu(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature)
        if method_to_int[self.method] ==5:
            self.forest = rfbabu(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,max_feature,self.bumping_iterations)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.forest.learn(X,y,sample_weight)
        return self
        
    def update(self, X: np.ndarray, y: np.ndarray,sample_weight=None):
        
        if self.forest is None:
            return self.fit(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.forest.update(X,y,sample_weight)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert (self.forest is not None)
        return self.forest.predict(X)
        


