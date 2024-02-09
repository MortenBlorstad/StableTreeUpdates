
from _stabletrees import Node
from _stabletrees import Tree as tree
# from _stabletrees import AbuTree as atree



from abc import ABCMeta
from abc import abstractmethod
from sklearn.base import BaseEstimator 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

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
        if X.ndim == 1:
            X = np.atleast_2d(X).T
        elif X.ndim != 2:
            raise ValueError("X needs to be 2-dimensional (num_obs, num_features)")
    
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


class Tree(BaseRegressionTree):
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
                    adaptive_complexity : bool = False, max_features:int = None, random_state : int = 0, alpha:float = 0, beta:float = 1) -> None:
        self.root = None
        super().__init__(criterion,max_depth, min_samples_split,min_samples_leaf, adaptive_complexity,max_features, random_state)
        self.alpha = alpha
        self.beta = beta
        self.tree = tree(criterions[self.criterion], self.max_depth,self.min_samples_split,self.min_samples_leaf, self.adaptive_complexity,self.max_features,self.learning_rate,self.random_state, self.alpha, self.beta)
    
    
    # def fit(self,X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None,iter:int = 1): 
    #     np.random.seed(self.random_state)
    #     X,y = self.check_input(X,y)
    #     if sample_weight is None:
    #         sample_weight = np.ones(shape=(len(y),))

    #     if iter <=1:
    #         self.tree.learn(X,y,sample_weight)
    #         self.root = self.tree.get_root()
    #         return self
        
    #     start_size = int(len(y)*0.9)
    #     update_size= (len(y)-start_size)
    #     n_fold = (update_size)//iter
    #     b_size =start_size
    #     # Generate bootstrap indices
    #     #bootstrap_indices = np.random.choice(len(y), size=len(y), replace=False)
    #     for i in range(iter-1):
    #         # Select the bootstrap samples
    #         #b_ind = bootstrap_indices[:b_size]#np.random.randint(0,len(y),size=len(y))
    #         X_b = X[:b_size]
    #         y_b = y[:b_size]
    #         sample_weight_b = sample_weight[:b_size]
    #         if i ==0:
    #             self.tree.learn(X_b, y_b, sample_weight_b)
    #             self.root = self.tree.get_root()
    #         else:
    #             self.tree.update(X_b, y_b, sample_weight_b)
    #             self.root = self.tree.get_root()
    #         #print(b_size,len(y),n_fold,start_size,update_size)
    #         b_size+=n_fold 
            
    #     self.tree.update(X, y, sample_weight)
    #     self.root = self.tree.get_root()
    #     return self
        

    def fit(self,X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None,iter:int = 1): 
        np.random.seed(self.random_state)
        X,y = self.check_input(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))

        if iter <=1:
            self.tree.learn(X,y,sample_weight)
            self.root = self.tree.get_root()
            return self
        
        n_fold = (len(y))//iter
        b_size =n_fold
        # Generate bootstrap indices
        #bootstrap_indices = np.random.choice(len(y), size=len(y), replace=False)
        for i in range(iter-1):
            # Select the bootstrap samples
            #b_ind = bootstrap_indices[:b_size]#np.random.randint(0,len(y),size=len(y))
            X_b = X[:b_size]
            y_b = y[:b_size]
            sample_weight_b = sample_weight[:b_size]
            if i ==0:
                self.tree.learn(X_b, y_b, sample_weight_b)
                self.root = self.tree.get_root()
            else:
                self.tree.update(X_b, y_b, sample_weight_b)
                self.root = self.tree.get_root()
            #print(b_size,len(y),n_fold,start_size,update_size)
            b_size+=n_fold 
            
        self.tree.update(X, y, sample_weight)
        self.root = self.tree.get_root()
        return self
    
    def predict_info(self,X : np.ndarray ):
        return self.tree.predict_info(X)



    def update(self, X : np.ndarray ,y : np.ndarray, sample_weight: np.ndarray = None):
        X,y = self.check_input(X,y)
        if sample_weight is None:
            sample_weight = np.ones(shape=(len(y),))
        self.tree.update(X,y,sample_weight)
        self.root = self.tree.get_root()
        return self  

if __name__ == "__main__":
    from sklearn.decomposition import PCA

    np.random.seed(0)
    noise = 5
    def p1(X):
        return 5*X[:,0]**2 #+0.15*(X[:,0]*X[:,1])  + (X[:,1])
    X_test = np.random.uniform(-4,4,(10000,1))
    y_test = np.random.normal(p1(X_test),scale=noise)
    pca = PCA(n_components=1)
    pca = pca.fit(X_test)

    repeats= 20
    predictions_iter1 = np.zeros((10000,repeats))
    predictions_iter2 = np.zeros((10000,repeats))



    X = np.random.uniform(-4,4,(250,1))
    y = np.random.normal(p1(X),scale=noise)
    mse_iter1 = 0
    mse_iter2 = 0
    stab_iter1 = 0
    stab_iter2 = 0
    t1 = Tree(adaptive_complexity=True,beta=0, alpha=0,random_state=0)
    t2=  Tree(adaptive_complexity=True,beta=0.8, alpha=0.2,random_state=0)

    t1.fit(X,y)
    t2.fit(X,y,iter=20) #40
    print("sadad")

        
    pred1 = t1.predict(X_test)
    pred2 = t2.predict(X_test)

    for i in range(repeats):
        np.random.seed(i)
        x = np.random.uniform(-4,4,(1,1))
        X_ = np.vstack((X, x))
        y_ = np.append(y, np.random.normal(p1(x),scale=noise))  



        t1.fit(X_,y_)
        t2.fit(X_,y_,iter=20) #40

        
        pred1_update = t1.predict(X_test)
        pred2_update = t2.predict(X_test)

        predictions_iter1[:,i]=pred1_update
        predictions_iter2[:,i]=pred2_update
        
        mse_iter1 +=np.mean((pred1_update-y_test)**2)
        mse_iter2 +=np.mean((pred2_update-y_test)**2)


        stab1 = np.mean((pred1-pred1_update)**2)
        stab2 = np.mean((pred2-pred2_update)**2)

        stab_iter1 +=stab1
        stab_iter2 +=stab2




    sd_iter1 = np.std(predictions_iter1, axis=1)
    mean_iter1 = np.mean(predictions_iter1, axis=1)

    sd_iter2 = np.std(predictions_iter2, axis=1)
    mean_iter2 = np.mean(predictions_iter2, axis=1)

    print(f"iter 1: loss  {mse_iter1/repeats:.3f}, stability {stab_iter1/repeats:.6f}, prediction deviation {np.mean(sd_iter1):.6f}" )
    print(f"iter 2: loss  {mse_iter2/repeats:.3f}, stability {stab_iter2/repeats:.6f}, prediction deviation {np.mean(sd_iter2):.6f}" )

    fig, (ax1,ax2) = plt.subplots(nrows=1,ncols=2)

    PCA_comp = pca.transform(X_test).ravel()
    ind = np.argsort(PCA_comp, axis=0)


    ax1.scatter(PCA_comp[ind],y_test[ind], c="red", s = 1)
    ax2.scatter(PCA_comp[ind],y_test[ind], c="red", s = 1)
    # ax1.scatter(X,y, c="blue", s = 1)
    # ax2.scatter(X,y, c="blue", s = 1)
    #ax1.plot(X_test,mean_iter1, c = "green")
    ax1.errorbar(PCA_comp[ind], mean_iter1[ind], yerr=sd_iter1[ind],c="blue", alpha=0.5, ecolor='lightblue', elinewidth=5, capsize=0, label='Standard Deviation')
    ax1.set_title(f"Baseline:\nloss  {mse_iter1/repeats:.3f}; \nstability {stab_iter1/repeats:.3f}; \nprediction deviation {np.mean(sd_iter1):.3f}")
    #ax2.plot(X_test,pred2, c ="black")
    ax2.set_title(f"Baseline with 20 iter:\nloss  {mse_iter2/repeats:.3f}; \nstability {stab_iter2/repeats:.3f}; \nprediction deviation {np.mean(sd_iter2):.3f}")
    ax2.errorbar(PCA_comp[ind], mean_iter2[ind], yerr=sd_iter2[ind],c="black", alpha=0.5, ecolor='lightgrey', elinewidth=5, capsize=0, label='Standard Deviation')

    plt.show()
