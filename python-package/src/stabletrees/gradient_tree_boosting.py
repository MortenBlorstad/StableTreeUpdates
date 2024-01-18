from stabletrees.tree import BaseRegressionTree,BaseLineTree,NaiveUpdate,AbuTree,TreeReevaluation,criterions
from _stabletrees import GBT
import numpy as np



class GradientBoosting(BaseRegressionTree):
    def __init__(self,n_estimators:int,max_features:str = "sqrt", criterion: str = "mse", max_depth: int = None, min_samples_split: int = 5, min_samples_leaf: int = 5,
                adaptive_complexity: bool = False,learning_rate: float = 0.1, random_state: int = None) -> None:
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf, adaptive_complexity, random_state)
        assert n_estimators>=1
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.ensemble = GBT(criterions[self.criterion],self.n_estimators,self.max_depth,self.min_samples_split,self.min_samples_leaf,self.adaptive_complexity,self.learning_rate)
       


    def fit(self,X,y):
        self.ensemble.learn(X,y)
        return self
        
    def update(self, X: np.ndarray, y: np.ndarray, gamma:float = 0.5):
        self.ensemble.update(X,y,gamma)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.ensemble.predict(X)
    
if __name__ =="__main__":
    def S1(pred1, pred2):
        return np.std(np.log((pred2+1.1)/(pred1+1.1)))#np.mean((pred1- pred2)**2)#

    def S2(pred1, pred2):
        return np.mean(abs(pred1- pred2))
    np.random.seed(0)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    n = 2000

    X = np.random.uniform(size=(n,1),low=0,high=4)
    y = np.random.normal(loc=X.ravel()**2,scale=1,size = n)
    #y = np.random.poisson(lam=np.exp(X.ravel()),size = n)
    X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)
    X_test = np.random.uniform(size=(n,1),low=0,high=4)
    y_test = np.random.normal(loc=X_test.ravel()**2,scale=1,size = n)
    #y_test = np.random.poisson(lam=np.exp(X_test.ravel()),size = n)

    criterion = "mse"
    gtb = GradientBoosting(1000  , criterion=criterion,adaptive_complexity=True, learning_rate=0.01).fit(X1,y1)
    print("fitted")
    gtb_pred1 = gtb.predict(X)
    print("predicted")
    print("initial")
    

    from matplotlib import pyplot as plt
    plt.subplot(3,2,1)
    
    plt.scatter(X[:,0],y, alpha = 0.1)
    t = AbuTree(criterion=criterion, adaptive_complexity=True, max_depth=0).fit(X1,y1)
    t_pred1 = t.predict(X)
    plt.title(f"ABU (D1)")
    print(mean_squared_error(y_test,t.predict(X_test)))
    print(mean_squared_error(y_test,gtb.predict(X_test)))
    plt.scatter(X[:,0],t_pred1,c ="red", alpha = 0.5, label = f"mse: {mean_squared_error(y_test,t.predict(X_test)):.3f}")
    plt.legend()
    plt.subplot(3,2,2)
    
    plt.scatter(X[:,0],y, alpha = 0.1)
    t =t.update(X,y)
    t_pred2 = t.predict(X)
    plt.title(f"ABU (D2) - stability: {S2(gtb_pred1,t_pred2):.3f}")
 
    print(mean_squared_error(y_test,t.predict(X_test)))
    plt.scatter(X[:,0],t_pred2,c ="red", alpha = 0.5, label = f"mse: {mean_squared_error(y_test,t.predict(X_test)):.3f}")
    plt.legend()
    #---
    plt.subplot(3,2,3)

    plt.scatter(X[:,0],y, alpha = 0.1)
    plt.title(f"GTB (D1)")
    plt.scatter(X[:,0],gtb_pred1[:],c ="red", alpha = 0.5, label = f"mse: {mean_squared_error(y_test,gtb.predict(X_test)):.3f}")
    plt.legend()

    plt.subplot(3,2,4)

    gtb1 = GradientBoosting(1000  , criterion=criterion,adaptive_complexity=True, learning_rate=0.01).fit(X,y)
    gtb_pred12 = gtb1.predict(X)
    print(mean_squared_error(y_test,gtb.predict(X_test)))
    print(S1(t_pred1,t_pred2))
    print(S1(gtb_pred1,gtb_pred12))
    plt.scatter(X[:,0],y, alpha = 0.1)
    plt.title(f"GTB retrained (D2) - stability: {S1(gtb_pred1,gtb_pred12):.3f}")
    plt.scatter(X[:,0],gtb_pred12[:],c ="red", alpha = 0.5, label = f"mse: {mean_squared_error(y_test,gtb1.predict(X_test)):.3f}")
    plt.legend()
    #---

    plt.subplot(3,2,5)

    plt.scatter(X[:,0],y, alpha = 0.1)
    plt.title(f"GTB (D1)")
    plt.scatter(X[:,0],gtb_pred1[:],c ="red", alpha = 0.5, label=f'mse: {mean_squared_error(y_test,gtb.predict(X_test)):.3f}')
    plt.legend()
    plt.subplot(3,2,6)

    gtb.update(X,y,0.75)
    gtb_pred2 = gtb.predict(X)
    print(mean_squared_error(y_test,gtb.predict(X_test)))
    plt.scatter(X[:,0],y, alpha = 0.1)
    plt.title(f"GTB updated (D2) - stability: {S1(gtb_pred1,gtb_pred2):.3f}")
    plt.scatter(X[:,0],gtb_pred2[:],c ="red", alpha = 0.5, label=f'mse: {mean_squared_error(y_test,gtb.predict(X_test)):.3f}')
    print(S1(t_pred1,t_pred2))
    print(S1(gtb_pred1,gtb_pred12))
    print(S1(gtb_pred1,gtb_pred2))
    plt.legend()
    plt.show()
