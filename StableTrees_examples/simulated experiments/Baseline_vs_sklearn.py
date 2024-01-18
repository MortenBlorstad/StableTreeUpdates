
from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.linear_model import PoissonRegressor,LinearRegression
from adjustText import adjust_text

SEED = 0
EPSILON = 1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)

def p1(X):
    return X[:,0]
def p2(X):
    return X[:,0]**2 + 0.75*X[:,1]
def p3(X):
    return X[:,0]**2 + 0.75*X[:,1] -  0.25*X[:,3] + 0.1*X[:,0]*X[:,2] - 0.05*X[:,1]*X[:,3]

cases = {"Case 1": {"features": [0], "p":p1},
         "Case 2": {"features": [0,3], "p":p2},
          "Case 3": {"features": [0,1,2,3], "p":p3}}



np.random.seed(SEED)
n = 1000
X1 = np.random.uniform(0,4,size=(n,1))
X2 = np.random.uniform(0,4,size=(n,1))
X3 = np.round(np.random.uniform(0,1,size=(n,1)),decimals=0)
X4 = np.round(np.random.uniform(0,5,size=(n,1)),decimals=0)

X = np.hstack((X1,X2,X3,X4))


# fixed
for (case, info), y_var in zip(cases.items(),[1,3,5]):
        np.random.seed(SEED)
        features = info["features"]
        p = info["p"]
        x = X[:,features]
        y = np.random.normal(loc = p(X), scale=y_var,size=n)
        kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
        criterion = "mse"
        models = {  
                        "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, max_depth=5),
                        "sklearn": DecisionTreeRegressor(random_state=0,min_samples_leaf=5,max_depth=5)
                        }
        stability = {k :[] for k in models.keys()}
        performance = {k :[] for k in models.keys()}
        for i,(train_index, test_index) in enumerate(kf.split(x)):
                
                X_12, y_12 = x[train_index],y[train_index]
                X_test,y_test = x[test_index],y[test_index]
                X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
                for name, model in models.items():
                        model.fit(X1,y1)
                        # if name =="glm" or name =="sklearn":
                        #     print(model.best_params_)
                        pred1 = model.predict(X_test)
                        if name =="glm" or name =="sklearn":
                                # print(model.best_params_)
                                model.fit(X_12,y_12)
                        else:
                                model.update(X_12,y_12)
                        pred2 = model.predict(X_test)
                        stability[name].append( S2(pred1,pred2))
                        performance[name].append(mean_squared_error(y_test, pred2))
        print(case)
        for name in models.keys():
                print("="*80)
                print(f"{name}")

                mse_scale = np.mean(performance["baseline"]); S_scale = np.mean(stability["baseline"]);
                loss_score = np.mean(performance[name])
                loss_SE = np.std(performance[name])/np.sqrt(len(performance[name]))
                loss_SE_norm = np.std(performance[name]/mse_scale)/np.sqrt(len(performance[name]))
                stability_score = np.mean(stability[name])
                stability_SE = np.std(stability[name])/np.sqrt(len(performance[name]))
                stability_SE_norm = np.std(stability[name]/S_scale)/np.sqrt(len(performance[name]))
                print(f"test - mse: {loss_score:.3f} ({loss_SE:.2f}), stability: {stability_score:.3f} ({stability_SE:.2f})")
                print(f"test - mse: {loss_score/mse_scale:.2f} ({loss_SE_norm:.2f}), stability: {stability_score/S_scale:.2f} ({stability_SE_norm:.2f})")
                print("="*80)


        print(" ")



#### optimized
parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [1,5], "ccp_alpha" : [0,0.1,0.3,0.01]} 
for (case, info), y_var in zip(cases.items(),[1,3,5]):
        np.random.seed(SEED)
        features = info["features"]
        p = info["p"]
        x = X[:,features]
        y = np.random.normal(loc = p(X), scale=y_var,size=n)
        kf = RepeatedKFold(n_splits= 10,n_repeats=1, random_state=SEED)
        criterion = "mse"
        models = {  
                        "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        "sklearn": GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)
                        }
        stability = {k :[] for k in models.keys()}
        performance = {k :[] for k in models.keys()}
        for i,(train_index, test_index) in enumerate(kf.split(x)):
                
                X_12, y_12 = x[train_index],y[train_index]
                X_test,y_test = x[test_index],y[test_index]
                X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
                for name, model in models.items():
                        model.fit(X1,y1)
                        # if name =="glm" or name =="sklearn":
                        #     print(model.best_params_)
                        pred1 = model.predict(X_test)
                        if name =="glm" or name =="sklearn":
                                # print(model.best_params_)
                                model.fit(X_12,y_12)
                        else:
                                model.update(X_12,y_12)
                        pred2 = model.predict(X_test)
                        stability[name].append( S2(pred1,pred2))
                        performance[name].append(mean_squared_error(y_test, pred2))
        print(case)
        for name in models.keys():
                print("="*80)
                print(f"{name}")

                mse_scale = np.mean(performance["baseline"]); S_scale = np.mean(stability["baseline"]);
                loss_score = np.mean(performance[name])
                loss_SE = np.std(performance[name])/np.sqrt(len(performance[name]))
                loss_SE_norm = np.std(performance[name]/mse_scale)/np.sqrt(len(performance[name]))
                stability_score = np.mean(stability[name])
                stability_SE = np.std(stability[name])/np.sqrt(len(performance[name]))
                stability_SE_norm = np.std(stability[name]/S_scale)/np.sqrt(len(performance[name]))
                print(f"test - mse: {loss_score:.3f} ({loss_SE:.2f}), stability: {stability_score:.3f} ({stability_SE:.2f})")
                print(f"test - mse: {loss_score/mse_scale:.2f} ({loss_SE_norm:.2f}), stability: {stability_score/S_scale:.2f} ({stability_SE_norm:.2f})")
                print("="*80)


        print(" ")


    
    