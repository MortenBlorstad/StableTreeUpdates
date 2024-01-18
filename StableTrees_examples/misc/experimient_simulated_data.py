from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.linear_model import PoissonRegressor,LinearRegression

SEED = 0
EPSILON = 1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)


def p1(X):
    return X[:,0]
def p2(X):
    return 0.01*X[:,0]**2 + 0.1*X[:,3]
def p3(X):
    return 0.01*X[:,0]**2 + 0.005*X[:,1]*X[:,2] + 0.1*X[:,3]

cases = {"case 1": {"features": [0], "p":p1},
         "case 2": {"features": [0,3], "p":p2},
          "case 3": {"features": [0,1,2,3], "p":p3}}

np.random.seed(SEED)
n = 1000
X1 = np.random.uniform(0,4,size=(n,1))
X2 = np.random.uniform(0,4,size=(n,1))
X3 = np.round(np.random.uniform(0,1,size=(n,1)),decimals=0)
X4 = np.round(np.random.uniform(0,4,size=(n,1)),decimals=0)

X = np.hstack((X1,X2,X3,X4))
# info = cases["case 2"]
# features = info["features"]
# p = info["p"]
# x = X[:,features]
# y = np.random.poisson(lam = np.exp(p(X)),size=n)
# print(np.exp(p(X)))

for case, info in cases.items():
        np.random.seed(SEED)
        features = info["features"]
        p = info["p"]
        x = X[:,features]
        y = np.random.poisson(lam = np.exp(p(X)),size=n)
        kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
        criterion = "poisson"
        models = {  
                        "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        #"sklearn": DecisionTreeRegressor(criterion = criterion,min_samples_leaf=5, max_depth=10),
                       "glm": PoissonRegressor(),
                        #"NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        #"TR":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, delta=0.1),
                        #"SR":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,lmbda=0.75),
                        "ABU":AbuTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        "BABU": BABUTree(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True)
                        }

        stability = {k :[] for k in models.keys()}
        performance = {k :[] for k in models.keys()}
        for i,(train_index, test_index) in enumerate(kf.split(x)):

                X_12, y_12 = x[train_index],y[train_index]
                X_test,y_test = x[test_index],y[test_index]
                X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
                for name, model in models.items():
                        model.fit(X1,y1)
                        # for x_t in X_test:
                        #         print(model.predict(x_t),x_t)
                        pred1 = model.predict(X_test)
                        if name =="glm" or name =="sklearn":
                                model.fit(X_12,y_12)
                        else:
                                model.update(X_12,y_12)
                        pred2 = model.predict(X_test)
                        stability[name].append( S1(pred1,pred2))
                        performance[name].append(mean_poisson_deviance(y_test+0.001, pred2+0.001))
        print(case)
        for name in models.keys():
                print("="*80)
                print(f"{name}")

                mse_scale = np.mean(performance["baseline"]); S_scale = np.mean(stability["baseline"]);

                print(f"test - mse: {np.mean(performance[name]):.3f} ({np.mean(performance[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S_scale:.2f})")
                print("="*80)
        print(" ")

    
    