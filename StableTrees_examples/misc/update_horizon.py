from stabletrees import BaseLineTree, AbuTreeI, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree
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
    return X[:,0]**2 + X[:,3]
def p3(X):
    return X[:,0]**2 + X[:,1]*X[:,2] + X[:,3]

cases = {"case 1": {"features": [0], "p":p1},
         "case 2": {"features": [0,3], "p":p2},
          "case 3": {"features": [0,1,2,3], "p":p3}}



criterion = "poisson"
models = {  
                "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=20, adaptive_complexity=True),
                "NU": NaiveUpdate(criterion = criterion,min_samples_leaf=20, adaptive_complexity=True),
                #"TR":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, delta=0.1),
                "SL":StabilityRegularization(criterion = criterion,min_samples_leaf=20, adaptive_complexity=True,lmbda=0.75),
                "ABU":AbuTreeI(criterion = criterion,min_samples_leaf=20, adaptive_complexity=True),
                "BABU": BABUTree(criterion = criterion,min_samples_leaf=20,adaptive_complexity=True)
                }
repeat =10
time = 10
prev_pred = {name:0 for name in models.keys()}
mse = {name:[0]*time for name in models.keys()}
stab = {name:[0]*time for name in models.keys()}
f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
plt.rcParams["figure.figsize"] = (16,10)
n = 1000
case = "case 1"
info = cases[case]
for i in range(repeat):
    np.random.seed(i)
    X1 = np.random.uniform(0,4,size=(n,1))
    X2 = np.random.uniform(0,4,size=(n,1))
    X3 = np.round(np.random.uniform(0,1,size=(n,1)),decimals=0)
    X4 = np.round(np.random.uniform(0,4,size=(n,1)),decimals=0)

    X = np.hstack((X1,X2,X3,X4))
    np.random.seed(i+1)
    features = info["features"]
    p = info["p"]
    x = X[:,features]
    y = np.random.poisson(lam = p(X),size=n)

    np.random.seed(i+1)
    X1 = np.random.uniform(0,4,size=(n,1))
    X2 = np.random.uniform(0,4,size=(n,1))
    X3 = np.round(np.random.uniform(0,1,size=(n,1)),decimals=0)
    X4 = np.round(np.random.uniform(0,4,size=(n,1)),decimals=0)

    X = np.hstack((X1,X2,X3,X4))
    x_test = X[:,features]
    y_test = np.random.poisson(lam = p(X),size=n)
 
    
    for name, model in models.items():
        model.fit(x,y)

        pred = model.predict(x_test)

        prev_pred[name] = pred

    
    for t in range(time): 

        X1 = np.random.uniform(0,4,size=(n,1))
        X2 = np.random.uniform(0,4,size=(n,1))
        X3 = np.round(np.random.uniform(0,1,size=(n,1)),decimals=0)
        X4 = np.round(np.random.uniform(0,4,size=(n,1)),decimals=0)
        X = np.hstack((X1,X2,X3,X4))
        x_t = X[:,features]   
        y_t = np.random.poisson(lam = p(X),size=n)
        x  = np.vstack((x,x_t))

        y = np.concatenate((y,y_t))

        for name, model in models.items():
            model.update(x,y)

            pred = model.predict(x_test)
            stab[name][t] += S1(prev_pred[name] ,pred)
            mse[name][t] += mean_poisson_deviance(y_test, pred)
            prev_pred[name] = pred
baseline_mse = [val/repeat for val in mse["baseline"]]
baseline_stab = [val/repeat for val in stab["baseline"]]
for name, model in models.items():
    print(name, [val/repeat for val in mse[name]])
    mse[name] = [(val/repeat) for val in mse[name]]
    stab[name] = [(val/repeat) for val in stab[name]]
    ax1.plot(np.arange(1,time+1,dtype=int),mse[name] , label = name)
    ax1.set_ylabel("mean poisson deviance",fontsize=12)
    ax2.plot(np.arange(1,time+1,dtype=int),stab[name] , label = name)
    ax2.set_ylabel(r"$std\left(\frac{log(\hat{y}_{m_2})}{log(\hat{y}_{m_1})}\right)$",fontsize=12)

plt.xticks(np.arange(1,time+1))
ax1.set_xlabel("time")
ax2.set_xlabel("time")
ax1.set_title("performance")
ax2.set_title("stability")
ax1.legend()
ax2.legend()

plt.show()