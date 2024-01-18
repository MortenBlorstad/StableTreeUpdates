from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.linear_model import PoissonRegressor,LinearRegression
from adjustText import adjust_text
from sklearn.metrics import mean_poisson_deviance

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
    return X[:,0]**2 + 0.75*X[:,1] -  0.25*X[:,3] + 0.1*X[:,0]*X[:,2]

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

plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}

colors = {"baseline":'#1f77b4',"NU":"#D55E00", "SL":"#CC79A7", 
            "TR":"#009E73", 
            "ABU":"#F0E442",
            "BABU": "#E69F00",}

criterion = "poisson"
models = {  
                "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                "NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                "TR":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, delta=0.05),
                "SL":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.75),
                "ABU":AbuTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                "BABU": BABUTree(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True)
                }
time = 10
prev_pred = {name:0 for name in models.keys()}
performance = {name:[0]*time for name in models.keys()}
stab = {name:[0]*time for name in models.keys()}
plt.rcParams.update(plot_params)
f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 4.8),dpi=500)

repeat =50
for i in range(repeat):
    np.random.seed(i)
    x = np.sort(np.random.uniform(0,4,(1000,1)),axis=0)
    y = np.random.poisson(lam = x.ravel(),size=1000)
    np.random.seed(i+1)
    x_test = np.sort(np.random.uniform(0,4,(1000,1)),axis=0)
    y_test = np.random.poisson(lam = x_test.ravel(),size=1000)
    
    
    for name, model in models.items():
        model.fit(x,y)

        pred = model.predict(x_test)

        prev_pred[name] = pred

    
    for t in range(time): 
        x_t = np.random.uniform(0,4,(1000,1))    
        y_t = np.random.poisson(lam = x_t.ravel(),size=1000)
        x  = np.vstack((x,x_t))

        y = np.concatenate((y,y_t))

        for name, model in models.items():
            model.update(x,y)

            pred = model.predict(x_test)
            stab[name][t] += S1(prev_pred[name], pred)
            performance[name][t] += mean_poisson_deviance(y_test,pred)
            prev_pred[name] = pred
scatters = []
texts = []
for name, model in models.items():
    print(name, [val/time for val in performance[name]])
    performance[name] = [val/repeat for val in performance[name]]
    stab[name] = [val/repeat for val in stab[name]]
    ax.plot(performance[name], stab[name] , label = name,linestyle='--', marker='o', markersize = 3,linewidth=1, c = colors[name])#np.arange(1,time+1,dtype=int)
    scatters+= [ax.scatter(x = x, y=y, s = 0.1, alpha=0) for (x,y) in zip(performance[name],stab[name])]
    texts += [ ax.text(x =x, y=y, s = r"$t="+str(i+1)+"$",fontsize=8, ha='right', va='center')  for i,(x,y) in enumerate(zip(performance[name],stab[name])) if  i==0]#(i+1) %5 ==0
    
    ax.set_ylabel("stability",fontsize=12)
    ax.set_xlabel("Poisson deviance",fontsize=12)
    plt.legend(fontsize = 8)

adjust_text(texts,add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax)
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\example_poisson_simulated_overtime_theory.png")
plt.close()
