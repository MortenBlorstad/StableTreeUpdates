from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,AbuTree2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.linear_model import PoissonRegressor,LinearRegression
from adjustText import adjust_text
from tqdm import tqdm

SEED = 0
EPSILON = 1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)


def p3(X,t):
    a = 0.2*(1+0.01*t); b = 0.75*(1+0.05*t); c = 0.25*(1+0.1*t); d = 0.01 * t
    return a*X[:,0]**2 + b*X[:,1] -  c*X[:,3] + d*X[:,0]*X[:,2]

np.random.seed(SEED)

plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}

colors = {"sklearn": "#009E73",
    "baseline":'#1f77b4',"NU":"#D55E00", "SL":"#CC79A7", "SL2":"#b76c96","SL3":"#a36085",
            "TR":"#009E73", 
            "ABU":"#F0E442",
            "ABU2":"#d8cd3b",
            "BABU": "#E69F00",
            }



criterion = "mse"
models = {  
                "sklearn": DecisionTreeRegressor(criterion="squared_error", min_samples_leaf=5, ccp_alpha=0.01 ),
                "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                "SL":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.1),
                "SL2":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.25),
                "SL3":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.5),
                "ABU": AbuTree2(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True),
                "BABU":AbuTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),


                }


time = 10
prev_pred = {name:0 for name in models.keys()}
mse = {name:[0]*time for name in models.keys()}
stab = {name:[0]*time for name in models.keys()}
n = 1000

n_features = 4

repeat =50
for i in tqdm(range(repeat), desc="running"):
    np.random.seed(i)
    t_weights = np.ones(n)
    x =  np.random.uniform(0,4,(n,n_features))#np.sort(,axis=0)
    y = np.random.normal(p3(x,0),1,n)
    np.random.seed(422+i)
    x_test = np.random.uniform(0,4,(10000,n_features))
    
    for name, model in models.items():
        
        model.fit(x,y,t_weights)

        pred = model.predict(x_test)

        prev_pred[name] = pred

    
    for t in range(time): 
        t_weights = np.concatenate((t_weights,np.ones(n)*(t+2)))/(t+2)
        x_t = np.random.uniform(0,4,(n,n_features))    
        y_t = np.random.normal(p3(x_t,t+1),1,n)
        x  = np.vstack((x,x_t))

        y = np.concatenate((y,y_t))
        y_test = np.random.normal(p3(x_test,t+1),1,x_test.shape[0])
        for name, model in models.items():
            if name =="sklearn":
                model.fit(x,y,t_weights)
            else:
                model.update(x,y,t_weights)

            pred = model.predict(x_test)
            stab[name][t] += np.mean((prev_pred[name]-pred)**2)
            mse[name][t] +=np.mean((y_test-pred)**2)
            prev_pred[name] = pred
scatters = []
texts = []
t = 0

labels = {
        "sklearn": "sklearn",
        "baseline": "baseline",
        "SL": "SL (γ=0.1)",
        "SL2": "SL (γ=0.25)",
        "SL3": "SL (γ=0.5)",
        "ABU": "ABU",
        "ABU2": "ABU2",
        "BABU": "BABU"
        }

for name, model in models.items():
        mse[name] = [val/repeat for val in mse[name]]
        stab[name] = [val/repeat for val in stab[name]]


plt.rcParams.update(plot_params)
f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 4.8),dpi=500)

for name, model in models.items():
    ax.plot(mse[name], stab[name] , label = labels[name],linestyle='--', marker='o', markersize = 3,linewidth=1, c = colors[name])#np.arange(1,time+1,dtype=int)
    scatters+= [ax.scatter(x = x, y=y, s = 0.1, alpha=0) for (x,y) in zip(mse[name],stab[name])]
    texts += [ ax.text(x =x, y=y, s = r"$t="+str(1)+"$",fontsize=8, ha='right', va='center')  for i,(x,y) in enumerate(zip(mse[name],stab[name])) if  i==0]#(i+1) %5 ==0
    
# ax.axvline(x=mse["baseline"][t], linestyle = "--", c = "#3776ab", lw = 0.5)
# ax.axhline(y=stab["baseline"][t], linestyle = "--", c = "#3776ab", lw = 0.5)
ax.set_ylabel("instability",fontsize=12)
ax.set_xlabel("loss",fontsize=12)
plt.legend(loc='upper left')
adjust_text(texts,add_objects=scatters,ax= ax)
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\example_mse_simulated_overtime_drift.png")
plt.close()


