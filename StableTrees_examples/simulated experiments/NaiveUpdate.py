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
    return X[:,0]**2 + 0.75*X[:,1]
def p3(X):
    return X[:,0]**2 + 0.75*X[:,1] -  0.25*X[:,3] + 0.1*X[:,0]*X[:,2]


cases = {"case 1": {"features": [0], "p":p1},
         "case 2": {"features": [0,3], "p":p2},
          "case 3": {"features": [0,1,2,3], "p":p3}}





criterion = "mse"
models = {  
                "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                "NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                }

#ns = [500,1000,1500,2000,2500,3000,3500,4000,4500, 5000]#
ns = np.arange(1000,11000,1000)
print(ns)
#ns = [1000]
stability = {k :[] for k in cases.keys()}
performance = {k :[] for k in cases.keys()}

for n in ns:
    np.random.seed(SEED)
    X1 = np.random.uniform(0,4,size=(n,1))
    X2 = np.random.uniform(0,4,size=(n,1))
    X3 = np.round(np.random.uniform(0,1,size=(n,1)),decimals=0)
    X4 = np.round(np.random.uniform(0,4,size=(n,1)),decimals=0)
    X = np.hstack((X1,X2,X3,X4))
    for case, info in cases.items():
        np.random.seed(SEED)
        features = info["features"]
        p = info["p"]
        x = X[:,features]
        y = np.random.normal(loc = p(X), scale=1,size=n)
        kf = RepeatedKFold(n_splits= 5, n_repeats=10, random_state=SEED)
        criterion = "mse"
        models = {  
                        "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        "NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True)
                        }
        stab= {k :[] for k in models.keys()}
        perform= {k :[] for k in models.keys()}
        for i,(train_index, test_index) in enumerate(kf.split(x)):
            X_12, y_12 = x[train_index],y[train_index]
            X_test,y_test = x[test_index],y[test_index]
            X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
            for name, model in models.items():
                model.fit(X1,y1)
                pred1 = model.predict(X_test)
                if name =="glm" or name =="sklearn":
                        model.fit(X_12,y_12)
                else:
                        model.update(X_12,y_12)
                pred2 = model.predict(X_test)
                stab[name].append( S2(pred1,pred2))
                perform[name].append(mean_squared_error(y_test, pred2))
        print(case)
        for name in models.keys():
            print("="*80)
            print(f"{name}")

            mse_scale = np.mean(perform["baseline"]); S_scale = np.mean(stab["baseline"]);

            print(f"test - mse: {np.mean(perform[name]):.3f} ({np.mean(perform[name])/mse_scale:.2f}), stability: {np.mean(stab[name]):.3f} ({np.mean(stab[name])/S_scale:.2f})")
            print("="*80)
        print(" ")
        performance[case].append(np.mean(perform["NU"])/mse_scale)
        stability[case].append(np.mean(stab["NU"])/S_scale)
print(performance)
print(stability)

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}
plt.rcParams.update(params)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,dpi=300)
colors = {"Case 1":"orange", "Case 2": "g", "Case 3":"r"}
ax1.plot(ns, [p for p in performance["case 1"]],c = "#E69F00", label = "case 1")
ax1.plot(ns,[p for p in performance["case 2"]],c = "#009E73", label = "case 2")
ax1.plot(ns, [p for p in performance["case 3"]],c = "#CC79A7", label = "case 3")
ax1.set_ylabel('mse',fontsize=10)
ax1.set_title("a) Loss")

# plt.plot([s for s in stability["case 1"]])
# plt.plot([s for s in stability["case 2"]])
# plt.plot([s for s in stability["case 3"]])
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
ax1.xaxis.set_major_locator(MultipleLocator(2000))
ax1.xaxis.set_minor_locator(AutoMinorLocator(2))

ax2.set_ylabel("stability",fontsize=10)
ax2.plot(ns, [s for s in stability["case 1"]],c = "#E69F00")
ax2.plot(ns, [s for s in stability["case 2"]],c = "#009E73")
ax2.plot(ns, [s for s in stability["case 3"]],c = "#CC79A7")
ax2.set_xlabel('Size of dataset',fontsize=10)
ax2.set_title("b) Stability")
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
# fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
fig.legend(lines_labels,     # The line objects
           labels=labels,   # The labels for each line
           loc="center right",   # Position of legend
           borderaxespad=0.1,
            bbox_to_anchor=(1, 0.55),    # Small spacing around legend box
           fontsize="10")
plt.tight_layout()
plt.subplots_adjust(right=0.84)
plt.savefig(f"StableTrees_examples\plots\\example_mse_simulated_NU.png")
plt.close()



    
    