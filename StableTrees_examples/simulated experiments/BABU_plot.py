
from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.linear_model import PoissonRegressor,LinearRegression
from adjustText import adjust_text
from pareto_efficient import is_pareto_optimal
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

# cases = {"Case 1": {"features": [0], "p":p1}}

# import matplotlib.pyplot as plt
# import numpy as np

#plt.style.use("ggplot")

# t = np.arange(0.0, 2.0, 0.1)
# s = np.sin(2 * np.pi * t)
# s2 = np.cos(2 * np.pi * t)
# plt.plot(t, s, "o-", lw=4.1)
# plt.plot(t, s2, "o-", lw=4.1)
# plt.xlabel("time (s)")
# plt.ylabel("Voltage (mV)")
# plt.title("Simple plot $\\frac{\\alpha}{2}$")
#plt.grid(True)


# tikzplotlib.save("test.tex")

np.random.seed(SEED)
n = 1000
X1 = np.random.uniform(0,4,size=(n,1))
X2 = np.random.uniform(0,4,size=(n,1))
X3 = np.round(np.random.uniform(0,1,size=(n,1)),decimals=0)
X4 = np.round(np.random.uniform(0,4,size=(n,1)),decimals=0)

X = np.hstack((X1,X2,X3,X4))

colors = {"Case 1":"#E69F00", "Case 2": "#009E73", "Case 3":"#CC79A7"}
#markers = {"baseline":"o", "NU": "v", "TR":"^", "TR":"s", "SL":"D","ABU":"+", "BABU": "*" }
markers = {"baseline":"$B$",
             "BABU1": r"BABU_{1}","BABU2": r"BABU_{3}" ,"BABU3": r"BABU_{5}","BABU4": r"BABU_{7}","BABU5": r"BABU_{10}","BABU6": r"BABU_{20}"   }

plot_info  = [] # (x,y,colors,marker)

params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}


for (case, info), y_var in zip(cases.items(),[1,3,3]):
        np.random.seed(SEED)
        features = info["features"]
        p = info["p"]
        x = X[:,features]
        y = np.random.normal(loc = p(X), scale=y_var,size=n)
        kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
        criterion = "mse"
        models = {  
                        "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        "BABU1":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=1),
                        "BABU2":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=3),
                        "BABU3":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=5),
                        "BABU4":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=7),
                        "BABU5":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=10),
                        "BABU6":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=20)
                        }
        stability = {k :[] for k in models.keys()}
        performance = {k :[] for k in models.keys()}
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
                        stability[name].append( S2(pred1,pred2))
                        performance[name].append(mean_squared_error(y_test, pred2))
        print(case)
        for name in models.keys():
            print("="*80)
            print(f"{name}")

            mse_scale = np.mean(performance["baseline"]); S_scale = np.mean(stability["baseline"]);

            print(f"test - mse: {np.mean(performance[name]):.3f} ({np.mean(performance[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S_scale:.2f})")
            print("="*80)
            if name != "baseline":
                    x = np.mean((performance[name]))/mse_scale
                    y = np.mean(stability[name])/S_scale
                    plot_info.append((x,y,colors[case],markers[name]))

        print(" ")
from matplotlib.lines import Line2D
fig, ax  = plt.subplots(dpi=500)
plt.rcParams.update(params)

X = np.zeros((len(plot_info)+1, 2))
X[1:,0] = [x for (x,y,c,s) in plot_info]
X[1:,1] = [y for (x,y,c,s) in plot_info]
X[0,0] = 1
X[0,1] = 1
ax.axvline(x=1, linestyle = "--", c = "#3776ab",lw = 0.5)
ax.axhline(y=1, linestyle = "--", c = "#3776ab",lw = 0.5)
plt.text(x = 1, y=1.01, s = "$baseline$",fontsize=8, ha='center', va='center')
texts = [plt.text(x = x, y=y, s = "$"+s+"$",fontsize=8, ha='center', va='center') for (x,y,c,s) in plot_info]
scatters = [ax.scatter(x = x, y=y, s = 4, c =c) for (x,y,c,_) in plot_info]
scatters.append(ax.scatter(x = [1], y=[1], s = 4, c ="#3776ab"))
adjust_text(texts,add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax)
legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                          markerfacecolor=v, markersize=14) for k,v in colors.items()  ]
legend_elements = [Line2D([0], [0], marker='s', color='w',markerfacecolor="#3776ab",markersize=14, label='baseline', linestyle = "--")] +legend_elements
#plt.plot([x for x,y in frontier],[y for x,y in frontier], c = "k", lw=0.1)

plt.xlabel("mse",fontsize=10)
plt.ylabel("stability",fontsize=10)
plt.ylim((0.05,1.1))
plt.xlim((0.98,1.025))
plt.legend(loc='upper right' , handles=legend_elements,fontsize="10")
plt.savefig(f"StableTrees_examples\plots\\example_mse_simulated_BABU_09_05.png")
plt.close()

    
    