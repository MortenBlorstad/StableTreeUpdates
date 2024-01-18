
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

#colors = {"Case 1":"#E69F00", "Case 2": "#009E73", "Case 3":"#CC79A7"}
#markers = {"baseline":"o", "NU": "v", "TR":"^", "TR":"s", "SL":"D","ABU":"+", "BABU": "*" }
markers = {"baseline":"$B$","NU":"NU", "SL3":"SL_{0.1}", "SL4":"SL_{0.25}","SL5":"SL_{0.5}",
            "SL6": "SL_{0.75}", "SL7": "SL_{0.9}",
            "TR1":"TR_{0,25}", "TR2":"TR_{0,10}","TR3":"TR_{0,5}",
            "TR4":"TR_{5,25}", "TR5":"TR_{5,10}","TR6":"TR_{5,5}",
            "TR7":"TR_{10,25}", "TR8":"TR_{10,10}","TR9":"TR_{10,5}",
            "ABU":"ABU",
             "BABU1": r"BABU_{1}","BABU2": r"BABU_{3}" ,"BABU3": r"BABU_{5}","BABU4": r"BABU_{7}","BABU5": r"BABU_{10}","BABU6": r"BABU_{20}"   }

colors = {"baseline":"$B$","NU":"#D55E00", "SL3":"#CC79A7", "SL4":"#CC79A7","SL5":"#CC79A7",
            "SL6": "#CC79A7", "SL7": "#CC79A7",
            "TR1":"#009E73", "TR2":"#009E73","TR3":"#009E73",
            "TR4":"#009E73", "TR5":"#009E73","TR6":"#009E73",
            "TR7":"#009E73", "TR8":"#009E73","TR9":"#009E73",
            "ABU":"#F0E442",
             "BABU1": "#E69F00","BABU2": "#E69F00" ,"BABU3": "#E69F00","BABU4": "#E69F00","BABU5": "#E69F00","BABU6": "#E69F00"}

colors2 = {"NU":"#D55E00", "SL":"#CC79A7", 
            "TR":"#009E73", 
            "ABU":"#F0E442",
            "BABU": "#E69F00",}


plot_info  = {case:[] for case in cases.keys()} # (x,y,colors,marker)

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
                        "NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        "TR1":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,delta=0.25,alpha=0.0),
                        "TR3":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,delta=0.05,alpha=0.0),
                        "TR4":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,delta=0.25,alpha=0.05),
                        "TR6":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,delta=0.05,alpha=0.05),
                        "TR7":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,delta=0.25,alpha=0.1),
                        "TR9":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,delta=0.05,alpha=0.1),
                        "SL3":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.1),
                        "SL4":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.25),
                        "SL5":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.5),
                        "SL6":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.75),
                        "SL7":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.9),
                        "ABU":AbuTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                        "BABU1":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=1),
                        # "BABU2":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=3),
                        # "BABU3":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=5),
                        # "BABU4":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=7),
                        # "BABU5":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=10),
                        # "BABU6":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=20)
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
                        plot_info[case].append((x,y,colors[name],markers[name]))

        print(" ")

print(plot_info)
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
fig, axes,  = plt.subplots(nrows =1, ncols=3,dpi=500,figsize=(12, 12/1.61803398875))#
axes = axes.ravel()
plt.rcParams.update(params)
for case,ax in zip(cases.keys(),axes):
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    frontier = []
    X = np.zeros((len(plot_info[case]), 2))
    X[:,0] = [x for (x,y,c,s) in plot_info[case]]
    X[:,1] = [y for (x,y,c,s) in plot_info[case]]
    for i in range(X.shape[0]):
        if is_pareto_optimal(i, X):
            frontier.append((X[i,0],X[i,1]))
    frontier = sorted(frontier)
    print(frontier)
    frontier = [ (frontier[0][0], 2) ] + frontier+ [ (2, frontier[-1][1]) ]
    print(plot_info[case])
    ax.axvline(x=1, linestyle = "--")
    ax.axhline(y=1, linestyle = "--")
    
    texts = [ax.text(x = x, y=y, s = r"$\mathbf{"+s+"}$",fontsize=8, ha='center', va='center',weight='heavy') if (x,y) in frontier else ax.text(x = x, y=y, s = "$"+s+"$",fontsize=8, ha='center', va='center') for (x,y,c,s) in plot_info[case]]
    scatters = [ax.scatter(x = x, y=y, s = 6, c =c) for (x,y,c,_) in plot_info[case]]
    adjust_text(texts,add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax)
    legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                            markerfacecolor=v, markersize=14) for k,v in colors2.items()  ]
    legend_elements = [Line2D([0], [0], color='b', lw=1, label='baseline', linestyle = "--")] +legend_elements
    #plt.plot([x for x,y in frontier],[y for x,y in frontier], c = "k", lw=0.1)

    ax.set_xlabel("mse",fontsize=10)
    ax.set_ylabel(r'$\left(f_1(x_i)-f_2(x_i)\right)^2$',fontsize=10)
    # ax.set_ylim((0.10,1.05))  
    # ax.set_xlim((0.95,1.15))
    plt.legend(loc='upper right', handles=legend_elements,fontsize="10")
#plt.show()
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\example_mse_simulated_all.png")
plt.close()

    
    