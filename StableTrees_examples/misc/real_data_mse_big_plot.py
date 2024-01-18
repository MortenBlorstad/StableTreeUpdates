import matplotlib.pyplot as plt
from stabletrees import BaseLineTree,AbuTree,NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,SklearnTree
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,mean_poisson_deviance
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import datapreprocess
from sklearn.linear_model import LinearRegression
from adjustText import adjust_text
import sys
sys.path.insert(0, './simulated experiments')
from pareto_efficient import is_pareto_optimal

SEED = 0
EPSILON = 2

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)

parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [5]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)

# from examples in R package ISLR2, https://cran.r-project.org/web/packages/ISLR2/ISLR2.pdf
datasets =["Boston", "Carseats","College", "Hitters", "Wage"]
targets = ["medv", "Sales", "Apps", "Salary", "wage"]
features =  [[ "crim", "rm"], ["Advertising", "Price"], ["Private", "Accept"], ["AtBat", "Hits"], ["year", "age"] ]
SEED = 0
EPSILON = 1.1

#colors = {"Boston":"orange", "Carseats": "g", "College":"r", "Hitters":"c", "Wage":"m"}
#markers = {"baseline":"o", "NU": "v", "TR":"^", "TR":"s", "SL":"D","ABU":"+", "BABU": "*" }
markers = {"baseline":"B","NU":"NU", "SL3":"SL_{0.1}", "SL4":"SL_{0.25}","SL5":"SL_{0.5}",
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



plot_info  = {ds:[] for ds in datasets} # (x,y,colors,marker)

plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}
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
                "BABU2":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=3),
                "BABU3":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=5),
                "BABU4":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=7),
                "BABU5":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=10),
                "BABU6":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=20)
                }
stability_all = {name:[] for name in models.keys()}
standard_stability_all= {name:[] for name in models.keys()}
mse_all= {name:[] for name in models.keys()}
for ds,target, feature in zip(datasets,targets, features):
    iteration = 1
    kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
    data = pd.read_csv("data/"+ ds+".csv") # load dataset
    # data preperation
    # data = data.dropna(axis=0, how="any") # remove missing values if any
    # data = data.loc[:, feature + [target]] # only selected feature and target variable
    # cat_data = data.select_dtypes("object") # find categorical features
    # if not cat_data.empty: # if any categorical features, one-hot encode them
    #     cat_data = pd.get_dummies(data.select_dtypes("object"), prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
    #     data = pd.concat([data.select_dtypes(['int','float']),cat_data],axis=1)
    data = datapreprocess.data_preperation(ds)
    #print(data.corr())
    
    y = data[target].to_numpy()
    X = data.drop(target, axis=1).to_numpy()
    #if ds == "College":
    #    y = np.log(y)
   
    # initial model 
    
    stability = {name:[] for name in models.keys()}
    standard_stability = {name:[] for name in models.keys()}
    mse = {name:[] for name in models.keys()}
    train_stability = {name:[] for name in models.keys()}
    train_standard_stability = {name:[] for name in models.keys()}
    train_mse = {name:[] for name in models.keys()}
    orig_stability = {name:[] for name in models.keys()}
    orig_standard_stability = {name:[] for name in models.keys()}


    orig_mse = {name:[] for name in models.keys()}
    for train_index, test_index in kf.split(X):
        X_12, y_12 = X[train_index],y[train_index]
        X_test,y_test = X[test_index],y[test_index]
        X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
        clf.fit(X1,y1)
        params = clf.best_params_
        # initial model 
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
                "BABU2":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=3),
                "BABU3":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=5),
                "BABU4":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=7),
                "BABU5":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=10),
                "BABU6":BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=20)
                }
        for name, model in models.items():
            model.fit(X1,y1)
            pred1 = model.predict(X_test)
            pred1_train = model.predict(X_12)
            pred1_orig= model.predict(X1)
            #print("before")
            if name =="GLM":
                model.fit(X_12,y_12)
            else:
                model.update(X_12,y_12)
            
            #print("after")
            pred2 = model.predict(X_test)
            pred2_orig= model.predict(X1)
            pred2_train =  model.predict(X_12)

            orig_mse[name].append(mean_squared_error(pred2_orig,y1))
            orig_stability[name].append(S1(pred1_orig,pred2_orig))
            orig_standard_stability[name].append(S2(pred1_orig,pred2_orig))

            train_mse[name].append(mean_squared_error(pred2_train,y_12))
            train_stability[name].append(S1(pred1_train,pred2_train))
            train_standard_stability[name].append(S2(pred1_train,pred2_train))
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(S1(pred1,pred2))
            standard_stability[name].append(S2(pred1,pred2))
        

    
    for name in models.keys():
        print("="*80)
        print(f"{name}")
        
        mse_scale = np.mean(mse["baseline"]);
        
        mse_scale = np.mean(mse["baseline"]); S_scale = np.mean(standard_stability["baseline"]);
        loss_score = np.mean(mse[name])
        loss_SE = np.std(mse[name])/np.sqrt(len(mse[name]))
        loss_SE_norm = np.std(mse[name]/mse_scale)/np.sqrt(len(mse[name]))
        stability_score = np.mean(standard_stability[name])
        stability_SE = np.std(standard_stability[name])/np.sqrt(len(mse[name]))
        stability_SE_norm = np.std(standard_stability[name]/S_scale)/np.sqrt(len(mse[name]))
        print(f"test - mse: {loss_score:.3f} ({loss_SE:.2f}), stability: {stability_score:.3f} ({stability_SE:.2f})")
        print(f"test - mse: {loss_score/mse_scale:.2f} ({loss_SE_norm:.2f}), stability: {stability_score/S_scale:.2f} ({stability_SE_norm:.2f})")
        print("="*80)
        mse_all[name] += [score/mse_scale for score in mse[name]]
        if name != "baseline":
            x = np.mean((mse[name]))/mse_scale
            y = np.mean(standard_stability[name])/S_scale
            plot_info[ds].append((x,y,colors[name],markers[name]))

    print()


from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
# create figure and axes
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8.27, 11), sharey=True,dpi=500)
axes = axes.ravel()
plt.rcParams.update(plot_params)

college_box_plot_info = []


# plot data on the axes
for ds,ax in zip(datasets,axes[:-1]):
    frontier = []
    X = np.zeros((len(plot_info[ds]), 2))
    X[:,0] = [x for (x,y,c,s) in plot_info[ds]]
    X[:,1] = [y for (x,y,c,s) in plot_info[ds]]
    for i in range(X.shape[0]):
        if is_pareto_optimal(i, X):
            frontier.append((X[i,0],X[i,1]))
    frontier = sorted(frontier)


    print(frontier)
    frontier = [ (frontier[0][0], 2) ] + frontier+ [ (2, frontier[-1][1]) ]
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.axvline(x=1, linestyle = "--")
    ax.axhline(y=1, linestyle = "--")
    ax.set_title(ds,fontsize = 10)
    texts = [ax.text(x = x, y=y, s = r"$\mathbf{"+s+"}$",fontsize=8, ha='center', va='center',weight='heavy') if (x,y) in frontier else ax.text(x = x, y=y, s = "$"+s+"$",fontsize=8, ha='center', va='center') for (x,y,c,s) in plot_info[ds]]
    scatters = [ax.scatter(x = x, y=y, s = 6, c =c) for (x,y,c,_) in plot_info[ds]]
    adjust_text(texts,add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax)
    ax.set_xlabel("mse",fontsize=10)
    ax.set_ylabel(r'$\left(f_1(x_i)-f_2(x_i)\right)^2$',fontsize=10)


    

# create a common legend for all the plots
legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                            markerfacecolor=v, markersize=14) for k,v in colors2.items()  ]
legend_elements = [Line2D([0], [0], color='b', lw=1, label='baseline', linestyle = "--")] +legend_elements
print()
axes[-1].legend( handles=legend_elements, loc='center',fontsize="10")
axes[-1].axis("off")
# adjust spacing between subplots
fig.tight_layout()


plt.savefig(f"StableTrees_examples\plots\\real_data_mse_pareto.png")
plt.close()

