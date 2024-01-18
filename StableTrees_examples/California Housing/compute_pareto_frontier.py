from stabletrees.random_forest import RF
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd

from sklearn.linear_model import TweedieRegressor
from stabletrees import BaseLineTree,AbuTree,NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,SklearnTree,AbuTree2
from adjustText import adjust_text
from pareto_efficient import is_pareto_optimal
from sklearn.datasets import fetch_california_housing
from tqdm import tqdm
SEED = 0
EPSILON = 1.1

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)

parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [5]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)

# from examples in R package ISLR2, https://cran.r-project.org/web/packages/ISLR2/ISLR2.pdf

SEED = 0
EPSILON = 1.1


markers = {"baseline":"baseline","NU":"NU","SL": "SL", "SL1":"SL_{0.1}", "SL2":"SL_{0.25}","SL3":"SL_{0.5}",
            "SL4": "SL_{0.75}", "SL5": "SL_{0.9}",
            "TR":"TR","TR1":"TR_{0,5}",
            "TR2":"TR_{5,5}", "TR3" :"TR_{10,5}",
            "ABU":"ABU",
            "ABU1":r"ABU_{0.1}","ABU2":r"ABU_{0.25}","ABU3":r"ABU_{0.5}","ABU4":r"ABU_{0.75}","ABU5":r"ABU_{0.9}",
            "BABU":"BABU", "BABU1": r"BABU_{1}","BABU2": r"BABU_{3}" ,"BABU3": r"BABU_{5}","BABU4": r"BABU_{7}","BABU5": r"BABU_{10}","BABU6": r"BABU_{20}"   }

colors = {"baseline":"#3776ab","NU":"#D55E00", "SL":"#CC79A7","SL1":"#CC79A7", "SL2":"#CC79A7","SL3":"#CC79A7",
            "SL4": "#CC79A7", "SL5": "#CC79A7",
            "TR":"#009E73","TR1":"#009E73", "TR2":"#009E73","TR3":"#009E73",
            "TR4":"#009E73", "TR5":"#009E73","TR6":"#009E73",
            "TR7":"#009E73", "TR8":"#009E73","TR9":"#009E73",
            "ABU":"#F0E442",
            "ABU1":"#F0E442",
            "ABU2":"#F0E442",
            "ABU3":"#F0E442",
            "ABU4":"#F0E442",
            "ABU5":"#F0E442",
            "BABU": "#E69F00", "BABU1": "#E69F00","BABU2": "#E69F00" ,"BABU3": "#E69F00","BABU4": "#E69F00","BABU5": "#E69F00","BABU6": "#E69F00"}

colors2 = {"baseline":"#3776ab", "NU":"#D55E00", "NU":"#D55E00", "SL":"#CC79A7", 
            "TR":"#009E73", 
            "ABU":"#F0E442",
            "BABU": "#E69F00",}


X,y = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=False) #get CH data
plot_info  = {"CH":[]} # (x,y,colors,marker)

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
                       "baseline": BaseLineTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        "SL1": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
                        "SL2": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
                        "SL3": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
                        "SL4": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
                        "SL5": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
                        "ABU": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        "ABU1": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.1),
                        "ABU2": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.25),
                        "ABU3": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.5),
                        "ABU4": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.75),
                        "ABU5": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.9),
                        "BABU": AbuTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                }
stability_all = {name:[] for name in models.keys()}
standard_stability_all= {name:[] for name in models.keys()}
mse_all= {name:[] for name in models.keys()}

kf = RepeatedKFold(n_splits= 10,n_repeats=5, random_state=SEED)

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


for _, (train_index, test_index) in tqdm(enumerate(kf.split(X)), desc="running repeated k-fold"):
    X_12, y_12 = X[train_index],y[train_index]
    X_test,y_test = X[test_index],y[test_index]
    X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
    # initial model 
    criterion = "mse"

    models = {  
                    "baseline": BaseLineTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                    "SL1": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
                    "SL2": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
                    "SL3": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
                    "SL4": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
                    "SL5": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
                    "ABU": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                    "ABU1": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.1),
                        "ABU2": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.25),
                        "ABU3": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.5),
                        "ABU4": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.75),
                        "ABU5": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.9),
                    "BABU": AbuTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
            }
    

    for name, model in models.items():

        model.fit(X1,y1)
        
        pred1 = model.predict(X_test)
        
        pred1_train = model.predict(X_12)
        
        pred1_orig= model.predict(X1)
        
        #print("before")
        if name == "standard":
            model.fit(X_12,y_12)
        else:
            model.update(X_12,y_12)
        #print("after")
        pred2 = model.predict(X_test)
        pred2_orig= model.predict(X1)
        pred2_train =  model.predict(X_12)


        orig_mse[name].append(mean_squared_error(pred2_orig,y1))
        orig_stability[name].append(S2(pred1_orig,pred2_orig))

        train_mse[name].append(mean_squared_error(pred2_train,y_12))
        train_stability[name].append(S2(pred1_train,pred2_train))

        mse[name].append(mean_squared_error(y_test,pred2))

        stability[name].append(S2(pred1,pred2))
    

for name in tqdm(models.keys(), desc = "computing pareto frontier"):
    print("="*80)
    print(f"{name}")
    
    mse_scale = np.mean(mse["baseline"]);
    
    mse_scale = np.mean(mse["baseline"]); S_scale = np.mean(stability["baseline"]);
    loss_score = np.mean(mse[name])
    loss_SE = np.std(mse[name])/np.sqrt(10) #np.sqrt(len(mse[name]))
    loss_SE_norm = np.std(mse[name]/mse_scale)/np.sqrt(10) #np.sqrt(len(mse[name]))
    stability_score = np.mean(stability[name])
    stability_SE = np.std(stability[name])/np.sqrt(10)  #np.sqrt(len(mse[name]))
    stability_SE_norm = np.std(stability[name]/S_scale)/np.sqrt(10) #/np.sqrt(len(mse[name]))
    print(f"test - mse: {loss_score:.3f} ({loss_SE:.3f}), stability: {stability_score:.3f} ({stability_SE:.3f})")
    print(f"test - mse: {loss_score/mse_scale:.3f} ({loss_SE_norm:.2f}), stability: {stability_score/S_scale:.3f} ({stability_SE_norm:.2f})")
    print("="*80)
    mse_all[name] += [score/mse_scale for score in mse[name]]
    if name!= "standard":
        x_abs =  np.mean((mse[name]))
        y_abs = np.mean(stability[name])
        x_abs_se = loss_SE
        y_abs_se =stability_SE
        x_se  = loss_SE_norm
        y_se  = stability_SE_norm
        x_r = x_abs/mse_scale
        y_r = y_abs/S_scale
        plot_info["CH"].append(("CH",name,x_r,y_r,colors[name],markers[name], x_abs,y_abs,x_se, y_se, x_abs_se, y_abs_se ))
print()


import itertools
import os
df_list = list(itertools.chain(*plot_info.values()))
df = pd.DataFrame(df_list, columns=["dataset","method",'loss', 'stability', 'color', "marker", 'loss_abs', "stability_abs",'loss_se', 'stability_se', 'loss_abs_se', 'stability_abs_se'  ] )

if os.path.isfile('results/tree_CH__results.csv'):
    old_df =pd.read_csv('results/tree_CH__results.csv')
    for i,(d,m) in enumerate(zip(df.dataset, df.marker)):
        index = old_df.loc[(old_df["dataset"] == d) & (old_df["marker"] ==m)].index
        values  = df.iloc[i]
        if len(index)>0:
            old_df.iloc[index]=values
        else:
            print(values)
            old_df  = old_df.append(values, ignore_index=True)

    old_df.to_csv('results/tree_CH__results.csv', index=False)
else:
    df.to_csv('results/tree_CH_results.csv', index=False)

