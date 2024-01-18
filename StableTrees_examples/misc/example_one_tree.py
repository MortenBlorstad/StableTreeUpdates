from stabletrees.random_forest import RF
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import datapreprocess
from sklearn.linear_model import TweedieRegressor
from stabletrees import BaseLineTree,AbuTree,NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,SklearnTree
from adjustText import adjust_text
from pareto_efficient import is_pareto_optimal
SEED = 0
EPSILON = 1.1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean(abs(pred1- pred2))

parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [5]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters)

# from examples in R package ISLR2, https://cran.r-project.org/web/packages/ISLR2/ISLR2.pdf
datasets =["Boston", "Carseats","College", "Hitters", "Wage"]
targets = ["medv", "Sales", "Apps", "Salary", "wage"]
features =  [[ "crim", "rm"], ["Advertising", "Price"], ["Private", "Accept"], ["AtBat", "Hits"], ["year", "age"] ]
SEED = 0
EPSILON = 1.1


markers = {"baseline":"B","NU":"NU","SL": "SL", "SL1":"SL_{0.1}", "SL2":"SL_{0.25}","SL3":"SL_{0.5}",
            "SL4": "SL_{0.75}", "SL5": "SL_{0.9}",
            "TR":"TR","TR1":"TR_{0,5}",
            "TR2":"TR_{5,5}", "TR3" :"TR_{10,5}",
            "ABU":"ABU",
            "BABU":"BABU", "BABU1": r"BABU_{1}","BABU2": r"BABU_{3}" ,"BABU3": r"BABU_{5}","BABU4": r"BABU_{7}","BABU5": r"BABU_{10}","BABU6": r"BABU_{20}"   }

colors = {"baseline":"$B$","NU":"#D55E00", "SL":"#CC79A7","SL1":"#CC79A7", "SL2":"#CC79A7","SL3":"#CC79A7",
            "SL4": "#CC79A7", "SL5": "#CC79A7",
            "TR":"#009E73","TR1":"#009E73", "TR2":"#009E73","TR3":"#009E73",
            "TR4":"#009E73", "TR5":"#009E73","TR6":"#009E73",
            "TR7":"#009E73", "TR8":"#009E73","TR9":"#009E73",
            "ABU":"#F0E442",
            "BABU": "#E69F00", "BABU1": "#E69F00","BABU2": "#E69F00" ,"BABU3": "#E69F00","BABU4": "#E69F00","BABU5": "#E69F00","BABU6": "#E69F00"}

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
                       "baseline": BaseLineTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        "NU": NaiveUpdate(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        "TR1": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0),
                        "TR2": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.05),
                        "TR3": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.1),
                        "SL1": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
                        "SL2": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
                        "SL3": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
                        "SL4": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
                        "SL5": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
                        "ABU": AbuTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        "BABU1": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=1),
                        "BABU2": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=3),
                        "BABU3": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=5),
                        "BABU4": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=7),
                        "BABU5": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=10),
                        "BABU6": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=20),
                }
stability_all = {name:[] for name in models.keys()}
standard_stability_all= {name:[] for name in models.keys()}
mse_all= {name:[] for name in models.keys()}
for ds,target, feature in zip(datasets,targets, features):
    # if ds ==  "Wage":#ds != "Boston" and 
    #     continue
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
    print(X.shape)
    if ds in ["College","Hitters","Wage"]:
       y = np.log(y)
   
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
        # initial model 
        criterion = "mse"

        
        models = {  
                       "baseline": BaseLineTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        # "NU": NaiveUpdate(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        # "TR1": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0),
                        # "TR2": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.05),
                        # "TR3": TreeReevaluation(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,delta=0.05,alpha=0.1),
                        # "SL1": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.1),
                        # "SL2": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.25),
                        # "SL3": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.5),
                        # "SL4": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.75),
                        # "SL5": StabilityRegularization(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,gamma=0.9),
                         "ABU": AbuTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True),
                        #"BABU1": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=1),
                        # "BABU2": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=3),
                         "BABU3": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=5),
                        # "BABU4": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=7),
                        # "BABU5": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=10),
                        # "BABU6": BABUTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,bumping_iterations=20),
                }
        
   
        for name, model in models.items():
            # print(name)
            # print(X1.shape, y1.shape)
            # has_nan = np.isnan(y1).any()
            # print("Array contains NaN:", has_nan)
            # has_inf = np.isinf(y1).any()
            # print("Array contains infinite values:", has_inf)

            # has_nan = np.isnan(X1).any()
            # print("Array contains NaN:", has_nan)
            # has_inf = np.isinf(X1).any()
            # print("Array contains infinite values:", has_inf)
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
            orig_stability[name].append(S1(pred1_orig,pred2_orig))
            orig_standard_stability[name].append(S2(pred1_orig,pred2_orig))

            train_mse[name].append(mean_squared_error(pred2_train,y_12))
            train_stability[name].append(S1(pred1_train,pred2_train))
            train_standard_stability[name].append(S2(pred1_train,pred2_train))
            mse[name].append(mean_squared_error(y_test,pred2))
            stability[name].append(S1(pred1,pred2))
            standard_stability[name].append(S2(pred1,pred2))
        

    print(ds)
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
        if name != "baseline" and name!= "standard":
            x_abs =  np.mean((mse[name]))
            y_abs = np.mean(standard_stability[name])
            x_abs_se = loss_SE
            y_abs_se =stability_SE
            x_se  = loss_SE_norm
            y_se  = stability_SE_norm
            x_r = x_abs/mse_scale
            y_r = y_abs/S_scale
            plot_info[ds].append((ds,x_r,y_r,colors[name],markers[name], x_abs,y_abs,x_se, y_se, x_abs_se, y_abs_se ))
    print()

from matplotlib import pyplot as plt
plt.rcParams.update({'figure.autolayout': True})

from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
# create figure and axes
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8.27, 11),dpi=500)#
axes = axes.ravel()
plt.rcParams.update(plot_params)

# college_box_plot_info = []
# import itertools
# import os
# df_list = list(itertools.chain(*plot_info.values()))
# df = pd.DataFrame(df_list, columns=["dataset",'loss', 'stability', 'color', "marker", 'loss_abs','stability_abs','loss_se','stability_se','loss_abs_se','stability_abs_se' ] )
# if os.path.isfile('results/tree_ISLR_results.csv'):
#     old_df =pd.read_csv('results/tree_ISLR_results.csv')
#     for i,(d,m) in enumerate(zip(df.dataset, df.marker)):
#         index = old_df.loc[(old_df["dataset"] == d) & (old_df["marker"] ==m)].index
#         values  = df.iloc[i]
#         if len(index)>0:
#             old_df.iloc[index]=values
#         else:
#             print(values)
#             old_df  = old_df.append(values, ignore_index=True)

#     old_df.to_csv('results/tree_ISLR_results.csv', index=False)
# else:
#      df.to_csv('results/tree_ISLR_results.csv', index=False)
