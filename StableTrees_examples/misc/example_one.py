from stabletrees import BaseLineTree,AbuTree,NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,SklearnTree,BABUTreeI
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,mean_poisson_deviance
from sklearn.model_selection import train_test_split,GridSearchCV,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import datapreprocess
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

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

colors = {"Boston":"orange", "Carseats": "g", "College":"r", "Hitters":"c", "Wage":"m"}
#markers = {"baseline":"o", "NU": "v", "TR":"^", "TR":"s", "SL":"D","ABU":"+", "BABU": "*" }
markers = {"baseline":"$B$", "NU": "$NU$", "TR":"$TR$", "TR":"$TR$", "SL":"$SL$", "SL1":"$SL1$", "SL2":"$SL2$","ABU":"$ABU$", "BABU": "$BABU$" }


models = {  
            "baseline": BaseLineTree(),
            #"GLM": LinearRegression(),
            #"sklearn": SklearnTree(),
            "NU": NaiveUpdate(),
            "TR":TreeReevaluation(delta=0.1),
            "SL":StabilityRegularization(),
            "SL1":StabilityRegularization(),
            "SL2":StabilityRegularization(),
            "ABU":AbuTree(),
            "BABU": BABUTree(),
            }
stability_all = {name:[] for name in models.keys()}
standard_stability_all= {name:[] for name in models.keys()}
mse_all= {name:[] for name in models.keys()}
color = 0
fig, ax = plt.subplots()
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
    if ds in ["College","Hitters", "Wage"]:
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
        clf.fit(X1,y1)
        params = clf.best_params_
        # initial model 
        criterion = "mse"
        models = {  
                "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                #"GLM": LinearRegression(),
                #"sklearn": SklearnTree(criterion = criterion,**params),
                #"NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,),
                #"TR":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, delta=0.25),
                "SL":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.75),
                #"SL1":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.25),
                #"SL2":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.5),
                #"ABU":AbuTree(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True),
                #"BABU": BABUTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,bumping_iterations=5)
                
                #  "baseline": BaseLineTree(**params), 
                # "NU": StableTree0(**params),
                #  "TR":TreeReevaluation(criterion = criterion,**params, delta=0.1)
                #  #"SR":StableTree2(**params,lmda=0.5),
                #  "ABU":AbuTreeI(**params)
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
        orig_mse_scale = np.mean(orig_mse["baseline"]); orig_S1_scale = np.mean(orig_stability["baseline"]); orig_S2_scale = np.mean(orig_standard_stability["baseline"]);
        train_mse_scale = np.mean(train_mse["baseline"]); train_S1_scale = np.mean(train_stability["baseline"]); train_S2_scale = np.mean(train_standard_stability["baseline"]);
        mse_scale = np.mean(mse["baseline"]); S1_scale = np.mean(stability["baseline"]); S2_scale = np.mean(standard_stability["baseline"]);
        
        print(f"orig - mse: {np.mean(orig_mse[name]):.3f} ({np.mean(orig_mse[name])/orig_mse_scale:.2f}), stability: {np.mean(orig_stability[name]):.3f} ({np.mean(orig_stability[name])/orig_S1_scale:.2f}), standard stability: {np.mean(orig_standard_stability[name]):.3f} ({np.mean(orig_standard_stability[name])/orig_S2_scale:.2f})")
        print(f"train - mse: {np.mean(train_mse[name]):.3f} ({np.mean(train_mse[name])/train_mse_scale:.2f}), stability: {np.mean(train_stability[name]):.3f} ({np.mean(train_stability[name])/train_S1_scale:.2f}), standard stability: {np.mean(train_standard_stability[name]):.3f} ({np.mean(train_standard_stability[name])/train_S2_scale:.2f})")
        print(f"test - mse: {np.mean(mse[name]):.3f} ({np.mean(mse[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S1_scale:.2f}), standard stability: {np.mean(standard_stability[name]):.3f} ({np.mean(standard_stability[name])/S2_scale:.2f})")
        print("="*80)
        mse_all[name] += [score/mse_scale for score in mse[name]]
        if name != "baseline":

            plt.text(x = np.mean(mse[name])/mse_scale, y = np.mean(standard_stability[name])/S2_scale, s = markers[name], fontsize=12,ha='center',c =colors[ds]) 
            plt.scatter(np.mean(mse[name])/mse_scale, np.mean(standard_stability[name])/S2_scale, c = colors[ds], marker="none",s = 700)
        stability_all[name] += [score/S1_scale for score in stability[name]]
        standard_stability_all[name] += [score/S2_scale for score in standard_stability[name]]
    print()
    color +=1 
# from matplotlib.lines import Line2D
# legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
#                           markerfacecolor=v, markersize=14) for k,v in colors.items()  ]
# legend_elements = [Line2D([0], [0], color='b', lw=1, label='baseline', linestyle = "--")] +legend_elements
# plt.axvline(x=1, linestyle = "--")
# plt.axhline(y=1, linestyle = "--")
# plt.xlabel("mse",fontsize=14)
# plt.ylabel(r'$\left(f_1(x_i)-f_2(x_i)\right)^2$',fontsize=14)
# plt.ylim((0,1.25))
# #plt.xlim((0,2))
# ax.legend(handles=legend_elements)
# plt.show()
# from matplotlib import pyplot as plt
# plt.rcParams.update({'figure.autolayout': True})
# labels, data = stability_all.keys(), stability_all.values()

# plt.boxplot(data,vert=0)
# plt.yticks(range(1, len(labels) + 1), labels,fontsize=16)
# plt.xlabel(r'$\log\left(\frac{f_1(x_i)}{f_2(x_i)}\right)$',fontsize=14)
# plt.ylabel("models")
# plt.savefig(f"StableTrees_examples\plots\\all_stability_ac.png")
# plt.close()


# labels, data = standard_stability_all.keys(), standard_stability_all.values()
# plt.boxplot(data,vert=0)
# plt.yticks(range(1, len(labels) + 1), labels,fontsize=16)
# plt.xlabel(r'$\left(f_1(x_i)-f_2(x_i)\right)^2$',fontsize=14)
# plt.ylabel("models")
# plt.savefig(f"StableTrees_examples\plots\\all_standard_stability_ac.png")
# plt.close()


# labels, data = mse_all.keys(), mse_all.values()
# plt.boxplot(data,vert=0)
# plt.xlabel('mse',fontsize=14)
# plt.ylabel("models")
# plt.yticks(range(1, len(labels) + 1), labels,fontsize=16)
# plt.savefig(f"StableTrees_examples\plots\\all_mse_ac.png")
# plt.close()

# SEED = 0
# EPSILON = 2
# kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)


# def S1(pred1, pred2):
#     return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

# def S2(pred1, pred2):
#     return np.mean((pred1- pred2)**2)

# if ds == "College":
#    y = np.log(y)

# initial model 
# n = 1000
# np.random.seed(SEED)
# X = np.random.uniform(low=0,high=4,size=(n,2))
# y = np.random.normal(loc=X[:,0]+X[:,1],scale=1,size=n)


# stability = {name:[] for name in models.keys()}
# standard_stability = {name:[] for name in models.keys()}
# mse = {name:[] for name in models.keys()}
# train_stability = {name:[] for name in models.keys()}
# train_standard_stability = {name:[] for name in models.keys()}
# train_mse = {name:[] for name in models.keys()}
# orig_stability = {name:[] for name in models.keys()}
# orig_standard_stability = {name:[] for name in models.keys()}

# stability_all = {name:[] for name in models.keys()}
# standard_stability_all= {name:[] for name in models.keys()}
# mse_all= {name:[] for name in models.keys()}
# orig_mse = {name:[] for name in models.keys()}
# criterion = "mse"
# models = {  
#             "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
#             "GLM": LinearRegression(),
#             "sklearn": SklearnTree(criterion = criterion,**params),
#             "NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,),
#             "TR":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, delta=0.5),
#             "SL":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,lmbda=0.75),
#             "ABU":AbuTreeI(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True),
#             "BABU": BABUTree(criterion = criterion, adaptive_complexity=True)
#             }
# for train_index, test_index in kf.split(X):
#     X_12, y_12 = X[train_index],y[train_index]
#     X_test,y_test = X[test_index],y[test_index]
#     X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
#     initial model 
#     criterion = "mse"
#     models = {  
#             "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
#             "GLM": LinearRegression(),
#             "sklearn": SklearnTree(criterion = criterion,**params),
#             "NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,),
#             "TR":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, delta=0.5),
#             "SL":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,lmbda=0.75),
#             "ABU":AbuTreeI(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True),
#             "BABU": BABUTree(criterion = criterion, adaptive_complexity=True)
#             }
#     for name, model in models.items():
#         model.fit(X1,y1)
#         pred1 = model.predict(X_test)
#         pred1_train = model.predict(X_12)
#         pred1_orig= model.predict(X1)
#         print("before")
#         if name =="GLM":
#             model.fit(X_12,y_12)
#         else:
#             model.update(X_12,y_12)
        
#         print("after")
#         pred2 = model.predict(X_test)
#         pred2_orig= model.predict(X1)
#         pred2_train =  model.predict(X_12)

#         orig_mse[name].append(mean_squared_error(pred2_orig,y1))
#         orig_stability[name].append(S1(pred1_orig,pred2_orig))
#         orig_standard_stability[name].append(S2(pred1_orig,pred2_orig))

#         train_mse[name].append(mean_squared_error(pred2_train,y_12))
#         train_stability[name].append(S1(pred1_train,pred2_train))
#         train_standard_stability[name].append(S2(pred1_train,pred2_train))
#         mse[name].append(mean_squared_error(y_test,pred2))
#         stability[name].append(S1(pred1,pred2))
#         standard_stability[name].append(S2(pred1,pred2))
    


# for name in models.keys():
#     print("="*80)
#     print(f"{name}")
#     orig_mse_scale = np.mean(orig_mse["baseline"]); orig_S1_scale = np.mean(orig_stability["baseline"]); orig_S2_scale = np.mean(orig_standard_stability["baseline"]);
#     train_mse_scale = np.mean(train_mse["baseline"]); train_S1_scale = np.mean(train_stability["baseline"]); train_S2_scale = np.mean(train_standard_stability["baseline"]);
#     mse_scale = np.mean(mse["baseline"]); S1_scale = np.mean(stability["baseline"]); S2_scale = np.mean(standard_stability["baseline"]);
    
#     print(f"orig - mse: {np.mean(orig_mse[name]):.3f} ({np.mean(orig_mse[name])/orig_mse_scale:.2f}), stability: {np.mean(orig_stability[name]):.3f} ({np.mean(orig_stability[name])/orig_S1_scale:.2f}), standard stability: {np.mean(orig_standard_stability[name]):.3f} ({np.mean(orig_standard_stability[name])/orig_S2_scale:.2f})")
#     print(f"train - mse: {np.mean(train_mse[name]):.3f} ({np.mean(train_mse[name])/train_mse_scale:.2f}), stability: {np.mean(train_stability[name]):.3f} ({np.mean(train_stability[name])/train_S1_scale:.2f}), standard stability: {np.mean(train_standard_stability[name]):.3f} ({np.mean(train_standard_stability[name])/train_S2_scale:.2f})")
#     print(f"test - mse: {np.mean(mse[name]):.3f} ({np.mean(mse[name])/mse_scale:.2f}), stability: {np.mean(stability[name]):.3f} ({np.mean(stability[name])/S1_scale:.2f}), standard stability: {np.mean(standard_stability[name]):.3f} ({np.mean(standard_stability[name])/S2_scale:.2f})")
#     print("="*80)
#     mse_all[name] += [score/mse_scale for score in mse[name]]
#     stability_all[name] += [score/S1_scale for score in stability[name]]
#     standard_stability_all[name] += [score/S2_scale for score in standard_stability[name]]
# print()

# plt.rcParams.update({'figure.autolayout': True})
# labels, data = stability_all.keys(), stability_all.values()

# plt.boxplot(data,vert=0)
# plt.yticks(range(1, len(labels) + 1), labels,fontsize=16)
# plt.xlabel(r'$\log\left(\frac{f_1(x_i)}{f_2(x_i)}\right)$',fontsize=14)
# plt.ylabel("models")
# plt.savefig(f"StableTrees_examples\plots\\sim_stability_ac.png",facecolor='white', transparent=False)
# plt.close()


# labels, data = standard_stability_all.keys(), standard_stability_all.values()
# plt.boxplot(data,vert=0)
# plt.yticks(range(1, len(labels) + 1), labels,fontsize=16)
# plt.xlabel(r'$\left(f_1(x_i)-f_2(x_i)\right)^2$',fontsize=14)
# plt.ylabel("models")
# plt.savefig(f"StableTrees_examples\plots\\sim_standard_stability_ac.png",facecolor='white', transparent=False)
# plt.close()


# labels, data = mse_all.keys(), mse_all.values()
# plt.boxplot(data,vert=0)
# plt.xlabel('mse',fontsize=14)
# plt.ylabel("models")
# plt.yticks(range(1, len(labels) + 1), labels,fontsize=16)
# plt.savefig(f"StableTrees_examples\plots\\sim_mse_ac.png",facecolor='white', transparent=False)
# plt.close()