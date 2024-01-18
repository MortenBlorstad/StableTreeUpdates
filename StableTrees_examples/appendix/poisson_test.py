
import numpy as np  
from stabletrees import BaseLineTree,AbuTree,AbuTree, BABUTree,StabilityRegularization
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn.linear_model import PoissonRegressor
n = 5000
np.random.seed(0)
X = np.random.uniform(low=0,high=4,size=(n,1))

y = np.random.poisson(lam=X[:,0]**2+ np.sin(X[:,0]),size=n)

#y = np.random.normal(loc = X.ravel()**2,scale=1,size=  n)
X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)

criterion = "mse"


# lm = PoissonRegressor().fit(X,y)
# t2 =BABUTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X1,y1)
# t1 = BaseLineTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X1,y1)
# t3 = StabilityRegularization(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5,gamma = 0.75).fit(X1,y1)

ind = np.argsort(X, axis=0).ravel()
# pred = t1.predict(X[ind,:])
# # t1.update(X,y)

# print(y.sum())
# print(y1.sum())
# # print(mean_poisson_deviance(y,t.predict(X)))

# # print(mean_poisson_deviance(y,t1.predict(X)))
# # print(pred- pred2)

# #t1.update(X2,y2)
# #pred3 =t1.predict(X)
# t1.update(X,y)

# t3.update(X,y)
# pred4 =t1.predict(X[ind,:])
# pred2 =t2.predict(X[ind,:])
# pred6 =t3.predict(X[ind,:])
# #print(pred)
# t = BaseLineTree(criterion=criterion, adaptive_complexity=True,min_samples_leaf=5).fit(X,y)
# pred5 =t.predict(X[ind,:])
# plt.scatter(X[:,0],y, s= 10)

# plt.scatter(X[ind,0],pred,c ="black",label= "base") #first
# #plt.scatter(X,pred3,c ="orange", s = 8) #t2
# plt.scatter(X[ind,0],pred2,c ="r",label= "BABU",alpha=0.5,s = 8) #combined
# plt.scatter(X[ind,0],pred6,c ="m",label= "SL",s = 8) #combined


# #plt.plot(X[ind,:],pred5,c ="y",label= "base retrained")

# plt.legend()
# plt.show()
# print(pred-pred2)

plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}
#fig = plt.figure(dpi=500,figsize=(11,11/1.61803398875))
fig = plt.figure(dpi=500,figsize=(6.67*1.5*2/3, 4*1.5/2))
plt.rcParams.update(plot_params),

np.random.seed(0)
plt.subplot(1,2,1)
n = 100

np.random.seed(0)
X = np.random.uniform(low=0,high=4,size=(n,1))

y = np.random.poisson(lam=X[:,0],size=n)

#y = np.random.normal(loc = X.ravel()**2,scale=1,size=  n)
X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)

criterion = "mse"
t = BaseLineTree(criterion="poisson", adaptive_complexity=False,max_depth=3,min_samples_leaf=5).fit(X,y)
#parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [1,5], "ccp_alpha" : [0,0.01, 0.1]} # , 
parameters = {'max_depth':[3],"min_samples_leaf": [5]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters).fit(X,y)
params = clf.best_params_
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', ls='-',lw=2, label = "second-order approximation"),
                   Line2D([0], [0], color='orange', ls='--',lw=2, label = "exact")]
plt.scatter(X,y, s= 2)
plt.xlabel(r"$\mathbf{x}$",fontsize=10)
plt.ylabel(r"$y$",fontsize=10)
plt.plot(np.sort(X,axis=0), t.predict(np.sort(X,axis=0)),c = "r", linewidth=3, label = "Adaptive Tree Complexity")
plt.plot(np.sort(X,axis=0), clf.predict(np.sort(X,axis=0)),c = "orange",linewidth=2,linestyle='dashed', label = "exact")
plt.legend(handles=legend_elements, loc='upper left',fontsize=8)

plt.subplot(1,2,2)
n = 1000

np.random.seed(0)
X = np.random.uniform(low=0,high=4,size=(n,1))

y = np.random.poisson(lam=X[:,0],size=n)
weigths = np.random.uniform(low=0,high=1,size=(n,))
t = BaseLineTree(criterion="poisson", adaptive_complexity=False,max_depth=3,min_samples_leaf=5).fit(X,y)
#parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [1,5], "ccp_alpha" : [0,0.01, 0.1]} # , 
parameters = {'max_depth':[3],"min_samples_leaf": [5]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters).fit(X,y)
params = clf.best_params_
print(params)
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', ls='-',lw=2, label = "second-order approximation"),
                   Line2D([0], [0], color='orange', ls='--',lw=2, label = "exact")]
plt.scatter(X,y, s= 2)
plt.xlabel(r"$\mathbf{x}$",fontsize=10)
plt.ylabel(r"$y$",fontsize=10)
plt.plot(np.sort(X,axis=0), t.predict(np.sort(X,axis=0)),c = "r", linewidth=3, label = "second-order approximation")
plt.plot(np.sort(X,axis=0), clf.predict(np.sort(X,axis=0)),c = "orange",linewidth=2,linestyle='dashed', label = "exact")
plt.legend(handles=legend_elements, loc='upper left',fontsize=8)
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\adaptive_vs_hyperparameters_poisson.png",bbox_inches='tight')
plt.close()