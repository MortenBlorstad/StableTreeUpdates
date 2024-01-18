import numpy as np
import pandas as pd
from stabletrees import AbuTree,BABUTree,BaseLineTree
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
EPSILON = 0.000001


plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}
fig = plt.figure(dpi=500,figsize=(6.67*1.5*2/3, 4*1.5/2))

plt.rcParams.update(plot_params),

np.random.seed(0)
n = 100
plt.subplot(1,2,1)
X = np.random.uniform(size=(n,1),low=0,high=4)
y = np.random.normal(loc=X.ravel(),scale=1,size = n)
parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [1,5], "ccp_alpha" : [0,0.01]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters).fit(X,y)
t = BaseLineTree(adaptive_complexity=True).fit(X,y)
params = clf.best_params_
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', ls='-',lw=2, label = "Adaptive Tree Complexity"),
                   Line2D([0], [0], color='orange', ls='--',lw=2, label = "Grid Search CV")]
plt.scatter(X,y, s= 2)
plt.xlabel(r"$\mathbf{x}$",fontsize=10)
plt.ylabel(r"$y$",fontsize=10)
plt.plot(np.sort(X,axis=0), t.predict(np.sort(X,axis=0)),c = "r", linewidth=3, label = "Adaptive Tree Complexity")
plt.plot(np.sort(X,axis=0), clf.predict(np.sort(X,axis=0)),c = "orange",linewidth=2,linestyle='dashed', label = "Grid Search CV")
plt.legend(handles=legend_elements, loc='upper left',fontsize=8)


np.random.seed(0)
n = 1000
plt.subplot(1,2,2)
X = np.random.uniform(size=(n,1),low=0,high=4)
y = np.random.normal(loc=X.ravel(),scale=1,size = n)
parameters = {'max_depth':[None, 5, 10],"min_samples_leaf": [1,5], "ccp_alpha" : [0,0.01]} # , 
clf = GridSearchCV(DecisionTreeRegressor(random_state=0), parameters).fit(X,y)
t = BaseLineTree(adaptive_complexity=True).fit(X,y)
params = clf.best_params_

plt.scatter(X,y, s= 2)
plt.xlabel(r"$\mathbf{x}$",fontsize=10)
plt.ylabel(r"$y$",fontsize=10)
plt.plot(np.sort(X,axis=0), t.predict(np.sort(X,axis=0)),c = "r", linewidth=3, label = "Adaptive Tree Complexity")
plt.plot(np.sort(X,axis=0), clf.predict(np.sort(X,axis=0)),c = "orange",linewidth=2,linestyle='dashed', label = "Grid Search CV")
plt.legend(handles=legend_elements, loc='upper left',fontsize=8)
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\adaptive_vs_hyperparameters.png",bbox_inches='tight')
plt.close()