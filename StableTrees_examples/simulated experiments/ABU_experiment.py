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

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)

np.random.seed(0)
n = 1000 # data size
B = 1000 # number of simulations
loss_d1 = []
loss_d2 = []
loss_d12 = []
loss_abu = []

stability_base = []
stability_abu = []
stability_t2 = []
plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}


for b in range(B):
    X = np.random.uniform(size=(n,2),low=0,high=4)
    y = np.random.normal(loc=X[:,0]**2+ X[:,1],scale=3,size = n)
    X_test = np.random.uniform(size=(n,2),low=0,high=4)
    y_test = np.random.normal(loc=X_test[:,0]**2+ X_test[:,1],scale=3,size = n)
    #X12,X_post,y12,y_post = train_test_split(X,y,test_size=0.3333,random_state=b)
    X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=b)
    # x_add = np.random.uniform(size=(n//2,2),low=0,high=4)
    # y_add = np.random.normal(loc=x_add[:,0]**2+ x_add[:,1],scale=1,size = n//2)
    X_post = np.vstack((X ,X1 ))
    y_post = np.concatenate((y,y1))
    # X_post = np.vstack((X ,x_add ))
    # y_post = np.concatenate((y,y_add))
    #print(X_post.shape, y_post.shape)
    tree1 = BaseLineTree(criterion="mse", adaptive_complexity=True).fit(X1,y1) #prior tree
    tree2 = BaseLineTree(criterion="mse", adaptive_complexity=True).fit(X,y) #tree only on D2
    tree3 = BaseLineTree(criterion="mse", adaptive_complexity=True).fit(X_post,y_post)   #posterior tree
    tree4 = AbuTree(criterion="mse", adaptive_complexity=True).fit(X1,y1) #ABU
    tree4.update(X,y)

    # stability_base.append(S2(tree1.predict(X),tree3.predict(X)))
    # stability_t2.append(S2(tree1.predict(X),tree2.predict(X)))
    # stability_abu.append(S2(tree1.predict(X),tree4.predict(X)))
    # loss_d1.append(mean_squared_error(y,tree1.predict(X)))
    # loss_d2.append(mean_squared_error(y,tree2.predict(X)))
    # loss_d12.append(mean_squared_error(y,tree3.predict(X)))
    # loss_abu.append(mean_squared_error(y,tree4.predict(X)))

    # stability_base.append(S2(tree1.predict(X_post),tree3.predict(X_post)))
    # stability_t2.append(S2(tree1.predict(X_post),tree2.predict(X_post)))
    # stability_abu.append(S2(tree1.predict(X_post),tree4.predict(X_post)))
    # loss_d1.append(mean_squared_error(y_post,tree1.predict(X_post)))
    # loss_d2.append(mean_squared_error(y_post,tree2.predict(X_post)))
    # loss_d12.append(mean_squared_error(y_post,tree3.predict(X_post)))
    # loss_abu.append(mean_squared_error(y_post,tree4.predict(X_post)))

    stability_base.append(S2(tree1.predict(X_test),tree3.predict(X_test)))
    stability_t2.append(S2(tree1.predict(X_test),tree2.predict(X_test)))
    stability_abu.append(S2(tree1.predict(X_test),tree4.predict(X_test)))
    loss_d1.append(mean_squared_error(y_test,tree1.predict(X_test)))
    loss_d2.append(mean_squared_error(y_test,tree2.predict(X_test)))
    loss_d12.append(mean_squared_error(y_test,tree3.predict(X_test)))
    loss_abu.append(mean_squared_error(y_test,tree4.predict(X_test)))

#overlap 
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(6.67*1.5*2/3, 4*1.5/2), dpi = 500)
plt.rcParams.update(plot_params)
sns.kdeplot(loss_d1,label=r'$P(w|\mathcal{D}_1)$',c = "#1F77B4" ,ax=ax1, lw = 0.75)
sns.kdeplot(loss_d2,label=r'$P(w|\mathcal{D}_2)$', c = "orange",ax=ax1, lw = 0.75)
sns.kdeplot(loss_d12,label=r'$P(w|\mathcal{D}_1 \cup \mathcal{D}_2)$', c = "#2CA02C",ax=ax1, lw = 0.75)
sns.kdeplot(loss_abu,label='ABU', c = "#F0E442",ax=ax1, lw = 0.75)
ax1.set_xlabel("mse",fontsize=10)
ax1.legend(loc = "upper right",fontsize=6)


sns.kdeplot(stability_t2,label=r'$P(w|\mathcal{D}_2)$', c = "orange",ax=ax2, lw = 0.75)
sns.kdeplot(stability_base,label=r'$P(w|\mathcal{D}_1 \cup \mathcal{D}_2)$',  c = "#2CA02C",ax=ax2, lw = 0.75)
sns.kdeplot(stability_abu,label='ABU', c = "#F0E442",ax=ax2, lw = 0.75)
ax2.set_xlabel("stability",fontsize=10)
ax2.legend(loc = "upper right",fontsize=6)
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\ABU_experiment_13.05_overlap.png",bbox_inches='tight')
plt.close()


# # non-overlap 
np.random.seed(0)
n = 1000 # data size
B = 1000 # number of simulations
loss_d1 = []
loss_d2 = []
loss_d12 = []
loss_abu = []

stability_base = []
stability_abu = []
stability_t2 = []
plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}

for b in range(B):
    X = np.random.uniform(size=(n,2),low=0,high=4)
    y = np.random.normal(loc=X[:,0]**2+ X[:,1],scale=3,size = n)
    X_test = np.random.uniform(size=(n,2),low=0,high=4)
    y_test = np.random.normal(loc=X_test[:,0]**2+ X_test[:,1],scale=3,size = n)

    # X = np.random.uniform(size=(n,1),low=0,high=4)
    # y = np.random.normal(loc=X[:,0],scale=1,size = n)
    # X_test = np.random.uniform(size=(n,1),low=0,high=4)
    # y_test = np.random.normal(loc=X_test[:,0],scale=1,size = n)
    #X12,X_post,y12,y_post = train_test_split(X,y,test_size=0.3333,random_state=b)
    X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=b)
    # x_add = np.random.uniform(size=(n//2,2),low=0,high=4)
    # y_add = np.random.normal(loc=x_add[:,0]**2+ x_add[:,1],scale=1,size = n//2)
    # X_post = np.vstack((X ,X1 ))
    # y_post = np.concatenate((y,y1))
    # X_post = np.vstack((X ,x_add ))
    # y_post = np.concatenate((y,y_add))
    #print(X_post.shape, y_post.shape)
    tree1 = BaseLineTree(criterion="mse", adaptive_complexity=True).fit(X1,y1) #prior tree
    tree2 = BaseLineTree(criterion="mse", adaptive_complexity=True).fit(X2,y2) #tree only on D_delta_t
    tree3 = BaseLineTree(criterion="mse", adaptive_complexity=True).fit(X,y)   #posterior tree
    tree4 = AbuTree(criterion="mse", adaptive_complexity=True).fit(X1,y1) #ABU
    tree4.update(X2,y2)

    # stability_base.append(S2(tree1.predict(X),tree3.predict(X)))
    # stability_t2.append(S2(tree1.predict(X),tree2.predict(X)))
    # stability_abu.append(S2(tree1.predict(X),tree4.predict(X)))
    # loss_d1.append(mean_squared_error(y,tree1.predict(X)))
    # loss_d2.append(mean_squared_error(y,tree2.predict(X)))
    # loss_d12.append(mean_squared_error(y,tree3.predict(X)))
    # loss_abu.append(mean_squared_error(y,tree4.predict(X)))

    # stability_base.append(S2(tree1.predict(X_post),tree3.predict(X_post)))
    # stability_t2.append(S2(tree1.predict(X_post),tree2.predict(X_post)))
    # stability_abu.append(S2(tree1.predict(X_post),tree4.predict(X_post)))
    # loss_d1.append(mean_squared_error(y_post,tree1.predict(X_post)))
    # loss_d2.append(mean_squared_error(y_post,tree2.predict(X_post)))
    # loss_d12.append(mean_squared_error(y_post,tree3.predict(X_post)))
    # loss_abu.append(mean_squared_error(y_post,tree4.predict(X_post)))

    stability_base.append(S2(tree1.predict(X_test),tree3.predict(X_test)))
    stability_t2.append(S2(tree1.predict(X_test),tree2.predict(X_test)))
    stability_abu.append(S2(tree1.predict(X_test),tree4.predict(X_test)))
    loss_d1.append(mean_squared_error(y_test,tree1.predict(X_test)))
    loss_d2.append(mean_squared_error(y_test,tree2.predict(X_test)))
    loss_d12.append(mean_squared_error(y_test,tree3.predict(X_test)))
    loss_abu.append(mean_squared_error(y_test,tree4.predict(X_test)))

# # non-overlap 
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(6.67*1.5*2/3, 4*1.5/2), dpi = 500)
plt.rcParams.update(plot_params)
sns.kdeplot(loss_d1,label=r'$P(w|\mathcal{D}_1)$',c = "#1F77B4",ax = ax1 , lw = 0.75)
sns.kdeplot(loss_d2,label=r'$P(w|\mathcal{D}_{\Delta_t})$', c = "orange", ax=ax1, lw = 0.75)
sns.kdeplot(loss_d12,label=r'$P(w|\mathcal{D}_2)$', c = "#2CA02C",ax=ax1, lw = 0.75)
sns.kdeplot(loss_abu,label='ABU', c = "#F0E442", ax=ax1, lw = 0.75)
ax1.set_xlabel("mse",fontsize=10)
ax1.legend(loc = "upper right",fontsize=6)
plt.legend()


sns.kdeplot(stability_t2,label=r'$P(w|\mathcal{D}_{\Delta_t})$', c = "orange", ax = ax2, lw = 0.75)
sns.kdeplot(stability_base,label=r'$P(w|\mathcal{D}_2)$', c = "#2CA02C", ax = ax2, lw = 0.75)
sns.kdeplot(stability_abu,label='ABU', c = "#F0E442", ax = ax2, lw = 0.75)
ax2.set_xlabel("stability",fontsize=10)
ax2.legend(loc = "upper right",fontsize=6)
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\ABU_experiment_13.05_nonoverlap.png",bbox_inches='tight')
plt.close()


