import numpy as np
import pandas as pd
from stabletrees import AbuTree, BaseLineTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
from matplotlib import pyplot as plt


np.random.seed(0)
n = 1000
X = np.random.uniform(size=(n,3),low=0,high=4)
y = np.random.normal(loc=X[:,0]+X[:,1],scale=1,size = n)
X = np.hstack((X, np.where(y<1.5,0,1).reshape(-1,1)+ np.where(y<1.5,0,1).reshape(-1,1)))
print(X.shape)
print("="*20)
tree = AbuTree(adaptive_complexity=True,min_samples_leaf=5,criterion="mse").fit(X,y)


print("tree",mean_squared_error(y,tree.predict(X)))
print("="*20)
tree2 = AbuTree(max_depth=5,min_samples_leaf=5,criterion="mse").fit(X,y)
print("tree2",mean_squared_error(y,tree2.predict(X)))
print("="*20)
X = X[:,::-1]



tree = AbuTree(adaptive_complexity=True,min_samples_leaf=5,criterion="mse").fit(X,y)
print("tree", mean_squared_error(y,tree.predict(X)))
print("="*20)

tree2 = AbuTree(max_depth=5,min_samples_leaf=5,criterion="mse").fit(X,y)
print("tree2",mean_squared_error(y,tree2.predict(X)))
print("="*20)