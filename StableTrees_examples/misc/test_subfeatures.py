from _stabletrees import RandomForest
from stabletrees import StabilityRegularization,BABUTree, TreeReevaluation, AbuTree
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_regression
import datapreprocess
np.random.seed(0)
n = 5000
import time

from matplotlib import pyplot as plt
data = datapreprocess.data_preperation("Boston")
#print(data.corr())

y = data["medv"].to_numpy()
X = data.drop("medv", axis=1).to_numpy()
X.shape[0]
# X = np.random.uniform(size=(n,1),low=0,high=4)
# y = np.random.normal(loc=X[:,0] ,scale=1,size = n)
# # X,y = make_regression(n,n_features= 10, n_informative= 5,noise=100, random_state=0)
# # y = (y  - np.min(y))/(np.max(y) - np.min(y))*10
X1,X2,y1,y2 = train_test_split(X,y,test_size=0.65,random_state=0)
X2,X_test,y2,y_test = train_test_split(X2,y2,test_size=0.25,random_state=0)
# X_test = np.random.uniform(size=(n,1),low=0,high=4)


# y_test = np.random.normal(loc=X_test[:,0],scale=1,size = n)

# X_test, y_test = make_regression(n,n_features= 10, n_informative= 5,noise=100, random_state=13)
# y_test = (y_test - np.min(y_test))/(np.max(y_test) - np.min(y_test))*10
#int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features
ind = np.random.choice(np.arange(0,X.shape[0]),size=X.shape[0],replace=False)
# X = X[ind]
# y  = y[ind]
print(X.shape)
rf2 = RandomForestRegressor(200, min_samples_leaf=5,min_samples_split=5,max_features=X.shape[1],random_state=0,bootstrap = True).fit(X1,y1)
pred1 = rf2.predict(X_test)
rf2.fit(X,y)

pred2 = rf2.predict(X_test)
print(f"sklearn, {np.mean((pred1-pred2)**2):.3f}, {np.mean((y_test-pred2)**2):.3}, {np.mean((y_test-pred1)**2):.3}")

# for i in range(5):
#     rf = RandomForest(0,200,10000,5,5,False,X.shape[1]//3,i)
#     rf.learn(X1,y1)
#     print("learned")
#     pred1 = rf.predict(X_test)
#     print("predict")
#     #rf.update(np.vstack((X1,X2)),np.concatenate((y1,y2),axis=0))
#     rf.update(X,y)
#     print("updated")
#     pred2 = rf.predict(X_test)
#     #print("predict")
#     print(f"{i}, {np.mean((pred1-pred2)**2):.3f}, {np.mean((y_test-pred2)**2):.3}, {np.mean((y_test-pred1)**2):.3}")
#     # plt.scatter(X[:,0],y, alpha = 0.1)

#     # plt.scatter(X[:,0],rf.predict(X),c ="red")
#     # plt.title("tree")
#     # plt.show()
 

tree = TreeReevaluation(adaptive_complexity=True,max_depth=4, max_features=2).fit(X1,y1)
tree.update(X,y)
tree.plot()
plt.show()


#tree.plot()
