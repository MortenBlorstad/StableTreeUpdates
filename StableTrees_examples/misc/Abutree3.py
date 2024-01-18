from _stabletrees import RandomForest
from stabletrees import BaseLineTree
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_regression
np.random.seed(0)
n = 500
import time

from matplotlib import pyplot as plt
X = np.random.uniform(size=(n,2),low=0,high=4)
y = np.random.normal(loc=X[:,0]+X[:,1] + X[:,1]*X[:,0],scale=1,size = n)
#X,y = make_regression(n,2,2,noise=100)
X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)


rf2 = RandomForestRegressor(100)
rf = RandomForest(0,100,10000,5,5,True,2)
tree = BaseLineTree(adaptive_complexity=True).fit(X,y)
t = 0
# for i in range(10):
#     start = time.time()
#     rf.learn(X,y)
#     end = time.time()
#     diff =  end - start
#     print(diff)
#     t += diff
# print(t/10) # 1.6359375 1.7265625 1.765625



from sklearn import linear_model

rf.learn(X,y)
rf2.fit(X,y)
# X = np.sort(X,axis=0)
# rf_pred = rf.predict(X)
# rf2_pred = rf2.predict(X)

x = X[:, 0]
z = X[:, 1]
rf = RandomForest(0,100,10000,5,5,True,1)
rf.learn(X,y)
models = {"linear":linear_model.LinearRegression().fit(X, y), "tree":BaseLineTree(adaptive_complexity=True).fit(X,y), "rf":rf  }



# create a meshgrid of points
x_pred = np.linspace(0, 4, 50)   # range of porosity values
z_pred = np.linspace(0, 4, 50)  # range of brittleness values
xx_pred, zz_pred = np.meshgrid(x_pred, z_pred)
model_viz = np.array([xx_pred.flatten(), zz_pred.flatten()]).T

predicted = rf.predict(model_viz)
plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax,(key,model) in zip(axes,models.items()):
    predicted = model.predict(model_viz)
    ax.scatter(x, z, y, facecolor=(0,0,0,0), s=5, alpha=0.5,edgecolor='#70b3f0')
    ax.scatter(xx_pred.flatten(), zz_pred.flatten(), predicted, facecolor=(0,0,0,0), s=2, edgecolor='r',alpha=0.75)
    ax.set_title(key)
    ax.set_xlabel('X_1', fontsize=10)
    ax.set_ylabel('X_2', fontsize=10)
    ax.set_zlabel('y', fontsize=10)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')



ax1.view_init(elev=10, azim=120)
ax2.view_init(elev=10, azim=120)
ax3.view_init(elev=10, azim=120)

fig.tight_layout()
plt.show()

# plt.subplot(1,2,1)
# plt.scatter(X[:,0],y, alpha = 0.1)

# plt.plot(X[:,0],rf_pred[:],c ="red")
# plt.title("my rf")

# plt.subplot(1,2,2)
# plt.scatter(X[:,0],y, alpha = 0.1)

# plt.plot(X[:,0],rf2_pred[:],c ="red")
# plt.title("Sklearn's rf")
# plt.show()


#1.70625 1.7109375


#0.0875