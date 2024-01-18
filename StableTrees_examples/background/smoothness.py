# create a meshgrid of points


from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error

SEED = 0
n = 1000
np.random.seed(SEED)
X = np.random.uniform(low=0,high=4,size=(n,2))
y = np.random.normal(loc=X[:,0]+X[:,1],scale=1,size=n)
tree = BaseLineTree(criterion="mse",adaptive_complexity=True,max_depth=3).fit(X,y)
node = tree.root

fig = plt.figure(figsize=(8, 8))
x = X[:, 0]
z = X[:, 1]
x_pred = np.linspace(0, 4, 50)   # range of porosity values
z_pred = np.linspace(0, 4, 50)  # range of brittleness values
xx_pred, zz_pred = np.meshgrid(x_pred, z_pred)
model_viz = np.array([xx_pred.flatten(), zz_pred.flatten()]).T


#plt.style.use('default')

tree = BaseLineTree(criterion="mse",adaptive_complexity=True,max_depth=3).fit(X,y)
ax1 = fig.add_subplot(111, projection='3d')




samples = 10
predicted = tree.predict(model_viz)#tree.predict(model_viz)
axes = [ax1]
print(xx_pred.shape,zz_pred.shape, predicted.reshape(50,50).shape)
for ax in axes:
    ax.scatter(x, z, y, facecolor=(0,0,0,0), s=20, alpha=0.75,edgecolor='#70b3f0')
    ax.plot_surface(xx_pred, zz_pred, predicted.reshape(50,50), edgecolor='r',alpha=0.1, rcount=20,ccount=20)
    ax.set_xlabel('$X_1$', fontsize=10)
    plt.xticks(np.arange(0, 4.1, step=1))
    ax.set_ylabel('$X_2$', fontsize=10)
    ax.set_zlabel('y', fontsize=10)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')



ax1.view_init(elev=30, azim=-120)


#fig.tight_layout()
plt.show()