
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
# tree.plot()
# plt.show()

index = 1
def _draw_line(node,x_min = 0,x_max=4,y_min=0,y_max=4):
    global index
    x_min_next = x_min
    x_max_next = x_max
    y_min_next = y_min
    y_max_next = y_max
    if node.is_leaf():
        plt.text((x_max+x_min)/2, (y_max+y_min)/2,f"$R_{index}$", fontsize=20,ha='center') 
        index+=1
        return
    dim = node.get_split_feature()
    val = node.get_split_value()
    if dim==0:
        plt.vlines(x = val,ymin=y_min, ymax= y_max)
        x_min_next = val
        x_max_next = val
    else:
        plt.hlines(y = val,xmin=x_min, xmax=x_max)
        y_min_next = val
        y_max_next = val
    
    _draw_line(node.get_left_node(),x_min,x_max_next,y_min ,y_max_next )
    _draw_line(node.get_right_node(),x_min_next,x_max,y_min_next ,y_max)


node = tree.root
fig = plt.figure(figsize=(16, 15))
_draw_line(node)
plt.ylim(0,4)
plt.xlim(0,4)
plt.xticks(np.arange(0, 4.1, step=1))
plt.yticks(np.arange(0, 4.1, step=1))
plt.xlabel('$X_1$', fontsize=16)
plt.ylabel('$X_2$', fontsize=16)
plt.show()