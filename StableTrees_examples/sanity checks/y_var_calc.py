from stabletrees import Tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
np.random.seed(0)




# x,y = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=False)
# x,_,y,_ = train_test_split(x,y,train_size=0.99,shuffle=True,random_state=0)
# 
n = 100000
outlier_factor = 0.05

x = np.random.uniform(-4,4,(n,1)) 
y = x[:,0]  +np.random.normal(0,1,n) + 0.0005*np.random.standard_cauchy(n) # 1/(np.pi*outlier_factor*(1+((x-0)/outlier_factor)**2))

print(np.var(y)/len(y))
# plt.scatter(x,y)

# plt.show()


# mse: 0.9969840492109535
tree = Tree(criterion = "mse",min_samples_leaf=5, adaptive_complexity=True, alpha=0,beta=0.5)
n = 2000
increment = 500

x_test = x[90000:]
y_test = y[90000:]

print("learn:")
print(len(x[:n]))
tree.fit(x[:n],y[:n])
pred = tree.predict(x_test)
print(f"mse: {np.mean((y_test-pred)**2)}")
for t in range(50):
    n+=increment
    print(f"update {t+1}:")
    tree.update(x[:n],y[:n])
    pred = tree.predict(x_test)
    print(f"mse: {np.mean((y_test-pred)**2)}")




    