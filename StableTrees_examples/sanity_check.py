from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,AbuTree2,AbuTree

from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
pca = PCA(n_components=1)

n =1000
noise =1
X = np.random.uniform(0,4,(n,2))
y = np.random.normal((X).sum(axis=1)**0.9,noise,n)

X_test = np.sort(np.random.uniform(0,4,(n,2)),axis=0)
y_test = np.random.normal((X_test).sum(axis=1)**0.9,noise,n)

X1,_,y1,_ =  train_test_split(X,y,train_size=0.5)
# X,y = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=False)
# X12,X_test,y1,y_test =  train_test_split(X,y,train_size=0.75)
# X1,_,y1,_ =  train_test_split(X,y,train_size=0.5)

pca.fit(X1)
pca1 = pca.transform(X_test)

abu = AbuTree2(adaptive_complexity=True,min_samples_leaf=5)
abu2 = AbuTree2(adaptive_complexity=True,min_samples_leaf=5)
t = BaseLineTree(adaptive_complexity=True,min_samples_leaf=5)

# print( X1[bootstrap_idx], X1)
# print( y1[bootstrap_idx], y1)
abu.fit(X1,y1)
abu2.fit(X1,y1)
t.fit(X1,y1)


fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)

pred1 = abu.predict(X_test)
pred12 = abu2.predict(X_test)
pred_ = t.predict(X_test)
abu.update(X,y)
abu2.update(X,y)
t.fit(X,y)
pred2 = abu.predict(X_test)
pred22 = abu2.predict(X_test)
pred__ = t.predict(X_test)
EPSILON = 5




print("ABU1:")
print(f"loss: {np.sum((y_test-pred2)**2)}")
print(f"stability: {np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))}")
print(f"stability: {np.sum((pred1-pred2)**2)}")
print("\n\nABU2:")
print(f"loss: {np.sum((y_test-pred22)**2)}")
print(f"stability: {np.std(np.log((pred22+EPSILON)/(pred12+EPSILON)))}")
print(f"stability: {np.sum((pred12-pred22)**2)}")
print("\n\nbaseline:")
print(f"loss: {np.sum((y_test-pred__)**2)}")
print(f"stability: {np.std(np.log((pred__+EPSILON)/(pred_+EPSILON)))}")
print(f"stability: {np.sum((pred_-pred__)**2)}")

ax1.scatter(pca1,y_test,c = "red", s = 2, alpha = 0.5)
ax1.set_ylabel("y")
ax1.set_xlabel("pca1")
ax1.set_title("trained at t=0")
ax1.scatter(pca1,pred1,c = "black",s = 3)
ax1.scatter(pca1,pred12,c = "green",s = 3)
ax2.scatter(pca1,y_test,c = "red", s = 2, alpha = 0.5)
ax2.set_ylabel("y")
ax2.set_xlabel("pca1")
ax2.set_title("updated at t=1")
#ax2.scatter(pca1,pred__,c = "blue",s = 3)
ax2.scatter(pca1,pred2,c = "black",s = 3)
ax2.scatter(pca1,pred22,c = "green",s = 2)
plt.show()
