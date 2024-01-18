from stabletrees import rnchisq,cir_sim_vec,cir_sim_mat
import numpy as np
from matplotlib import pyplot as plt
np.random.seed(0)
nsims = 100
mat = cir_sim_mat(nsims,100)
print(np.max(mat,axis=1).mean())
print(np.max(mat,axis=1).min())
print(np.max(mat,axis=1).max())
for i in range(nsims):
    plt.plot(mat[i,:])
plt.show()

mat = cir_sim_mat(nsims,100)
print(np.max(mat,axis=1).mean())
print(np.max(mat,axis=1).min())
print(np.max(mat,axis=1).max())
for i in range(nsims):
    plt.plot(mat[i,:])
plt.show()
mat = cir_sim_mat(nsims,100)
print(np.max(mat,axis=1).mean())
print(np.max(mat,axis=1).min())
print(np.max(mat,axis=1).max())
for i in range(nsims):
    plt.plot(mat[i,:])
plt.show()


from stabletrees import BaseLineTree, StableTree2
from sklearn.datasets import make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd


# target =  "Apps"
# feature = ["Private", "Accept"]
# data = pd.read_csv("data/"+ "College"+".csv") # load dataset
# # data preperation
# data = data.dropna(axis=0, how="any") # remove missing values if any
# data = data.loc[:, feature + [target]] # only selected feature and target variable
# cat_data = data.select_dtypes("object") # find categorical features
# if not cat_data.empty: # if any categorical features, one-hot encode them
#     cat_data = pd.get_dummies(data.select_dtypes("object"), prefix=None, prefix_sep="_", dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None)
#     data = pd.concat([data.select_dtypes(['int','float']),cat_data],axis=1)

# print(data.describe())
# y = data[target].to_numpy()
# X = data.drop(target, axis=1).to_numpy()

# #X,y = make_regression(2000, n_features=5, n_informative=5, noise=100, random_state=0) # load_diabetes(return_X_y=True) #

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

# # X_train = np.array([0.5,0.55, 0.75,0.5]).reshape(-1,1)
# # y_train = np.array([100,150,200, 102])
# tree = StableTree2(min_samples_split=5, adaptive_complexity=True).fit(X_train,y_train)


# print(mean_squared_error(y_test, tree.predict(X_test)))
# tree.plot()

# tree = StableTree2(min_samples_split=5, adaptive_complexity=False).fit(X_train,y_train)


# print(mean_squared_error(y_test, tree.predict(X_test)))
# tree.plot()




