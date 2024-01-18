import pandas as pd
import tarfile
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from stabletrees import BaseLineTree,AbuTree,NaiveUpdate,TreeReevaluation,StabilityRegularization, BABUTree
from sklearn.tree import DecisionTreeRegressor
SEED = 0
plt.rcParams["figure.figsize"] = (20,12)


with tarfile.open("data\poisson\\freMTPLfreq.tar.gz", "r:*") as tar:
    csv_path = tar.getnames()[0]
    df = pd.read_csv(tar.extractfile(csv_path), header=0)


df["Frequency"] = df["ClaimNb"] / df["Exposure"]

print(
    "Average Frequency = {}".format(np.average(df["Frequency"], weights=df["Exposure"]))
)
print(
    "Fraction of exposure with zero claims = {0:.1%}".format(
        df.loc[df["ClaimNb"] == 0, "Exposure"].sum() / df["Exposure"].sum()
    )
)

df["Density_binned"] = pd.cut(df.Density, include_lowest=True, bins=[0,40,200,500,4500,np.inf])
df["DriverAge_binned"]  = pd.cut(df.DriverAge , bins=[17,22,26,42,74,np.inf])
df["CarAge_binned"]  = pd.cut(df.CarAge, include_lowest=True , bins=[0,15,np.inf])
df["brandF"] = np.where(df.Brand=="Japanese (except Nissan) or Korean","F","other")
df["Power_glm"] = ["DEF" if p in ["d","e","f"] else "other" if p in ["d","e","f"] else "GH" for p in df.Power ]
df.insert(len(df.columns)-1, 'Frequency', df.pop('Frequency'))


tree_preprocessor = ColumnTransformer(
    [
        ("categorical",
            OrdinalEncoder(),
            ["Brand", "Power", "Gas", "Region"],
        ),
        ("numeric", "passthrough", ["CarAge","DriverAge","Density"]),
    ],
    remainder="drop",
)

glm_preprocessor = ColumnTransformer(
    [
        (
            "categorical",
            OrdinalEncoder(),
            ["CarAge_binned", "DriverAge_binned", "Gas", "Density"],
        ),
        (
            "onehot_categorical",
            OneHotEncoder(),
            ["Gas","Power_glm", "brandF"],
        ),
    ],
    remainder="drop",
)
tree_preprocessor.fit_transform(df)
df_train, df_test = train_test_split(df, test_size=0.5, random_state=0)
df_1, df_2 = train_test_split(df_train, test_size=0.5, random_state=0)

tree_preprocessor.fit_transform(df)
glm_preprocessor.fit_transform(df)

t = AbuTree(criterion="poisson",min_samples_leaf=5, adaptive_complexity=True).fit(tree_preprocessor.transform(df_1),df_1.Frequency, sample_weight=df_1["Exposure"])#
#t = BaseLineTree(criterion="poisson",min_samples_leaf=5, adaptive_complexity=True).fit(tree_preprocessor.transform(df_train),df_train.Frequency)#
#df_train.Frequency = np.maximum(7, df_train.ClaimNb/df_train.Exposure )
# plt.hist(df_train.Frequency, log=True )
# plt.show()
# plt.hist(df_test.Frequency , log=True)
# plt.show()
pred = t.predict(tree_preprocessor.transform(df_test) )
mask = pred > 0
print(all(mask))
print(mean_poisson_deviance(df_test.ClaimNb, pred*df_test.Exposure))
t.update(tree_preprocessor.transform(df_train),df_train.Frequency, sample_weight=df_train["Exposure"])
#t.update(tree_preprocessor.transform(df),df.Frequency)

pred = t.predict(tree_preprocessor.transform(df_test) )
mask = pred > 0
print(all(mask))
print(mean_poisson_deviance(df_test.ClaimNb, pred*df_test.Exposure))

t1 = TreeReevaluation(criterion="poisson",min_samples_leaf=5, adaptive_complexity=True).fit(tree_preprocessor.transform(df_1),df_1.Frequency, sample_weight=df_1["Exposure"])
#t = BaseLineTree(criterion="poisson",min_samples_leaf=5, adaptive_complexity=True).fit(tree_preprocessor.transform(df_train),df_train.Frequency)#
pred = t1.predict(tree_preprocessor.transform(df_test) )
mask = pred > 0
print(all(mask))
print(mean_poisson_deviance(df_test.ClaimNb, pred*df_test.Exposure))
t1.update(tree_preprocessor.transform(df_train),df_train.Frequency, sample_weight=df_train["Exposure"])
#t.update(tree_preprocessor.transform(df),df.Frequency)

pred = t1.predict(tree_preprocessor.transform(df_test) )
mask = pred > 0
print(all(mask))
print(mean_poisson_deviance(df_test.ClaimNb, pred*df_test.Exposure))



# from sklearn.tree import DecisionTreeRegressor
# poisson_t = DecisionTreeRegressor(criterion="poisson",min_samples_leaf=5, max_depth=8)
# parameters = {"ccp_alpha" : [0,0.01,1e-3,1e-4, 1e-5]} # , 
# #parameters = {'max_depth':[ 5],"min_samples_leaf": [5], "ccp_alpha" : [0]} # ,
# clf = GridSearchCV(DecisionTreeRegressor(criterion="poisson",random_state=0), parameters)
# clf.fit(
#     tree_preprocessor.transform(df_train),df_train.Frequency, sample_weight=df_train["Exposure"]) #, sample_weight=df_train["Exposure"]
# print(clf.best_params_)
# pred = clf.predict(tree_preprocessor.transform(df_test) )
# mask = pred > 0
# print(mean_poisson_deviance(df_test.ClaimNb, pred*df_test.Exposure))