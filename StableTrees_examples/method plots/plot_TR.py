from sklearn.datasets import load_diabetes
from stabletrees import TreeReevaluation
import numpy as np
from matplotlib import pyplot as plt
from plotter import plot
from sklearn.model_selection import train_test_split
import pandas as pd
import datapreprocess

# X1 = np.array([4,5,6]).reshape(-1,1)
# y1 = np.array([0,1,4])
# X12 = np.array([4,5,6,6]).reshape(-1,1)
# y12 = np.array([0,1,4,5])
# X,y = load_diabetes(return_X_y=True)
targets = ["medv", "Sales", "Apps", "Salary", "wage"]
features =  [[ "crim", "rm"], ["Advertising", "Price"], ["Private", "Accept"], ["AtBat", "Hits"], ["year", "age"] ]

ds = "Carseats"
data = pd.read_csv("data/"+ ds+".csv") # load dataset
    
data = datapreprocess.data_preperation(ds)
print(data)

y = data["Sales"].to_numpy()
X = data.drop("Sales", axis=1).to_numpy()
# X1,X2, y1, y2 = train_test_split(X,y,test_size=0.3,random_state=0)

# t = TreeReevaluation(max_depth=3,delta=0.05,alpha=0.1).fit(X1,y1)

X1,X2, y1, y2 = train_test_split(X,y,test_size=0.5,random_state=0)

t = TreeReevaluation(max_depth=2,delta=0.05,alpha=0.05).fit(X1,y1)

plt.figure(figsize=(4, 4) ,dpi=500)

# plot(t.root)
# plt.tight_layout()
# plt.savefig(f"StableTrees_examples\plots\\method_example_TR_part1.png",bbox_inches='tight')
# plt.close()
t.update(X,y)

plt.figure(figsize=(4, 4) ,dpi=500)
plot(t.root, indices=[2])
plt.tight_layout()
plt.savefig(f"StableTrees_examples\plots\\method_example_TR_part2.png",bbox_inches='tight')
plt.close()

