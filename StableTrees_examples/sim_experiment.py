from stabletrees import Tree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold

from adjustText import adjust_text



SEED = 0
EPSILON = 1


def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)

plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}

criterion = "mse"
abu = Tree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,alpha=0.2,beta=1.2)
base = Tree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,alpha=0,beta=0)
np.random.seed(SEED)
n = 20000
noise = 4
loss_score =[]
stab_score =[]
bloss_score =[]
bstab_score =[]

x = np.random.uniform(-4,4,(n,1))
y =  + np.random.normal(x.ravel()**2,noise)
x_test = np.random.uniform(-4,4,(10000,1))
y_test = x_test.ravel()**2 +np.random.normal(0,noise,10000)
print("learn")
start = 1000
abu.fit(x[:start],y[:start])
base.fit(x[:start],y[:start])
pred1 = abu.predict(x_test)
pred_b1 = base.predict(x_test)
increment = (n)//20
print(increment)
for n in [1000,2000,5000,10000]:
    np.random.seed(SEED)
    start+=increment
    x = np.random.uniform(-4,4,(n,1))
    y =  + np.random.normal(x.ravel()**2,noise)
    abu.update(x,y)
    base.update(x,y)
    pred2 = abu.predict(x_test)
    pred_b2 = base.predict(x_test)
    
    loss_score.append(np.mean((pred2-y_test)**2))
    stab_score.append(np.mean((pred2-pred1)**2))

    bloss_score.append(np.mean((pred_b2-y_test)**2))
    bstab_score.append(np.mean((pred_b2-pred_b1)**2))
    pred1 = pred2
    pred_b1 = pred_b2

plt.plot(bloss_score,bstab_score,'-o', label = "base", )
plt.plot(loss_score,stab_score,'-o', label = "uwr")
plt.text(loss_score[0],stab_score[0], s= "t=1" )
plt.legend()
plt.show()

    


