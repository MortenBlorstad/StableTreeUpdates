from stabletrees import Tree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.linear_model import PoissonRegressor,LinearRegression
from adjustText import adjust_text
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

brown_palette = [
    (1, 1, 1),
    (0.8, 0.6, 0.4),
    (0.7, 0.5, 0.3),
    (0.6, 0.4, 0.2),
    (0.5, 0.3, 0.1),
]

cmap_brown = LinearSegmentedColormap.from_list('brown_palette', brown_palette, N=256)

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
abu = Tree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,alpha=0,beta=1)
np.random.seed(SEED)
n = 1000
noise = 2
x = np.sort(np.random.uniform(-4,4,(n,1)),axis=0) #np.random.normal(0,noise/2,(n,1)) #
y = x.ravel()**2 + np.random.normal(0,noise,n)




x_test = np.sort(np.random.uniform(-4,4,(1000,1)),axis=0) #np.sort(np.random.normal(0,noise/2,(1000,1)),axis=0) #
y_test = x_test.ravel()**2 +np.random.normal(0,noise,1000)

n_t = 1000
x_t = np.random.uniform(-4,4,(n_t,1))  #np.random.normal(0,noise/2,(n,1)) #  
y_t = x_t.ravel()**2  + np.random.normal(0,noise,n_t)



x2  = np.vstack((x,x_t))
y2 = np.concatenate((y,y_t))

x_t = np.random.uniform(-4,4,(n_t,1))  #np.random.normal(0,noise/2,(n,1)) #  
y_t = x_t.ravel()**2  + np.random.normal(0,noise,n_t)

x3  = np.vstack((x,x_t))
y3 = np.concatenate((y,y_t))

abu.fit(x,y)

info=abu.predict_info(x_test)

y_hat = abu.predict(x_test)
var_y_hat = info[:,2]
var_y = info[:,3]
#gamma = (1+var_y)/(1+var_y_hat)
gamma = info[:,1]*info[:,4]
upper_bound = y_hat + np.sqrt(var_y_hat)
lower_bound = y_hat - np.sqrt(var_y_hat)


#-----------
f, ((ax1,ax2,ax3),( ax5, ax6,ax7)) = plt.subplots(2, 3, figsize = (16, 8),dpi=100)
norm_var  = Normalize(vmin=0, vmax=max(var_y_hat))
cmap_var  = plt.get_cmap('Oranges')
cmap_y_var  = plt.get_cmap('Blues')
cmap_gamma  = plt.get_cmap(cmap_brown)
# Plot predictions with uncertainty and color-coded fill

for i in range(len(x_test) - 1):
    ax1.fill_betweenx(y =[-7, np.max(y_hat) + 5], x1=x_test[i], x2=x_test[i + 1], color=cmap_var(norm_var(var_y_hat[i])))


ax1.plot(x_test, y_hat, label='Predicted Y', c = "black")
ax1.scatter(x_test, y_test, c = "red",s=1)
sm = ScalarMappable(cmap=cmap_var, norm=norm_var)
sm.set_array([])  # dummy empty array for the colorbar
cbar = plt.colorbar(sm, label='Prediction variance', ax=ax1)

#-----------

norm_y_var  = Normalize(vmin=0, vmax=max(var_y))

for i in range(len(x_test) - 1):
    ax2.fill_betweenx(y =[-7, np.max(y_hat) + 5], x1=x_test[i], x2=x_test[i + 1], color=cmap_y_var(norm_y_var(var_y[i])))



ax2.plot(x_test, y_hat, label='Predicted Y', c = "black")
ax2.scatter(x_test, y_test, c = "red",s=1)

#plt.fill_between(x.ravel(), lower_bound, upper_bound, alpha=0.3, label='Uncertainty')
# Add colorbar
sm_y_var  = ScalarMappable(cmap=cmap_y_var, norm=norm_y_var)
sm_y_var.set_array([])  # dummy empty array for the colorbar
cbar = plt.colorbar(sm_y_var, label='response variance', ax=ax2)

#-----------
norm_gamma  = Normalize(vmin=0, vmax=max(gamma))

for i in range(len(x_test) - 1):
    ax3.fill_betweenx(y =[-7, np.max(y_hat) + 5], x1=x_test[i], x2=x_test[i + 1], color=cmap_gamma(norm_gamma (gamma[i])))




ax3.plot(x_test, y_hat, label='Predicted Y', c = "black")
ax3.scatter(x_test, y_test, c = "red",s=1)

#plt.fill_between(x.ravel(), lower_bound, upper_bound, alpha=0.3, label='Uncertainty')
# Add colorbar
sm_gamma  = ScalarMappable(cmap=cmap_gamma, norm=norm_gamma)
sm_gamma.set_array([])  # dummy empty array for the colorbar
cbar = plt.colorbar(sm_gamma, label=r'$\gamma$', ax=ax3)

# =======

abu.update(x2,y2)
print("asda")
info=abu.predict_info(x_test)

y_hat = abu.predict(x_test)
var_y_hat = info[:,2]
var_y = info[:,3]
gamma =  info[:,1]*info[:,4]
upper_bound = y_hat + np.sqrt(var_y_hat)
lower_bound = y_hat - np.sqrt(var_y_hat)


#-----------
norm_y_var  = Normalize(vmin=0, vmax=max(var_y))
norm_var  = Normalize(vmin=0, vmax=max(var_y_hat))

cmap_var  = plt.get_cmap('Oranges')
cmap_y_var  = plt.get_cmap('Blues')
cmap_gamma  = plt.get_cmap(cmap_brown)
# Plot predictions with uncertainty and color-coded fill
for i in range(len(x_test) - 1):
    ax5.fill_betweenx(y =[-7, np.max(y_hat) + 5], x1=x_test[i], x2=x_test[i + 1], color=cmap_var(norm_var(var_y_hat[i])))


ax5.plot(x_test, y_hat, label='Predicted Y', c = "black")
ax5.scatter(x_test, y_test, c = "red",s=1)
sm = ScalarMappable(cmap=cmap_var, norm=norm_var)
sm.set_array([])  # dummy empty array for the colorbar
cbar = plt.colorbar(sm, label='Prediction variance', ax=ax5)

#-----------


for i in range(len(x_test) - 1):
    ax6.fill_betweenx(y =[-7, np.max(y_hat) + 5], x1=x_test[i], x2=x_test[i + 1], color=cmap_y_var(norm_y_var(var_y[i])))



ax6.plot(x_test, y_hat, label='Predicted Y', c = "black")
ax6.scatter(x_test, y_test, c = "red",s=1)

#plt.fill_between(x.ravel(), lower_bound, upper_bound, alpha=0.3, label='Uncertainty')
# Add colorbar
sm_y_var  = ScalarMappable(cmap=cmap_y_var, norm=norm_y_var)
sm_y_var.set_array([])  # dummy empty array for the colorbar
cbar = plt.colorbar(sm_y_var, label='response variance', ax=ax6)

#-----------


for i in range(len(x_test) - 1):
    ax7.fill_betweenx(y =[-7, np.max(y_hat) + 5], x1=x_test[i], x2=x_test[i + 1], color=cmap_gamma(norm_gamma (gamma[i])))


norm_gamma  = Normalize(vmin=0, vmax=np.max(gamma))

ax7.plot(x_test, y_hat, label='Predicted Y', c = "black")
ax7.scatter(x_test, y_test, c = "red",s=1)

#plt.fill_between(x.ravel(), lower_bound, upper_bound, alpha=0.3, label='Uncertainty')
# Add colorbar
sm_gamma  = ScalarMappable(cmap=cmap_gamma, norm=norm_gamma)
sm_gamma.set_array([])  # dummy empty array for the colorbar
cbar = plt.colorbar(sm_gamma, label=r'$\gamma$', ax=ax7)

plt.show()