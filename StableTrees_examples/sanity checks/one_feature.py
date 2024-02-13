import numpy as np
from stabletrees import Tree
import matplotlib.pyplot as plt

x = np.random.uniform(-4,4,1000)
y = np.random.normal(x.ravel(),1,1000)

tree = Tree(adaptive_complexity=5, min_samples_leaf=5)
tree.fit(x,y)

plt.scatter(x,y)
plt.scatter(x,tree.predict(x))
plt.show()



