

from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,AbuTree2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

for i in range(10):
    t1 = BaseLineTree(criterion = "mse", min_samples_leaf=5, adaptive_complexity=True,random_state=0)
    t2 = BaseLineTree(criterion = "mse", min_samples_leaf=5, adaptive_complexity=True,random_state=0)
    t3= AbuTree(criterion = "mse", min_samples_leaf=5, adaptive_complexity=True,random_state=0)
    t4= AbuTree(criterion = "mse", min_samples_leaf=5, adaptive_complexity=True,random_state=0)
    x = np.random.uniform(0,4,(1000,1))
    y = np.random.normal(x.ravel(),1,1000)
    x_t = np.random.uniform(0,4,(1000,1))
    y_t = np.random.normal(x.ravel(),1,1000)
    x_test = np.random.uniform(0,4,(1000,1))
    y_test = np.random.normal(x_test.ravel(),1,1000)
    t1.fit(x,y)
    t2.fit(x,y)
    t3.fit(x,y)
    t4.fit(x,y)
    y1 = t1.predict(x_test)
    y2 = t2.predict(x_test)
    y3 = t3.predict(x_test)
    y4 = t4.predict(x_test)
    t1.fit(x,y)
    t2.fit(x,y)
    x  = np.vstack((x,x_t))
    y = np.concatenate((y,y_t))
    t3.update(x,y)
    t4.update(x,y)

    
    print(i,np.all(y1 ==t1.predict(x_test)))
    print(i,np.all(y2 ==t2.predict(x_test)))
    print(i,np.all(y1 ==y2))
    print(i,np.all(y3==y4))
    print(i,np.all(t3.predict(x_test)==t4.predict(x_test)))


np.random.seed(0)

x = np.random.uniform(0,4,(1000,1))
y = np.random.normal(x.ravel(),1,1000)
t1 = BaseLineTree(criterion = "mse", min_samples_leaf=5, adaptive_complexity=True,random_state=0)
t1.fit(x,y)
x_test = np.random.uniform(0,4,(1000,1))
y_test = np.random.normal(x_test.ravel(),1,1000)

y1 = t1.predict(x_test)
plt.scatter(x_test,y_test, c = "blue")

plt.scatter(x_test,y1, c = "red")
plt.show()

