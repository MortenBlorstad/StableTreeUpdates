import numpy as np  
from stabletrees import BaseLineTree,AbuTreeI, AbuTree,NaiveUpdate, StabilityRegularization
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
import pandas as pd
if __name__ == "__main__":
    np.random.seed(0)
    # X  = np.array([[1.1,1],[1,2],[2.1,4],[2,3],[4,5],[6,10], [4,6]])
    # y = X[:,0]*2 + X[:,1]*0.5 + np.random.normal(0,0.5, X.shape[0])
    from sklearn.datasets import load_boston,load_diabetes
    from sklearn.model_selection import train_test_split,RepeatedKFold
    


    # data = pd.read_csv("data/"+ "boston" +".csv") # load dataset
    # target = "medv"
    # feature =  [ "crim", "rm"]
    # data = data.dropna(axis=0, how="any") # remove missing values if any
    # data = data.loc[:, feature + [target]] # only selected feature and target variable
    # y = data[target].to_numpy()+ 100
    # X = data.drop(target, axis=1).to_numpy()

    # kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=0)
    # model = AbuTreeI(criterion="poisson")
    # i =0
    # for train_index, test_index in kf.split(X):        
    #     if i ==12:
    #         X_12, y_12 = X[train_index],y[train_index]
    #         X_test,y_test = X[test_index],y[test_index]

    #     i +=1
        
    # X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=0)   
    # model.fit(X1,y1)  
    # pred1 = model.predict(X_test)
    # pred1_train = model.predict(X_12)
    # pred1_orig= model.predict(X1)
    # print(X_12.shape)
    # print("before")
    # print("\n"*10)
    
    # model.update(X_12,y_12)
    # model.plot()
    # plt.show()

    # print("after")
    # model.plot()
    # plt.show()
    #X,y = load_diabetes(return_X_y=True)
    
    
    df = pd.read_csv("C:\\Users\\mb-92\\OneDrive\\Skrivebord\\studie\\StableTrees\\StableTrees_examples\\test_data.csv")
    X = df["x"].to_numpy().reshape(-1,1)
    y= df["y"].to_numpy()
    # X = np.random.multivariate_normal([4 ,10], np.array([[1, 0.5], [0.5,1]]), size=1000)
    # def formula(X, noise = 1):
    #     return  np.exp(0.75*X[:,0] + 0.1*X[:,1]  + np.random.normal(0,noise))
    # y = formula(X)+1
    # n = 1000
    # X = np.random.uniform(size=(n,1))
    # y = np.random.poisson(lam=X.ravel(),size = n) +1

    # n = 200
    # X =np.random.uniform(size=(n,1), low = 0,high = 10)
    # y = np.random.poisson(X.ravel(),size=n) #np.exp(X.ravel()+  np.random.normal(0, 0.5, size=n)) 
    # # print(2*(y.mean() - y).sum())
    # #for i in range(100):
    # X_test =np.random.uniform(size=(n,1), low = 0,high = 10)
    # y_test = np.random.poisson(X.ravel(),size=n)

    X1,X2,y1,y2 = train_test_split(X,y,test_size=0.5,random_state=0)
    # X1 = X[0:250,:]
    # y1 = y[0:250]
    # X2 = X[250:500,:]
    # y2 = X[250:500]
    tree = BaseLineTree(adaptive_complexity=True,min_samples_leaf=5,criterion="mse").fit(X1,y1)
    #tree2 = AbuTreeI(adaptive_complexity=True,min_samples_leaf=5,criterion="poisson").fit(X1,y1)
    tree3 = AbuTree(adaptive_complexity=True,min_samples_leaf=5,criterion="mse").fit(X1,y1)
   
    # tree = BaseLineTree(max_depth=5,min_samples_leaf=5,criterion="poisson").fit(X1,y1)
    # tree2 = AbuTreeI(max_depth=5,min_samples_leaf=5,criterion="poisson").fit(X1,y1)
    # tree3 = AbuTree(max_depth=5,min_samples_leaf=5,criterion="poisson").fit(X1,y1)
    nu = StabilityRegularization(adaptive_complexity=True,min_samples_leaf=5,criterion="mse", lmbda=0.5).fit(X1,y1)
    
    nu.update(X,y)
    


    # print(mean_poisson_deviance(y, tree.predict(X)))
    # #print(mean_poisson_deviance(y, tree2.predict(X)))
    # print(mean_poisson_deviance(y, tree3.predict(X)))
    
    # print(np.sort(np.unique(tree.predict(X))))
    # #print(np.sort(np.unique(tree2.predict(X))))
    # print(np.sort(np.unique(tree3.predict(X))))
    # #tree.update(X,y)
    

    # tree.plot()
    # plt.show()

    plt.subplot(1,2,1)
    ypred = tree.predict(X)
    plt.scatter(X[:,0],y, alpha = 0.1)

    plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)

    # plt.subplot(1,3,2)
    # ypred = tree2.predict(X)
    # plt.scatter(X[:,0],y, alpha = 0.1)

    # plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)


    
    plt.subplot(1,2,2)
    ypred = tree3.predict(X)
    plt.scatter(X[:,0],y, alpha = 0.1)

    plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
    plt.show()


    
    
    #tree2.update(X2,y2)
    

    tree = BaseLineTree(adaptive_complexity=True,criterion="mse").fit(X1,y1)
    #tree = BaseLineTree(max_depth=5,criterion="poisson").fit(X1,y1)

    plt.subplot(1,4,1)
    ypred = tree.predict(X)
    plt.scatter(X[:,0],y, alpha = 0.1)
    plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
    plt.title("Prior tree")
    
    #tree4 = BaseLineTree(max_depth=5,criterion="poisson").fit(X,y)
    tree = BaseLineTree(adaptive_complexity=True,min_samples_leaf=5,criterion="mse").fit(X,y)

    plt.subplot(1,4,2)
    ypred = tree.predict(X)
    plt.scatter(X[:,0],y, alpha = 0.1)
    plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
    plt.title("Train 2 - only D2")

    # plt.subplot(1,4,3)
    # ypred = tree2.predict(X)
    # plt.scatter(X[:,0],y, alpha = 0.1)

    # plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
    # plt.title(f"Posterior tree improved")

    plt.subplot(1,4,3)
    tree3.update(X,y)
    ypred = tree3.predict(X)
    plt.scatter(X[:,0],y, alpha = 0.1)
    plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
    plt.title(f"Posterior tree")

    plt.subplot(1,4,4)
    tree4 = AbuTree(adaptive_complexity=True,criterion="mse").fit(X1,y1)
    tree4.update(X,y)
    ypred = tree4.predict(X)
    plt.scatter(X[:,0],y, alpha = 0.1)

    plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
    plt.title("Train 2 - D1 & D2")
    plt.show()




    # print(mean_poisson_deviance(y_test, tree.predict(X_test)))
    # #print(mean_poisson_deviance(y_test, tree2.predict(X_test)))
    # print(mean_poisson_deviance(y_test, tree3.predict(X_test)))
    # print(mean_poisson_deviance(y_test, tree4.predict(X_test)))

    #X2 = X[0:250,:]; y2 = y[0:250]
    
    #print(np.exp(0) - y )
    # tree = AbuTreeI(criterion="mse", max_depth=4).fit(X,y)
    # tree.plot()
    # plt.show()
    
    # ypred = tree.predict(X)
    # plt.scatter(X[:,0],y, alpha = 0.1)

    # plt.scatter(X[:,0],ypred[:],c ="red", alpha = 0.5)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    

    
    # ax.scatter(X[:,0],X[:,1],y, alpha = 0.1)
    # pos = np.array([ypred.flatten(),ypred.flatten()]).reshape(-1,1)
    # index = np.argsort(ypred)

    # ax.scatter(X[index,0],X[index,1],ypred[index],c ="red", alpha = 0.5)
    # plt.show()
   

    # tree.plot()
    # plt.show()


    # tree = AbuTreeI(criterion="poisson").fit(X1,y1)
    
    # ypred = tree.predict(X1)
    # plt.subplot(1,2,1)
    # plt.scatter(X1,y1)
    # plt.scatter(X1,ypred,c ="red")
    # plt.xlabel("X")
    # plt.ylabel("y")
    # plt.title("Prior tree")
    # tree = tree.update(X2,y2)
    # ypred = tree.predict(X2)
    
    # plt.subplot(1,2,2)
    # plt.scatter(X2,y2)
    
    # plt.scatter(X2,ypred,c ="red")
    
    # plt.xlabel(f"X2")
    # plt.ylabel(f"y2")
    # plt.title(f"Posterior tree")
    # ypred = tree.predict(X)
    # print(mean_squared_error(y,ypred))
    
    # plt.show()
    # criterion = "poisson"
    # tree = AbuTreeI(criterion=criterion,adaptive_complexity=True).fit(X1,y1)
    # ypred = tree.predict(X1)
    # plt.subplot(1,4,1)
    # plt.scatter(X1,y1)
    # plt.scatter(X1,ypred,c ="red")
    # plt.xlabel("X1")
    # plt.ylabel("y1")
    # #plt.ylim(top = 400)
    # plt.title("Prior tree")

    # ypred = tree.predict(X1)
    # print(mean_squared_error(y1,ypred))
    
    
  
    # t = AbuTreeI(criterion=criterion,adaptive_complexity=True).fit(X2,y2)
    # ypred = t.predict(X2)
    # plt.subplot(1,4,2)
    # plt.scatter(X2,y2)
    # plt.scatter(X2,ypred,c ="red")
    # plt.xlabel("X2")
    # plt.ylabel("y2")
    # plt.title("Train 2 - only D2")
    # ypred = t.predict(X)
    # print(mean_squared_error(y,ypred))

    
    # tree = tree.update(X2,y2)
    # ypred = tree.predict(X2)
    

    # ypred = tree.predict(X2)
    
    # plt.subplot(1,4,3)
    # plt.scatter(X2,y2)
    # plt.scatter(X2,ypred,c ="red")
    
    # plt.xlabel(f"X2")
    # plt.ylabel(f"y2")
    # plt.title(f"Posterior tree")
    # ypred = tree.predict(X)
    # print(mean_squared_error(y,ypred))
    
    

    # t = AbuTreeI(criterion=criterion,adaptive_complexity=True).fit(X,y)
    # ypred = t.predict(X)
    # print(mean_squared_error(y,ypred))
    # plt.subplot(1,4,4)
    # plt.scatter(X,y)
    # plt.scatter(X,ypred,c ="red")
    # plt.xlabel("X")
    # plt.ylabel("y")
    # plt.title("Train 2 - D1 & D2")

    # plt.show()

# ##
#     tree = AbuTree().fit(X,y)
#     ypred = tree.predict(X)
#     print(1,mean_squared_error(y,ypred))
#     plt.subplot(2,4,1)
#     plt.scatter(X,y)
#     plt.scatter(X,ypred,c ="red")
#     plt.xlabel("X")
#     plt.ylabel("y")
#     plt.title("Prior tree")

#     t = BaseLineTree(adaptive_complexity=True).fit(X,y)
#     ypred = t.predict(X)
#     print(5,mean_squared_error(y,ypred))
#     plt.subplot(2,4,5)
#     plt.scatter(X,y)
#     plt.scatter(X,ypred,c ="red")
#     plt.xlabel("X")
#     plt.ylabel("y")
#     plt.title("standard tree")
   
#     n = 500
#     np.random.seed(123)
#     for i,g in enumerate(range(3)):
#         X =np.random.uniform(size=(n,1), low = 0,high = 4)
#         y =np.random.normal(X.ravel(),1, size=n)
#         tree = tree.update(X,y)
#         ypred = tree.predict(X)
#         print(i+2,mean_squared_error(y,ypred))
#         plt.subplot(2,4,i+2)
#         plt.scatter(X,y)
#         plt.scatter(X,ypred,c ="red")
#         plt.xlabel(f"X{i+2}")
#         plt.ylabel(f"y{i+2}")
#         plt.title(f"Posterior tree {i+1}")
#         t = t.update(X,y)
#         ypred = t.predict(X)
#         print(4+i+2,mean_squared_error(y,ypred))
#         plt.subplot(2,4,4+i+2)
#         plt.scatter(X,y)
#         plt.scatter(X,ypred,c ="red")
#         plt.xlabel("X")
#         plt.ylabel("y")
#         plt.title("standard tree")
   
#     plt.show()
    
    # tree.plot()
    # plt.show()
    
    
        