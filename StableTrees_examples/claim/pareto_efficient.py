import numpy as np



def strong_dominance(x,y):
    strict = False
    for xi,yi in zip(x,y):
        if xi>yi:
            return False
        elif xi<yi:
            strict =True
    return strict

def weak_dominance(x,y):
    for xi,yi in zip(x,y):
        if xi>yi:
            return False
    return True

def is_pareto_optimal(i, X):
    l = []
    for j in range(X.shape[0]):
        l.append(not strong_dominance(X[j,:], X[i,:]))
    return all(l)



def func(x):
    return 0.5*x**3 - 0.1*x**2 - 0.1*x**0.5- 0.2*x + 0.2

if __name__ == "__main__":
    np.random.seed(0)
    from matplotlib import pyplot as plt 
    X = np.random.uniform(0.1, 1,size = (3000,2))

    mask = np.where( (X[:,1]*X[:,0] > np.random.normal(0.1, 0.05, size = 3000) ) & (X[:,1]* X[:,0]  < np.random.normal(0.3,0.05, size = 3000)))


    X= X[mask]
    X += np.random.normal(0.01, 0.05,size = (X.shape[0],2))
    frontier = []
    for i in range(X.shape[0]):
        if is_pareto_optimal(i, X):
            frontier.append((X[i,0],X[i,1]))
    frontier = sorted(frontier)
    frontier = [ (frontier[0][0], 2) ] + frontier+ [ (2, frontier[-1][1]) ]
    plt.scatter(X[:,0],X[:,1], s=2, c = "grey")
    plt.scatter([x for x,y in frontier],[y for x,y in frontier], c = "r")
    plt.plot([x for x,y in frontier],[y for x,y in frontier], c = "r")
    # i = np.arange(0.1,0.8,0.01)
    # plt.plot(i, func(i))

    plt.xlabel("stability")
    plt.ylabel("performance")
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.show()