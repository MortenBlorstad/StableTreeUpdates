from stabletrees import BaseLineTree, AbuTree, NaiveUpdate,TreeReevaluation,StabilityRegularization,BABUTree,AbuTree2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_poisson_deviance, mean_squared_error
from sklearn.linear_model import PoissonRegressor,LinearRegression
from adjustText import adjust_text
from sklearn.datasets import fetch_california_housing
from tqdm import tqdm
SEED = 0
EPSILON = 1

def S1(pred1, pred2):
    return np.std(np.log((pred2+EPSILON)/(pred1+EPSILON)))#np.mean((pred1- pred2)**2)#

def S2(pred1, pred2):
    return np.mean((pred1- pred2)**2)


x_all,y_all = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=False)


plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}

colors = {"baseline":"#3776ab","NU":"#D55E00", "SL":"#CC79A7",
          "SL1":"#dba1c1",
          "SL2":"#d186af",
          "SL3":"#CC79A7",
          "SL4":"#a36085",
          "SL5":"#7a4864",
            "TR":"#009E73","TR1":"#009E73", "TR2":"#009E73","TR3":"#009E73",
            "TR4":"#009E73", "TR5":"#009E73","TR6":"#009E73",
            "TR7":"#009E73", "TR8":"#009E73","TR9":"#009E73",
            "ABU":"#f4ec7a",
            "ABU1":"#f3e967","ABU2":"#F0E442","ABU3":"#d8cd3b","ABU4":"#c0b634","ABU5":"#a89f2e",
            "BABU": "#E69F00", "BABU1": "#E69F00","BABU2": "#E69F00" ,"BABU3": "#E69F00","BABU4": "#E69F00","BABU5": "#E69F00","BABU6": "#E69F00"}


compute = False


criterion = "mse"
models = {  
                "baseline": BaseLineTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                #"NU": NaiveUpdate(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                #"TR":TreeReevaluation(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, delta=0.05),
                #"SL1":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.1),
                #"SL2":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.25),
                "SL3":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.5),
                #"SL4":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.75),
                "SL5":StabilityRegularization(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True,gamma=0.9),
                "ABU": AbuTree2(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True),
                #"ABU1": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.1),
                "ABU2": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.25),
                "ABU3": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.5),
                #"ABU4": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.75),
                #"ABU5": AbuTree2(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=0.9),
                #"BABU":AbuTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                #"BABU": BABUTree(criterion = criterion,min_samples_leaf=5,bumping_iterations=1,adaptive_complexity=True)
                }
if compute:
    time = 5
    prev_pred = {name:0 for name in models.keys()}
    mse = {name:[0]*time for name in models.keys()}
    stab = {name:[0]*time for name in models.keys()}


    from sklearn.model_selection import KFold
   


    repeat =50
    for i in tqdm(range(repeat), desc="running"):
        kf = KFold(n_splits=time+2,shuffle=True,random_state=i)
        inds = list(kf.split(x_all))
        for name, model in models.items():

            ind0 =inds[0][-1]
            ind_test =inds[-1][-1]
            # print(len(ind0))
            # print(len(ind_test))
            x = x_all[ind0]
            y = y_all[ind0]
            x_test = x_all[ind_test]
            y_test = y_all[ind_test]
            model.fit(x,y)

            pred = model.predict(x_test)

            prev_pred[name] = pred

        
        for t in range(time): 
            ind =inds[t+1][-1]
            # print(len(ind))
            x_t = x_all[ind]   
            y_t = y_all[ind]
            x  = np.vstack((x,x_t))
            y = np.concatenate((y,y_t))
            for name, model in models.items():
                model.update(x,y)

                pred = model.predict(x_test)
                stab[name][t] += np.mean((prev_pred[name]-pred)**2)
                mse[name][t] +=np.mean((y_test-pred)**2)
                prev_pred[name] = pred
    


    

    for name, model in models.items():
            mse[name] = [val/repeat for val in mse[name]]
            stab[name] = [val/repeat for val in stab[name]]

    import pandas as pd

    results  = pd.DataFrame()
    results= {"name": [], "loss": [],"stability":[] }
    for i, name in enumerate(models.keys()):
            results["name"].append(name)
            results["loss"].append(mse[name])
            results["stability"].append(stab[name])



    results = pd.DataFrame(results)
    # Save the DataFrame to a text file
    output_file = 'CH_overtime__results UAI.txt'
    results.to_csv(output_file, sep='\t', index=False)

if not compute:
    labels = {"baseline":"baseline",
            "SL": "SL", "SL1":"SL_{0.1}", "SL2":"SL_{0.25}","SL3":"SL_{0.5}",
            "SL4": "SL_{0.75}", "SL5": "SL_{0.9}",
            "ABU":"ABU",
            "ABU1":r"ABU_{0.1}","ABU2":r"ABU_{0.25}","ABU3":r"ABU_{0.5}","ABU4":r"ABU_{0.75}","ABU5":r"ABU_{0.9}",
            "BABU":"BABU"  }
    
    scatters = []
    texts = []
    import itertools
    import pandas as pd
    import ast
    from pareto_efficient import is_pareto_optimal
    def string_to_list(string):
        try:
            return ast.literal_eval(string)
        except ValueError:
            return []
    df = pd.read_csv('CH_overtime__results UAI.txt', sep='\t')
    # Apply the conversion to 'loss' and 'stability' columns
    df['loss'] = df['loss'].apply(string_to_list)
    df['stability'] = df['stability'].apply(string_to_list)


    mse = {} 
    stab = {}
    for index, row in df.iterrows():

        mse[row["name"]] = row["loss"]
        stab[row["name"]] = row["stability"]


    plt.rcParams.update(plot_params)
    f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 8),dpi=500)
    ax.tick_params(axis='both', which='major', labelsize=20)

    
    X = np.zeros((len(models),5,2 )) # n_model, time, loss and stab
    for i,name in enumerate(models.keys()):
        X[i,:,0] = mse[name]
        X[i,:,1] = stab[name]
    frontier = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if is_pareto_optimal(i, X[:,j]):
                frontier.append((X[i,j,0],X[i,j,1]))
        frontier = sorted(frontier)

    print(frontier)
    for name, model in models.items():
        scatters+= [ax.scatter(x = x, y=y, s = 140,marker='o', c = "black",edgecolors = "black") for (x,y) in frontier]
        ax.plot(mse[name], stab[name] ,label = r"$"+labels[name]+"$", linestyle='--', marker='o', markersize = 8,linewidth=2, c = colors[name])#np.arange(1,time+1,dtype=int)
        scatters+= [ax.scatter(x = x, y=y, s = 120, alpha=0) for (x,y) in zip(mse[name],stab[name])]
       
        #texts += [ ax.text(x =x, y=y, s = r"$"+labels[name]+"$" ,fontsize=20, ha='right', va='center')  for i,(x,y) in enumerate(zip(mse[name],stab[name])) if  i==0]

        texts += [ ax.text(x =x, y=y, s = r"$t="+str(1)+"$",fontsize=20, ha='right', va='center')  for i,(x,y) in enumerate(zip(mse[name],stab[name])) if  i==0]#(i+1) %5 ==0
        
    # ax.axvline(x=mse["baseline"][t], linestyle = "--", c = "#3776ab", lw = 0.5)
    # ax.axhline(y=stab["baseline"][t], linestyle = "--", c = "#3776ab", lw = 0.5)
    mx = 0
    mn = np.inf
    for name in models.keys():
        if mse[name][0] > mx:
            mx = mse[name][0]
        if mse[name][-1] < mn:
            mn = mse[name][-1]

    ax.set_xlim( xmax = mx+0.02)  
    ax.set_ylabel("instability",fontsize=12*2)
    ax.set_xlabel("loss",fontsize=12*2)
    plt.legend(loc='upper left',fontsize=8*2, ncols=2)
    adjust_text(texts,add_objects=scatters,ax= ax)
    plt.tight_layout()
    plt.savefig(f"StableTrees_examples\plots\\california_overtime_with_BABU1.png")
    plt.close()


    plt.rcParams.update(plot_params)
    f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 8),dpi=500)

    for name, model in models.items():
        if name == "BABU":
            continue
        ax.plot(mse[name], stab[name] , label = labels[name],linestyle='--', marker='o', markersize = 3,linewidth=1, c = colors[name])#np.arange(1,time+1,dtype=int)
        scatters+= [ax.scatter(x = x, y=y, s = 0.1, alpha=0) for (x,y) in zip(mse[name],stab[name])]
        texts += [ ax.text(x =x, y=y, s = r"$t="+str(1)+"$",fontsize=8, ha='right', va='center')  for i,(x,y) in enumerate(zip(mse[name],stab[name])) if  i==0]#(i+1) %5 ==0
        
    # ax.axvline(x=mse["baseline"][t], linestyle = "--", c = "#3776ab", lw = 0.5)
    # ax.axhline(y=stab["baseline"][t], linestyle = "--", c = "#3776ab", lw = 0.5)
        
    ax.set_ylabel("instability",fontsize=12)
    ax.set_xlabel("loss",fontsize=12)
    plt.legend(loc='upper left')
    adjust_text(texts,add_objects=scatters,ax= ax)
    plt.tight_layout()
    plt.savefig(f"StableTrees_examples\plots\\california_overtime.png")
    plt.close()

