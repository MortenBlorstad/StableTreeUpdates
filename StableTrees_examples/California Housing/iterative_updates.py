from stabletrees import Tree
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from adjustText import adjust_text
from sklearn.datasets import fetch_california_housing
from tqdm import tqdm
SEED = 0
EPSILON = 1




x_all,y_all = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=False)

# ind = y_all<5
# y_all= y_all[ind]
# x_all = x_all[ind]

plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}

def color_scatter(alpha, beta):
        if alpha==0 and beta==0:
            return "#3776ab" # baseline
        if alpha==0 and beta>0:
            return "#F0E442" # ABU
        if alpha>0 and beta==0:
            return "#CC79A7" # SL
        if 0<alpha<0.5:
            return "#edbb4c" # combi of SL and ABU
        if 0<alpha<0.8:
            return "#E69F00" # combi of SL and ABU
        return "#b87f00" 


compute = False
# print(np.var(y_all)/len(y_all))
# plt.hist(y_all, bins = 100)
# plt.show()

criterion = "mse"
models = {  
                "baseline": Tree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, alpha=0,beta=0),
                #"SL":Tree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, alpha=1,beta=0),
                "SLABU": Tree(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0,beta=1),
                #"SLABU1": Tree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0,beta=1.2),
                #"SLABU2": Tree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.6,beta=2),
                #"SLABU3": Tree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=1.2,beta=0.4)
                }
time = 10
if compute:

    prev_pred = {name:0 for name in models.keys()}
    mse = {name:[0]*time for name in models.keys()}
    stab = {name:[0]*time for name in models.keys()}


    from sklearn.model_selection import KFold
   


    repeat =5
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
            print(t)
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
    results= {"name": [], "loss": [],"stability":[], "marker":[] }
    for name, model in models.items():
            results["name"].append(name)
            results["marker"].append(f"({model.alpha}, {model.beta})")
            results["loss"].append(mse[name])
            results["stability"].append(stab[name])



    results = pd.DataFrame(results)
    # Save the DataFrame to a text file
    output_file = 'StableTrees_examples/results/multiple_update_iterations_experiment.txt'
    results.to_csv(output_file, sep='\t', index=False)

if not compute:
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
    df = pd.read_csv('StableTrees_examples/results/multiple_update_iterations_experiment.txt', sep='\t')
    # Apply the conversion to 'loss' and 'stability' columns
    df['loss'] = df['loss'].apply(string_to_list)
    df['stability'] = df['stability'].apply(string_to_list)


    mse = {} 
    stab = {}
    marker = {}
    for index, row in df.iterrows():
        marker[row["name"]] = row["marker"]
        mse[row["name"]] = row["loss"]
        stab[row["name"]] = row["stability"]


    plt.rcParams.update(plot_params)
    f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 8),dpi=500)
    ax.tick_params(axis='both', which='major', labelsize=20)

    
    X = np.zeros((len(models),time,2 )) # n_model, time, loss and stab
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
        ax.plot(mse[name], stab[name] ,label = marker[name], linestyle='--', marker='o', markersize = 8,linewidth=2, c = color_scatter(model.alpha,model.beta))
        scatters+= [ax.scatter(x = x, y=y, s = 120, alpha=0) for (x,y) in zip(mse[name],stab[name])]
       

        texts += [ ax.text(x =x, y=y, s = r"$t="+str(1)+"$",fontsize=20, ha='right', va='center')  for i,(x,y) in enumerate(zip(mse[name],stab[name])) if  i==0]#(i+1) %5 ==0
        

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
    plt.savefig(f"StableTrees_examples\plots\\multiple_update_iterations_experiment_remake.png")
    plt.close()


