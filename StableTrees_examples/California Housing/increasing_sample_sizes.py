from stabletrees import BaseLineTree, AbuTree,StabilityRegularization,AbuTree2
import numpy as np
from matplotlib import pyplot as plt
from adjustText import adjust_text
from tqdm import tqdm
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 0
EPSILON = 1
np.random.seed(SEED)

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

colors = {"baseline":'#1f77b4',"NU":"#D55E00", "SL":"#CC79A7",
           "SL1":"#dba1c1",
          "SL2":"#d186af",
          "SL3":"#CC79A7",
          "SL4":"#a36085",
          "SL5":"#7a4864",
            "ABU":"#f4ec7a",
            "ABU1":"#f3e967","ABU2":"#F0E442","ABU3":"#d8cd3b","ABU4":"#c0b634","ABU5":"#a89f2e",
            "BABU": "#E69F00",
            }

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
                "ABU": AbuTree2(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0),
                "ABU2": AbuTree2(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.25),
                "ABU3": AbuTree2(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.5),
                #"BABU":AbuTree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True),
                #"BABU": BABUTree(criterion = criterion,min_samples_leaf=5,bumping_iterations=1,adaptive_complexity=True)
                }

sample_sizes = np.array([1000,5000,10000,15000]) #np.power(10,np.arange(2,6), dtype=int)

sample_size_to_marker = {
    1000: 'o',   # Circle
    5000: 's',  # Square
    10000: 'X', # Cross
    15000: '*', # Star 
}

print(sample_sizes)

prev_pred = {name:0 for name in models.keys()}
mse = {name:[0]*len(sample_sizes) for name in models.keys()}
stab = {name:[0]*len(sample_sizes)  for name in models.keys()}


repeat =50

compute = False
if compute:
    for i in tqdm(range(repeat),desc="running"):
        for t,n in enumerate(sample_sizes): 
            xs,x_test,ys,y_test = train_test_split(x_all, y_all, test_size=5000, random_state=i)

            for name, model in models.items():
                x,x_t,y,y_t = train_test_split(xs, ys, train_size=n//2, random_state=i+1)
                x_t= x_t[:n//2]
                y_t= y_t[:n//2]

                model.fit(x,y)

                pred = model.predict(x_test)
                prev_pred[name] = pred

                x2  = np.vstack((x,x_t))

                y2 = np.concatenate((y,y_t))

                model.update(x2,y2)
        
                pred = model.predict(x_test)
                stab[name][t] += np.mean((prev_pred[name]-pred)**2)
                mse[name][t] +=np.mean((y_test-pred)**2)
                prev_pred[name] = pred


    for name, model in models.items():
            mse[name] = [val/repeat for val in mse[name]]
            stab[name] = [val/repeat for val in stab[name]]

    results  = pd.DataFrame()
    results= {"name": [], "loss": [],"stability":[] }
    for i, name in enumerate(models.keys()):
            results["name"].append(name)
            results["loss"].append(mse[name])
            results["stability"].append(stab[name])



    results = pd.DataFrame(results,)
    # Save the DataFrame to a text file
    output_file = 'CH_increasing_sample_size_results.txt'
    results.to_csv(output_file, sep='\t', index=False)

if not compute:
    scatters = []
    texts = []
    import ast
    from pareto_efficient import is_pareto_optimal
    def string_to_list(string):
        try:
            return ast.literal_eval(string)
        except ValueError:
            return []
    df = pd.read_csv('CH_increasing_sample_size_results.txt', sep='\t')
    df['loss'] = df['loss'].apply(string_to_list)
    df['stability'] = df['stability'].apply(string_to_list)
    mse = {} 
    stab = {}
    for index, row in df.iterrows():
        mse[row["name"]] = row["loss"]
        stab[row["name"]] = row["stability"]

    X = np.zeros((len(models),4,2 )) # n_model, time, loss and stab
    for i,name in enumerate(models.keys()):
        X[i,:,0] = mse[name]
        X[i,:,1] = stab[name]
    frontier = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if is_pareto_optimal(i, X[:,j]):
                frontier.append((X[i,j,0],X[i,j,1],i,j))
        frontier = sorted(frontier)

    print(frontier)



    labels = {"baseline":"baseline","NU":"NU","SL": "SL", "SL1":"SL_{0.1}", "SL2":"SL_{0.25}","SL3":"SL_{0.5}",
                "SL4": "SL_{0.75}", "SL5": "SL_{0.9}",
                "TR":"TR","TR1":"TR_{0,5}",
                "TR2":"TR_{5,5}", "TR3" :"TR_{10,5}",
                "ABU":"ABU",
                "ABU1":r"ABU_{0.1}","ABU2":r"ABU_{0.25}","ABU3":r"ABU_{0.5}","ABU4":r"ABU_{0.75}","ABU5":r"ABU_{0.9}",
                "BABU":"BABU", "BABU1": r"BABU_{1}","BABU2": r"BABU_{3}" ,"BABU3": r"BABU_{5}","BABU4": r"BABU_{7}","BABU5": r"BABU_{10}","BABU6": r"BABU_{20}"   }


    


    plt.rcParams.update(plot_params)
    f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 8),dpi=500)
    ax.tick_params(axis='both', which='major', labelsize=20)

    sample_size_legends = []
    for size, marker in sample_size_to_marker.items():
        sample_size_legends.append(ax.scatter([], [], marker=marker, color='black', label=f'n={size}',s=120))

    # for name, model in models.items():
    #     ax.plot(mse[name], stab[name] , label = labels[name],linestyle='--', marker='o', markersize = 3,linewidth=1, c = colors[name])#np.arange(1,time+1,dtype=int)
    #     scatters+= [ax.scatter(x = x, y=y, s = 0.1, alpha=0) for (x,y) in zip(mse[name],stab[name])]
    #     #texts += [ ax.text(x =x, y=y, s = r"$n="+str(n)+"$",fontsize=8, ha='left', va='top')  for i,(n,x,y) in enumerate(zip(sample_sizes,mse[name],stab[name])) if  name =="baseline"]#(i+1) %5 ==0

    texts =[] 
    scatters = []
    for name, model in models.items():
        # Plot line
        #ax.plot(mse[name], stab[name], label=labels[name], linestyle='--', linewidth=1, c=colors[name])

        
        
        ax.plot(mse[name], stab[name], linestyle='--', linewidth=2, c=colors[name])
        # Plot scatter points with different markers
        for i, (n, x, y) in enumerate(zip(sample_sizes, mse[name], stab[name])):
            if i ==0:
                texts.append(ax.text(x = x, y=y, s = r"$"+labels[name]+"$",fontsize=20))
            marker = sample_size_to_marker[n]  
            scatters.append(ax.scatter(x, y, marker=marker, s=120, c=colors[name]))
    names = list(models.keys())
    scatters+= [ax.scatter(x = x, y=y, s = 140,marker=sample_size_to_marker[sample_sizes[j]],linewidths=2,c=colors[names[i]],edgecolors = "black") for (x,y,i,j) in frontier]
        
    ax.set_xlim(xmax = 0.61)     
    ax.set_ylabel("instability",fontsize=24)
    ax.set_xlabel("loss",fontsize=24)
    plt.legend(loc='upper left',fontsize="20")
    adjust_text(texts,add_objects=scatters,ax= ax)
    plt.tight_layout()
    plt.savefig(f"StableTrees_examples\plots\\increasing_sample_sizes_with_BABU.png")
    plt.close()
 

    ####################

    plt.rcParams.update(plot_params)
    f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 8),dpi=500)
    ax.tick_params(axis='both', which='major', labelsize=20)

    sample_size_legends = []
    for size, marker in sample_size_to_marker.items():
        sample_size_legends.append(ax.scatter([], [], marker=marker, color='black', label=f'n={size}',s=120))

    # for name, model in models.items():
    #     ax.plot(mse[name], stab[name] , label = labels[name],linestyle='--', marker='o', markersize = 3,linewidth=1, c = colors[name])#np.arange(1,time+1,dtype=int)
    #     scatters+= [ax.scatter(x = x, y=y, s = 0.1, alpha=0) for (x,y) in zip(mse[name],stab[name])]
    #     #texts += [ ax.text(x =x, y=y, s = r"$n="+str(n)+"$",fontsize=8, ha='left', va='top')  for i,(n,x,y) in enumerate(zip(sample_sizes,mse[name],stab[name])) if  name =="baseline"]#(i+1) %5 ==0

    texts =[] 
    scatters = []
    for name, model in models.items():
        if name == "BABU":
            continue
        
        
        ax.plot(mse[name], stab[name], linestyle='--', linewidth=2, c=colors[name])
        # Plot scatter points with different markers
        for i, (n, x, y) in enumerate(zip(sample_sizes, mse[name], stab[name])):
            if i ==0:
                texts.append(ax.text(x = x, y=y, s = r"$"+labels[name]+"$",fontsize=20))
            marker = sample_size_to_marker[n]  
            scatters.append(ax.scatter(x, y, marker=marker, s=120, c=colors[name]))
    names = list(models.keys())
    scatters+= [ax.scatter(x = x, y=y, s = 140,marker=sample_size_to_marker[sample_sizes[j]],linewidths=2,c=colors[names[i]],edgecolors = "black") for (x,y,i,j) in frontier]
        
    ax.set_xlim(xmax = 0.61)     
    ax.set_ylabel("instability",fontsize=24)
    ax.set_xlabel("loss",fontsize=24)
    plt.legend(loc='upper left',fontsize="20")
    adjust_text(texts,add_objects=scatters,ax= ax)
    plt.tight_layout()
    plt.savefig(f"StableTrees_examples\plots\\increasing_sample_sizes.png")
    plt.close()