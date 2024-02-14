from stabletrees import Tree
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





x_all,y_all = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=False)

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
        if alpha<0.4:
            return "#f0c566" # combi of SL and ABU
        elif alpha<1:
            return "#E69F00" # combi of SL and ABU
        elif alpha<1.2:
            return "#b87f00" # combi of SL and ABU
        elif alpha<2:
            return "#8a5f00" # combi of SL and ABU
        elif alpha<2.5:
            return "#452f00" # combi of SL and ABU
        return "#E69F00" 

criterion = "mse"
models = {  
                "baseline": Tree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, alpha=0,beta=0),
                "SL":Tree(criterion = criterion,min_samples_leaf=5, adaptive_complexity=True, alpha=2,beta=0.2),
                "SLABU": Tree(criterion = criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.2,beta=0.6),
                "SLABU1": Tree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=0.4,beta=0.6),
                "SLABU2": Tree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=1,beta=1),
                "SLABU3": Tree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True, alpha=1.2,beta=0.4)
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
    results= {"name": [], "loss": [],"stability":[], "marker":[] }
    for name, model in models.items():
            results["name"].append(name)
            results["marker"].append(f"({model.alpha}, {model.beta})")
            results["loss"].append(mse[name])
            results["stability"].append(stab[name])



    results = pd.DataFrame(results)
    # Save the DataFrame to a text file
    output_file = 'StableTrees_examples/results/varied_sample_size_experiment.txt'
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
    df = pd.read_csv('StableTrees_examples/results/varied_sample_size_experiment.txt', sep='\t')
    df['loss'] = df['loss'].apply(string_to_list)
    df['stability'] = df['stability'].apply(string_to_list)
    mse = {} 
    stab = {}
    markers = {}
    for index, row in df.iterrows():
        markers[row["name"]] = row["marker"]
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

    plt.rcParams.update(plot_params)
    f, ax = plt.subplots(1, 1, sharex=True, figsize = (8, 8),dpi=500)
    ax.tick_params(axis='both', which='major', labelsize=20)

    sample_size_legends = []
    for size, marker in sample_size_to_marker.items():
        sample_size_legends.append(ax.scatter([], [], marker=marker, color='black', label=f'n={size}',s=120))


    texts =[] 
    scatters = []
    for name, model in models.items():

        ax.plot(mse[name], stab[name], linestyle='--', linewidth=2, c=color_scatter(model.alpha,model.beta))
        # Plot scatter points with different markers
        for i, (n, x, y) in enumerate(zip(sample_sizes, mse[name], stab[name])):
            if i ==0:
                texts.append(ax.text(x = x, y=y, s = markers[name],fontsize=20))
            scatters.append(ax.scatter(x, y, marker=sample_size_to_marker[n] , s=120, c = color_scatter(model.alpha,model.beta)))
    names = list(models.keys())
    scatters+= [ax.scatter(x = x, y=y, s = 140,marker=sample_size_to_marker[sample_sizes[j]],linewidths=2,c= color_scatter(models[names[i]].alpha,models[names[i]].beta),edgecolors = "black") for (x,y,i,j) in frontier]
        
    ax.set_xlim(xmax = 0.61)    
    ax.set_ylabel("instability",fontsize=24)
    ax.set_xlabel("loss",fontsize=24)
    plt.legend(loc='upper left',fontsize="20")
    adjust_text(texts,add_objects=scatters,ax= ax)
    plt.tight_layout()
    plt.savefig(f"StableTrees_examples\plots\\varied_sample_size_experiment.png")
    plt.close()
 