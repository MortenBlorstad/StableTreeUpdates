
from sklearn.model_selection import train_test_split,RepeatedKFold
import numpy as np
import pandas as pd
import itertools
from stabletrees import AbuTree
from sklearn.metrics import mean_squared_error
from adjustText import adjust_text
from pareto_efficient import is_pareto_optimal
from sklearn.datasets import fetch_california_housing
from tqdm import tqdm

SEED = 0
EPSILON = 1.1

def stability_measure(pred1, pred2):
    return np.mean((pred1- pred2)**2)


X,y = fetch_california_housing(download_if_missing=True, return_X_y=True, as_frame=False) #get CH data
plot_info  = {"CH":[]} # (x,y,colors,marker)

hyperparameters = {
    "alpha": np.round(np.arange(0,2.01,0.2),2),
    "beta": np.round(np.arange(0,2.01,0.2),2)
}
search_grid = list(itertools.product(hyperparameters["alpha"], hyperparameters["beta"]))


criterion = "mse"
# alpha =0, beta= 0 -> baseline
stability_all = {c:[] for c in search_grid}
mse_all= {c:[] for c in search_grid}




compute = False
if compute:
    kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)

    # initial model 
    stability = {c:[] for c in search_grid}
    mse = {c:[] for c in search_grid}

    
   

    for alpha, beta in tqdm(search_grid, desc=f"running repeated k-fold for {len(search_grid)} hyperparameter settings"):

        model = AbuTree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=alpha, beta=beta)
        
        for _, (train_index, test_index) in enumerate(kf.split(X)):
            X_12, y_12 = X[train_index],y[train_index]
            X_test,y_test = X[test_index],y[test_index]
            X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)
            # initial model 
            model.fit(X1,y1)
            
            pred1 = model.predict(X_test)
            
            pred1_train = model.predict(X_12)
            
            pred1_orig= model.predict(X1)
            
            
            model.update(X_12,y_12)
            #print("after")
            pred2 = model.predict(X_test)
            pred2_orig= model.predict(X1)
            pred2_train =  model.predict(X_12)

            mse[(alpha, beta)].append(mean_squared_error(y_test,pred2))

            stability[(alpha, beta)].append(stability_measure(pred1,pred2))
        

    for alpha, beta in tqdm(search_grid, desc = "computing pareto frontier"):
        print("="*80)
        print(f"{(alpha, beta)}")
        
        mse_scale = np.mean(mse[(0,0)]);
        
        mse_scale = np.mean(mse[(0,0)]); S_scale = np.mean(stability[(0,0)]);
        loss_score = np.mean(mse[(alpha, beta)])
        loss_SE = np.std(mse[(alpha, beta)])/np.sqrt(10) #np.sqrt(len(mse[name]))
        loss_SE_norm = np.std(mse[(alpha, beta)]/mse_scale)/np.sqrt(10) #np.sqrt(len(mse[name]))
        stability_score = np.mean(stability[(alpha, beta)])
        stability_SE = np.std(stability[(alpha, beta)])/np.sqrt(10)  #np.sqrt(len(mse[name]))
        stability_SE_norm = np.std(stability[(alpha, beta)]/S_scale)/np.sqrt(10) #/np.sqrt(len(mse[name]))
        print(f"test - mse: {loss_score:.3f} ({loss_SE:.3f}), stability: {stability_score:.3f} ({stability_SE:.3f})")
        print(f"test - mse: {loss_score/mse_scale:.3f} ({loss_SE_norm:.2f}), stability: {stability_score/S_scale:.3f} ({stability_SE_norm:.2f})")
        print("="*80)
        mse_all[(alpha, beta)] += [score/mse_scale for score in mse[(alpha, beta)]]
        
        x_abs =  np.mean((mse[(alpha, beta)]))
        y_abs = np.mean(stability[(alpha, beta)])
        x_abs_se = loss_SE
        y_abs_se =stability_SE
        x_se  = loss_SE_norm
        y_se  = stability_SE_norm
        x_r = x_abs/mse_scale
        y_r = y_abs/S_scale
        
        plot_info["CH"].append(("CH",(alpha, beta),x_r, y_r, alpha, beta , x_abs,y_abs,x_se, y_se, x_abs_se, y_abs_se ))
    print()


    import itertools
    import os
    df_list = list(itertools.chain(*plot_info.values()))
    df = pd.DataFrame(df_list, columns=["dataset","method",'loss', 'stability', "alpha","beta", 'loss_abs', "stability_abs",'loss_se', 'stability_se', 'loss_abs_se', 'stability_abs_se'  ] )

    # if os.path.isfile('results/tree_CH__results_alpha_beta.csv'):
    #     old_df =pd.read_csv('results/tree_CH__results_alpha_beta.csv')
    #     for i,(d,m) in enumerate(zip(df.dataset, df.marker)):
    #         index = old_df.loc[(old_df["dataset"] == d) & (old_df["marker"] ==m)].index
    #         values  = df.iloc[i]
    #         if len(index)>0:
    #             old_df.iloc[index]=values
    #         else:
    #             print(values)
    #             old_df  = old_df.append(values, ignore_index=True)

    #     old_df.to_csv('results/tree_CH__results_alpha_beta.csv', index=False)
    # else:
    df.to_csv('StableTrees_examples/results/main_experiment.csv', index=False)
else:

    #######################
    # Script to plot pareto frontier for the ISLR datasets for the tree-based methods
    #######################

    from matplotlib import pyplot as plt
    from adjustText import adjust_text
    from pareto_efficient import is_pareto_optimal
    import numpy as np
    import pandas as pd
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter

    plot_params = {"ytick.color" : "black",
            "xtick.color" : "black",
            "axes.labelcolor" : "black",
            "axes.edgecolor" : "black",
#            'figure.autolayout': True,
            "font.family" : "serif",
            'text.latex.preamble': r"\usepackage{amsmath}",
            "font.serif" : ["Computer Modern Serif"]}

    # create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8),dpi=500)#

    plt.rcParams.update(plot_params)


    df =pd.read_csv('StableTrees_examples/results/main_experiment.csv')

    point_style = {"tree":"o", "randomforest":"v", "agtboost": "D"}
    method = "tree"
   

        
    plot_info = df[df.dataset == "CH"]
    print(plot_info)
    frontier = []
    X = np.zeros((len(plot_info)+1, 2))
    X[1:,0] = [row['loss_abs'] for index, row  in plot_info.iterrows()]
    X[1:,1] = [row['stability_abs'] for index, row  in plot_info.iterrows()]
    X[0,0] = 1
    X[0,1] = 1
    for i in range(X.shape[0]):
        if is_pareto_optimal(i, X):
            frontier.append((X[i,0],X[i,1]))
    frontier = sorted(frontier)

    baseline = "(0.0, 0.0)"
    print(frontier)
    #frontier = [ (frontier[0][0], 2) ] + frontier+ [ (2, frontier[-1][1]) ]
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', labelsize=20)
    print(plot_info[plot_info.method == baseline]["loss_abs"].to_numpy()[0])
    print(plot_info.loss_abs[plot_info.method == baseline])
    ax.axvline(x=plot_info.loss_abs[plot_info.method == baseline][0], linestyle = "--", c = "#3776ab", lw = 2)
    ax.axhline(y=plot_info.stability_abs[plot_info.method == baseline][0], linestyle = "--", c = "#3776ab", lw = 2)
    frontier_selected = [pos for i, pos in enumerate(frontier) if i % 5 ==0]
    
    def color_scatter(alpha, beta):
        if alpha==0 and beta==0:
            return "#3776ab" # baseline
        if alpha==0 and beta>0:
            return "#F0E442" # ABU
        if alpha>0 and beta==0:
            return "#CC79A7" # SL
        return "#E69F00" # combi of SL and ABU

    scatters = [ax.scatter(x = row['loss_abs'], y=row['stability_abs'],edgecolors="black",c = color_scatter(row['alpha'],row['beta']), s = 80) if (row['loss_abs'],row['stability_abs']) in frontier else ax.scatter(x = row['loss_abs'], y=row['stability_abs'],c = color_scatter(row['alpha'],row['beta']), s = 40) for index, row  in plot_info.iterrows()]

    texts = [ax.text(x = row['loss_abs'], y=row['stability_abs'], s = r"$\mathbf{("+str(row['alpha'])+","+ str(row['beta'])+")}$",fontsize=12,weight='heavy') for index, row  in plot_info.iterrows() if (row['loss_abs'],row['stability_abs']) in frontier_selected]
    #ax.set_ylim((0.2,1.05))
    adjust_text(texts,x =X[:,0], y = X[:,1],add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.5),ax= ax, force_text = (0.3,0.3))#
    ax.set_xlabel("loss",fontsize=24)
    ax.set_ylabel('instability',fontsize=24)

    colors2 = {"baseline":"#3776ab", 
            "SL":"#CC79A7", 
            "ABU":"#F0E442",
            "SL+ABU": "#E69F00"}

        

    legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                                markerfacecolor=v, markersize=12) for k,v in colors2.items()  ]
    print()
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(colors2),fontsize = 12)

    #ax.axis("off")
    # adjust spacing between subplots
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    #plt.show()
    plt.savefig(f"StableTrees_examples\plots\\main_experiment.png")
    plt.close()
