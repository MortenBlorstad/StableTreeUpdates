

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,RepeatedKFold
import numpy as np
import pandas as pd
import itertools
import datapreprocess
from stabletrees import Tree
from pareto_efficient import is_pareto_optimal
from tqdm import tqdm
SEED = 0
EPSILON = 1.1


def color_scatter(alpha, beta):
        if alpha==0 and beta==0:
            return "#3776ab" # baseline
        if alpha==0 and beta>0:
            return "#F0E442" # ABU
        if alpha>0 and beta==0:
            return "#CC79A7" # SL
        return "#E69F00" # combi of SL and ABU

def stability_measure(pred1, pred2):
    return np.mean((pred1- pred2)**2)

# from examples in R package ISLR2, https://cran.r-project.org/web/packages/ISLR2/ISLR2.pdf
datasets =["Boston", "Carseats","College", "Hitters", "Wage"]
targets = ["medv", "Sales", "Apps", "Salary", "wage"]

plot_info  = {ds:[] for ds in datasets} # (x,y,colors,marker)



hyperparameters = {
    "alpha": np.round(np.arange(0,1.01,0.5),2),
    "beta": np.round(np.arange(0,1.01,0.5),2)
}
search_grid = list(itertools.product(hyperparameters["alpha"], hyperparameters["beta"]))


criterion = "mse"

stability_all = {c:[] for c in search_grid}
mse_all= {c:[] for c in search_grid}

compute = True

if compute:
    for ds,target in zip(datasets,targets ):
        kf = RepeatedKFold(n_splits= 5,n_repeats=10, random_state=SEED)
        stability = {c:[] for c in search_grid}
        mse = {c:[] for c in search_grid}

        data = pd.read_csv("..//data/"+ ds+".csv") # load dataset
        data = datapreprocess.data_preperation(ds)

        y = data[target].to_numpy()
        X = data.drop(target, axis=1).to_numpy()
        
        print(X.shape)
        if ds in ["College","Hitters","Wage"]:
            y = np.log(y)
        for alpha, beta in tqdm(search_grid, desc=f"running repeated k-fold for {len(search_grid)} hyperparameter settings"):

            # initial model 
            model = Tree(criterion=criterion,min_samples_leaf=5,adaptive_complexity=True,alpha=alpha, beta=beta)
            
            for train_index, test_index in kf.split(X):
                X_12, y_12 = X[train_index],y[train_index]
                X_test,y_test = X[test_index],y[test_index]
                X1,X2,y1,y2 =  train_test_split(X_12, y_12, test_size=0.5, random_state=SEED)

                model.fit(X1,y1)
                
                pred1 = model.predict(X_test)
            
                pred1_train = model.predict(X_12)
            
                pred1_orig= model.predict(X1)
                
                
                model.update(X_12,y_12)
                pred2 = model.predict(X_test)
                pred2_orig= model.predict(X1)
                pred2_train =  model.predict(X_12)

                mse[(alpha, beta)].append(mean_squared_error(y_test,pred2))

                stability[(alpha, beta)].append(stability_measure(pred1,pred2))

                

        print(ds)
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
            plot_info[ds].append((ds,(alpha, beta),x_r, y_r, alpha, beta , x_abs,y_abs,x_se, y_se, x_abs_se, y_abs_se ))
        print()

    # df_list = list(itertools.chain(*plot_info.values()))
    # df = pd.DataFrame(df_list, columns=["dataset","method",'loss', 'stability', "alpha","beta", 'loss_abs', "stability_abs",'loss_se', 'stability_se', 'loss_abs_se', 'stability_abs_se'  ] )
    # df.to_csv('StableTrees_examples/results/main_experiment_ISLR.csv', index=False)
else:
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
    
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=( 11.7, 2*11.7/3), dpi=500)
    axes = axes.ravel()
    plt.rcParams.update(plot_params)
    df =pd.read_csv('StableTrees_examples/results/main_experiment_ISLR.csv')
    datasets =["Boston", "Carseats","College", "Hitters", "Wage"]
    for ds,ax in zip(datasets,axes[:-1]):
        # if ds != "College":
        #     continue
        plot_info = df[df.dataset == ds]
        frontier = []
        X = np.zeros((len(plot_info)+1, 2))
        X[1:,0] = [row['loss'] for index, row  in plot_info.iterrows()]
        X[1:,1] = [row['stability'] for index, row  in plot_info.iterrows()]
        X[0,0] = 1
        X[0,1] = 1
        for i in range(X.shape[0]):
            if is_pareto_optimal(i, X):
                frontier.append((X[i,0],X[i,1]))
        frontier = sorted(frontier)


        print(frontier)
        frontier = [ (frontier[0][0], 2) ] + frontier+ [ (2, frontier[-1][1]) ]
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='both', which='major', labelsize=10*3/2)
        ax.axvline(x=1, linestyle = "--", c = "#3776ab", lw = 1*3/2)
        ax.axhline(y=1, linestyle = "--", c = "#3776ab", lw = 1*3/2)
        ax.set_title(ds,fontsize = 12*3/2)
        
        scatters = [ax.scatter(x = row['loss'], y=row['stability'],edgecolors="black",c = color_scatter(row['alpha'],row['beta']), s = 40*3/2) if (row['loss'],row['stability']) in frontier else ax.scatter(x = row['loss'], y=row['stability'],c = color_scatter(row['alpha'],row['beta']), s = 20*3/2) for index, row  in plot_info.iterrows()]

        #texts = [ax.text(x = row['loss'], y=row['stability'], s = r"$\mathbf{"+row['marker']+"}$",fontsize=8,weight='heavy') if (row['loss'],row['stability']) in frontier else ax.text(x = row['loss'], y=row['stability'], s = "$"+row['marker']+"$",fontsize=8) for index, row  in plot_info.iterrows()]
        #adjust_text(texts,x =X[:,0], y = X[:,1],add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax, force_text = (0.3,0.3))#
        ax.set_xlabel("loss",fontsize=12*3/2)
        ax.set_ylabel('instability',fontsize=12*3/2)


        
    colors2 = {"baseline":"#3776ab", 
                "SL":"#CC79A7", 
                "ABU":"#F0E442",
                "SL+ABU": "#E69F00",}


    # create a common legend for all the plots
    legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                                markerfacecolor=v, markersize=10*3/2) for k,v in colors2.items()  ]

    axes[-1].legend( handles=legend_elements, loc='center',fontsize=10*3/2)
    axes[-1].axis("off")
    # adjust spacing between subplots
    fig.tight_layout()
    #plt.show()
    plt.savefig(f"StableTrees_examples\plots\main_experiment_ISLR.png")
    plt.close()


