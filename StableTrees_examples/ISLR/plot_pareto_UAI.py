#######################
# Script to plot pareto frontier for the ISLR datasets for the tree-based methods
#######################

from matplotlib import pyplot as plt
from adjustText import adjust_text
from pareto_efficient import is_pareto_optimal
import numpy as np
import pandas as pd
#plt.rcParams.update({'figure.autolayout': True})
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter



def color_scatter(alpha, beta):
        if alpha==0 and beta==0:
            return "#3776ab" # baseline
        if alpha==0 and beta>0:
            return "#F0E442" # ABU
        if alpha>0 and beta==0:
            return "#CC79A7" # SL
        return "#E69F00" # combi of SL and ABU

#each Axes has a brand new prop_cycle, so to have differently
# colored curves in different Axes, we need our own prop_cycle
# Note: we CALL the axes.prop_cycle to get an itertoools.cycle

plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         #'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}




# create figure and axes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=( 11.7, 2*11.7/3), dpi=500)
axes = axes.ravel()
plt.rcParams.update(plot_params)

college_box_plot_info = []
import itertools




df_ISLR =pd.read_csv('StableTrees_examples/results/main_experiment_ISLR.csv')
df_california  = pd.read_csv('StableTrees_examples/results/main_experiment.csv')

df = pd.concat([df_california,df_ISLR], ignore_index=True)

datasets =["California","Boston", "Carseats","College", "Hitters", "Wage"]



for ds,ax in zip(datasets,axes):
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
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.tick_params(axis='both', which='major', labelsize=10*3/2)


    ax.axvline(x=1, linestyle = "--", c = "#3776ab", lw = 1*3/2)
    ax.axhline(y=1, linestyle = "--", c = "#3776ab", lw = 1*3/2)
    ax.set_title(ds,fontsize = 12*3/2)
    loss = [row['loss'] for index, row in plot_info.iterrows() ]
    stab = [row['stability'] for index, row in plot_info.iterrows()]
    scatters = [ax.scatter(x = row['loss'], y=row['stability'],edgecolors="black",c = color_scatter(row['alpha'],row['beta']), s = 40*3/2) if (row['loss'],row['stability']) in frontier else ax.scatter(x = row['loss'], y=row['stability'],c = color_scatter(row['alpha'],row['beta']), s = 20*3/2) for index, row  in plot_info.iterrows()]

    #texts = [ax.text(x = row['loss'], y=row['stability'], s = r"$\mathbf{"+row['marker']+"}$",fontsize=8*3/2,weight='heavy') if (row['loss'],row['stability']) in frontier else ax.text(x = row['loss'], y=row['stability'], s = "$"+row['marker']+"$",fontsize=8*3/2) for index, row  in plot_info.iterrows()]
    
    # if ds == "Wage":
    #     ax.set_xlim(xmin = np.min(loss)- 0.002 ,xmax =np.max(loss) + 0.0005)
    #     ax.set_ylim(ymin = 0.05,ymax = np.max(stab)+0.1)
    # elif ds in ["Boston","Carseats"]:
    #     ax.set_xlim(xmin = np.min(loss)- 0.005 ,xmax =np.max(loss) + 0.01)
    #     ax.set_ylim(ymin = 0.175,ymax = np.max(stab)+0.1)
    # elif ds =="California":
    #     ax.set_ylim(ymin = 0.3,ymax = np.max(stab)+0.1)
    #     ax.set_xlim(xmin = np.min(loss) - 0.02,xmax = np.max(loss) + 0.01)
    # elif ds =="College":
    #     ax.set_xlim(xmin = np.min(loss) - 0.02,xmax = np.max(loss) + 0.005)
    #     ax.set_ylim(ymin = 0.175,ymax = np.max(stab)+0.1)
    # else:    
    #     ax.set_xlim(xmin = np.min(loss) - 0.02,xmax = np.max(loss) + 0.01)
    #     ax.set_ylim(ymin = 0.175,ymax = np.max(stab)+0.1)


        

        


    # if ds =="Carseats" and method == "randomforest":
    #     ax.set_xlim((0.965,1.125))
    #     ax.set_ylim((0.45,1.05))
    #ax.set_ylim((0.0,1.05))
    # if ds =="Wage" and method == "tree":
    #     ax.set_xlim((0.99,1.0275))
        
    # if ds =="Boston" and method == "tree":
    #     ax.set_xlim((0.92,1.05))
    #ax.set_ylim((0.2,1.05))
    #adjust_text(texts,x =X[:,0], y = X[:,1],add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax, force_text = (0.3,0.3))#
    ax.set_xlabel("loss",fontsize=12*3/2)
    ax.set_ylabel('instability',fontsize=12*3/2)
        


    
colors2 = {"baseline":"#3776ab", 
            "Constant":"#CC79A7", 
            "UWR":"#F0E442",
            "Combined": "#E69F00"}
legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                            markerfacecolor=v, markersize=14) for k,v in colors2.items()  ]

fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=len(colors2),fontsize = 10*3/2)

fig.tight_layout()
fig.subplots_adjust(top=0.9)

plt.savefig(f"StableTrees_examples\plots\\main_experiment_all.png")
plt.close()