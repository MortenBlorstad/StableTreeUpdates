#######################
# Script to plot pareto frontier for the ISLR datasets for the tree-based methods
#######################

from matplotlib import pyplot as plt
from adjustText import adjust_text
from pareto_efficient import is_pareto_optimal
import numpy as np
import pandas as pd
plt.rcParams.update({'figure.autolayout': True})
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
plot_params = {"ytick.color" : "black",
          "xtick.color" : "black",
          "axes.labelcolor" : "black",
          "axes.edgecolor" : "black",
         'figure.autolayout': True,
          "font.family" : "serif",
          'text.latex.preamble': r"\usepackage{amsmath}",
          "font.serif" : ["Computer Modern Serif"]}

# create figure and axes
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8.27, 11),dpi=500)#
axes = axes.ravel()
plt.rcParams.update(plot_params)

college_box_plot_info = []
import itertools


# switch to plot pareto frontier for different tree-based methods
method =  "tree" #"randomforest" #"agtboost" # 
#df =pd.read_csv(f'results/{method}_ISLR_results_10_5.csv')
df =pd.read_csv('results/ISLR_pareto.csv')

datasets =["Boston", "Carseats","College", "Hitters", "Wage"]


point_style = {"tree":"o", "randomforest":"v", "agtboost": "D"}

print(df)
print(df[df.dataset == "Boston"])
plot_info = df[df.dataset == "Boston"]
for index, row in plot_info.iterrows():
    print(row['loss'],row['stability'],row['marker'])
# plot data on the axes
print(len(plot_info))

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
    ax.axvline(x=1, linestyle = "--", c = "#3776ab", lw = 0.5)
    ax.axhline(y=1, linestyle = "--", c = "#3776ab", lw = 0.5)
    ax.set_title(ds,fontsize = 10)
    
    scatters = [ax.scatter(x = row['loss'], y=row['stability'], s = 6, c =row['color'],marker = point_style[method]) for index, row  in plot_info.iterrows()]
    #scatters.append(ax.scatter(x=[1],y=[1], c = "#3776ab",s = 6,marker = "v"))
    texts = [ax.text(x = row['loss'], y=row['stability'], s = r"$\mathbf{"+row['marker']+"}$",fontsize=8,weight='heavy') if (row['loss'],row['stability']) in frontier else ax.text(x = row['loss'], y=row['stability'], s = "$"+row['marker']+"$",fontsize=8) for index, row  in plot_info.iterrows()]
    #texts.append(ax.text(x = 1, y=1, s = r"$baseline$",fontsize=8) )
    if ds =="Wage" and method == "agtboost":
        ax.set_xlim((0.999,1.008))
        ax.set_ylim((0.30,1.05))

    if ds =="Carseats" and method == "agtboost":
        ax.set_xlim((0.995,1.0525))

    if (ds =="College" or ds =="Hitters") and method == "agtboost":
        ax.set_xlim((0.99,1.07))




    # if ds =="Carseats" and method == "randomforest":
    #     ax.set_xlim((0.965,1.125))
    #     ax.set_ylim((0.45,1.05))
    #ax.set_ylim((0.0,1.05))
    if ds =="Wage" and method == "tree":
        ax.set_xlim((0.99,1.0275))
    #ax.set_ylim((0.2,1.05))
    adjust_text(texts,x =X[:,0], y = X[:,1],add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax, force_text = (0.3,0.3))#
    ax.set_xlabel("mse",fontsize=10)
    ax.set_ylabel('stability',fontsize=10)


    
colors2 = {"baseline":"#3776ab",
            "NU":"#D55E00",
            "TR":"#009E73", 
            "SL":"#CC79A7", 
            "ABU":"#F0E442",
            "BABU": "#E69F00",}

# colors2 = {
#             "NU":"#D55E00",
#             "TR":"#009E73", 
#             "SL":"#CC79A7", 
#             "ABU":"#F0E442",
#             "BABU": "#E69F00",}



# colors2 = { "SL":"#CC79A7"}
# create a common legend for all the plots
legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                            markerfacecolor=v, markersize=14) for k,v in colors2.items()  ]
# legend_elements = [Line2D([0], [0], marker='s', color='w', label="$baseline$",
#                             markerfacecolor="#3776ab", markersize=14) ] +legend_elements
print()
axes[-1].legend( handles=legend_elements, loc='center',fontsize="10")
axes[-1].axis("off")
# adjust spacing between subplots
fig.tight_layout()
#plt.show()
plt.savefig(f"StableTrees_examples\plots\\real_data_mse_pareto_{method}.png")
plt.close()