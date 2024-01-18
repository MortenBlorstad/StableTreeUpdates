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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8),dpi=500)#

plt.rcParams.update(plot_params)


df =pd.read_csv('results/tree_CH_results.csv')

point_style = {"tree":"o", "randomforest":"v", "agtboost": "D"}
method = "tree"
# print(df)
# print(df[df.dataset == "Boston"])
# plot_info = df[df.dataset == "Boston"]
# for index, row in plot_info.iterrows():
#     print(row['loss'],row['stability'],row['marker'])
# # plot data on the axes
# print(len(plot_info))

    
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


print(frontier)
frontier = [ (frontier[0][0], 2) ] + frontier+ [ (2, frontier[-1][1]) ]
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.tick_params(axis='both', which='major', labelsize=20)

print(plot_info[plot_info.method == "baseline"]["loss_abs"].to_numpy())
print(plot_info.loss_abs[plot_info.method == "baseline"][0])
ax.axvline(x=plot_info.loss_abs[plot_info.method == "baseline"][0], linestyle = "--", c = "#3776ab", lw = 2)
ax.axhline(y=plot_info.stability_abs[plot_info.method == "baseline"][0], linestyle = "--", c = "#3776ab", lw = 2)

scatters = [ax.scatter(x = row['loss_abs'], y=row['stability_abs'], s = 40, c =row['color'],marker = point_style[method]) for index, row  in plot_info.iterrows()]

texts = [ax.text(x = row['loss_abs'], y=row['stability_abs'], s = r"$\mathbf{"+row['marker']+"}$",fontsize=20,weight='heavy') if (row['loss_abs'],row['stability_abs']) in frontier else ax.text(x = row['loss_abs'], y=row['stability_abs'], s = "$"+row['marker']+"$",fontsize=20) for index, row  in plot_info.iterrows()]
#ax.set_ylim((0.2,1.05))
adjust_text(texts,x =X[:,0], y = X[:,1],add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax, force_text = (0.3,0.3))#
ax.set_xlabel("loss",fontsize=24)
ax.set_ylabel('instability',fontsize=24)


    
colors2 = {"baseline":"#3776ab",
            "SL":"#CC79A7", 
            "ABU":"#F0E442",
            "BABU": "#E69F00",}#


# colors2 = { "SL":"#CC79A7"}
# create a common legend for all the plots
legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                            markerfacecolor=v, markersize=28) for k,v in colors2.items()  ]
# legend_elements = [Line2D([0], [0], marker='s', color='w', label="$baseline$",
#                             markerfacecolor="#3776ab", markersize=14) ] +legend_elements
print()
ax.legend( handles=legend_elements, loc='upper left',fontsize="20")
#ax.axis("off")
# adjust spacing between subplots
fig.tight_layout()
#plt.show()
plt.savefig(f"StableTrees_examples\plots\\CH_pareto_frontier.png")
plt.close()