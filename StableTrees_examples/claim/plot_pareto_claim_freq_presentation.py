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
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8.27, 8.27),dpi=500)#

plt.rcParams.update(plot_params)

import itertools
method = "tree"
df =pd.read_csv(f'results/claim_freq_results_tree.csv')

plot_info = df[df.method =="tree"]
frontier = []
X = np.zeros((len(plot_info)+1, 2))
X[1:,0] = [row['loss_abs'] for index, row  in plot_info.iterrows()]
X[1:,1] = [row['stability_abs'] for index, row  in plot_info.iterrows()]
X[0,0] = 0.25554634862817177
X[0,1] = 2.636053784373512e-05
for i in range(X.shape[0]):
    if is_pareto_optimal(i, X):
        frontier.append((X[i,0],X[i,1]))
frontier = sorted(frontier)

# ax2 = ax.twinx()
# ax3 = ax.twiny()
# [ax2.scatter(x = row['loss'] , y=row['stability'] , s = 1, c =c,alpha = 0) for index, row  in plot_info.iterrows()]
# #[ax3.scatter(x = x_, y=y, s = 1, c =c,alpha = 0) for index, row  in plot_info.iterrows()]

print(frontier)
frontier = [ (frontier[0][0], 2) ] + frontier+ [ (2, frontier[-1][1]) ]
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
# ax.axvline(x=0.25554634862817177, linestyle = "--", c = "#8C564B", lw = 0.5)
# ax.axhline(y=2.636053784373512e-05, linestyle = "--", c = "#8C564B", lw = 0.5)
print(np.array(plot_info.m=="baseline"))
print(plot_info[plot_info.m=="baseline"].loss.item())
ax.axvline(x=plot_info[plot_info.m=="baseline"].loss_abs.item(), linestyle = "--", c = "#1f77b4", lw = 0.5)
ax.axhline(y=plot_info[plot_info.m=="baseline"].stability_abs.item(), linestyle = "--", c = "#1f77b4", lw = 0.5)

marker_to_method = {"tree": "o", "rf": "v", "GTB":"D" }
scatters = [ax.scatter(x = row['loss_abs'], y=row['stability_abs'], marker=marker_to_method[row['method']], c =row['color']) for index, row  in plot_info.iterrows()]
# scatters.append(ax.scatter(x=[0.25554634862817177],y=[2.636053784373512e-05], marker = "X", c = "#8C564B"))
texts = [ax.text(x = row['loss_abs'], y=row['stability_abs'], s = r"$\mathbf{"+row['m']+"}$",fontsize=8,weight='heavy') if (row['loss_abs'],row['stability_abs']) in frontier else ax.text(x = row['loss_abs'], y=row['stability_abs'], s = "$"+row['m']+"$",fontsize=8) for index, row  in plot_info.iterrows()]
#texts.append(ax.text(x = 0.25554634862817177, y=2.636053784373512e-05, s = r"$\mathbf{GLM}$",fontsize=8,weight='heavy') )
adjust_text(texts,x =X[:,0], y = X[:,1],add_objects=scatters, arrowprops=dict(arrowstyle="-", color='k', lw=0.1),ax= ax, force_text = (0.3,0.3))#
ax.set_xlabel("Poisson deviance",fontsize=10)
ax.set_ylabel('stability',fontsize=10)
#ax.set_xlim(0.24875,0.2575 )

# ax.set_xlim(0.2515,0.2523 )  
# ax.set_ylim(0,0.013 )   
colors2 = {"Baseline": "#1f77b4", "NU":"#D55E00", "SL":"#CC79A7", 
            "TR":"#009E73", 
            "ABU":"#F0E442",
            "BABU": "#E69F00",}
# create a common legend for all the plots
legend_elements = [Line2D([0], [0], marker='s', color='w', label=k,
                            markerfacecolor=v, markersize=14) for k,v in colors2.items()  ]

# legend_elements = [Line2D([0], [0],  marker = "X" ,color='w',markerfacecolor="#8C564B", markersize=14, label='GLM')] 
# legend_elements += [Line2D([0], [0],  marker = "o" ,color='w',markerfacecolor=None,markeredgecolor = "black",markersize=10, label='tree')] 
# legend_elements += [Line2D([0], [0],  marker = "v" ,color='w',markerfacecolor=None,markeredgecolor = "black",markersize=10, label='random forest')] 
# legend_elements += [Line2D([0], [0],  marker = "D" ,color='w',markerfacecolor=None,markeredgecolor = "black", markersize=10, label='GTB')] 
# legend_elements += [Line2D([0], [0], marker='s', color='w', label=k, markerfacecolor=v, markersize=14) for k,v in colors2.items()  ]


print()
ax.legend( handles=legend_elements,loc="center right",   # Position of legend
           borderaxespad=0.1,
            bbox_to_anchor=(1, 0.55),    # Small spacing around legend box
           fontsize="10")
# adjust spacing between subplots
plt.tight_layout()
plt.subplots_adjust(right=0.84)
#plt.show()
plt.savefig(f"StableTrees_examples\plots\\claim_freq_pareto_presentation_tree1.png")
plt.close()