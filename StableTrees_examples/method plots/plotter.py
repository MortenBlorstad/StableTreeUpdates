
from matplotlib import pyplot as plt
import numpy as np

# colnames = {0: "age",1 : "sex", 2:"bmi", 3:"bp", 4: "tc", 5: "ldl", 6: "hdl", 7: "tch", 8:"ltg", 9: "glu" }


colnames = {0: "CompPrice",
            1 : "Income",
            2:"Advertising",
            3:"Population", 
            4: "Price", 
            5: "ShelveLoc",
            6:"age", 
            7: "Education", 
            8:"Urban", 
            9: "US" }



font_size = 8
def plot(node, index = 0, indices = []):
        '''
        plots the tree. A visualisation of the tree
        '''
        #plt.rcParams["figure.figsize"] = (20,10)
        __plot(node,index = index,indices=indices )
        plt.plot(0, 0, alpha=1) 
        plt.axis("off")
    
def __plot(node,x=0,y=-1,off_x = 100000,off_y = 10, color = "royalblue", index = 0, indices = []):
    '''
    a helper method to plot the tree. 
    '''
    # No child.

    for ind in indices:
        if ind == index:
            color = "orange"

    props = dict(facecolor=color,boxstyle='round', alpha=0.1)
    # No child.
    if node is None:
        return
    if node.is_leaf():
        textstr = ''.join((
        f"$w$ = {node.prediction:.2f}\n",
        #f"impurity: {node.score:.3f}\n",
        f"samples: {node.n_samples}"))
        
        plt.plot(x+10, y, alpha=1) 
        plt.plot(x-10, y, alpha=1) 
        plt.text(x, y, textstr, fontsize=font_size-1, bbox=props,ha='center')
        #plt.text(x, y,f"{node.prediction:.4f}", fontsize=8,ha='center')
        
        return 
  
    

    textstr = ''.join((
        #f"$x_{node.split_feature} \leq{node.split_value:.4f}$ \n",
        f"${colnames[node.split_feature]} \leq{node.split_value:.1f}$ \n",
        f"samples: {node.n_samples}"
        ))


        
    new_x, new_y = x-off_x,y-off_y
    #plt.text(x, y,f"X[{node.feature}] <= {node.value:.4f}", fontsize=8,ha='center')
    #plt.text(x, y-2,f"impurity: {node.score:.3f}", fontsize=8,ha='center')
    #plt.text(x, y-4,f"nsamples: {node.nsamples}", fontsize=8,ha='center')
    plt.text(x, y, textstr, fontsize=font_size, bbox=props,ha='center')
    plt.annotate("", xy=(new_x, new_y+4), xytext=(x-2, y-1),
        arrowprops=dict(arrowstyle="->"))
    bbox = plt.gca().get_children()[0].get_bbox_patch()
    bbox.set_width(0.5)
    __plot(node.get_left_node(),new_x, new_y, off_x*0.5,color = color, index= index*2+1, indices=indices)
            
    
    new_x, new_y = x+off_x,y-off_y
    # plt.text(x, y,f"X[{node.feature}] <= {node.value:.4f}", fontsize=8,ha='center')
    # plt.text(x, y-2,f"impurity: {node.score:.3f}", fontsize=8,ha='center')
    # plt.text(x, y-4,f"nsamples: {node.nsamples}", fontsize=8,ha='center')
    plt.text(x, y, textstr, fontsize=font_size, bbox=props,ha='center')
    plt.annotate("", xy=(new_x , new_y+4), xytext=(x+2, y-1),
        arrowprops=dict(arrowstyle="->"))
    bbox = plt.gca().get_children()[0].get_bbox_patch()
    bbox.set_width(0.5)
    __plot(node.get_right_node(), new_x, new_y,off_x*0.5,color = color, index= index*2+2, indices=indices)
    

def __plot_decision_lines(node,X,min_v,max_v,x_min,x_max):
    
    if node.is_leaf():
        v1 = (x_min -min_v)/(max_v-min_v)
        v2 = (x_max -min_v)/(max_v-min_v)
        print(v1,v2,min_v,max_v)
        plt.axhline(y =node.prediction, xmin = v1, xmax= v2,color = 'r')
        return
    
    plt.axvline(x = node.value,color = 'b')
    mask = X[:,0]<=node.value
    x_max = np.max(X[~mask,0])
    x_min = np.min(X[mask,0])
    __plot_decision_lines(node.left,X[mask],min_v,max_v,
                            x_min = x_min, x_max = node.value)
    __plot_decision_lines(node.right,X[~mask],min_v,max_v,
                            x_min = node.value,x_max=x_max)



def plot_decision_lines(node,X,y, X2 = None,y2 = None):
    plt.rcParams["figure.figsize"] = (20,10)
    plt.scatter(x = X,y = y, c = "orange",label = "t0")
    if X2 is not None and y2 is not None:
        plt.scatter(x = X2,y = y2, c = "blue",label = "t1")
    plt.legend()
    plt.ylabel("y")
    plt.xlabel("X")
    
    min_v = np.min(X[:,0]) - 0.01*np.abs(np.min(X[:,0]))
    max_v = np.max(X[:,0]) + 0.01*np.abs(np.max(X[:,0]))
    plt.xlim(min_v, max_v)
    __plot_decision_lines(node,X,
        min_v = min_v, max_v = max_v,
        x_min = min_v,x_max = max_v)
   
