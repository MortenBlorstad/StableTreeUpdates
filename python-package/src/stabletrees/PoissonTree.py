import numpy as np
from matplotlib import pyplot as plt
EPSILON = 10 * np.finfo('double').eps
class PoissonSplitter():
    """
        Poisson deviance = 2/n * sum(y_true * log(y_true/y_pred) + y_pred - y_true)

    """
    def __init__(self, mn, mx, lmda=1) -> None:
        pass
    
    
    def get_split(self,X,y, y_prev = None):
        min_score = np.inf
        best_split = None
        split_feature = None
        sorted_indices =X
        sorted_indices = np.argsort(X,axis=0)
        n = X.shape[0]
        mse_root = np.mean((y - y.mean())**2)
    
        any_split = False
        for i in range(X.shape[1]):
            feature = X[:,i]

            sorted_index = sorted_indices[:,i]
            if feature[sorted_index[0]] == feature[sorted_index[n-1]]:
                continue

            any_split_, score, split_value =self.find_split(feature,y,sorted_index, y_prev)
            #print(i,min_score>score,min_score,score, n)
            #print(i,score)
            if any_split_ and min_score>score:
                any_split = True
                min_score=score
                best_split = split_value
                split_feature=i
                
        return any_split, split_feature,mse_root, min_score, best_split

    def find_split(self,feature,y,sorted_index, y_prev = None):
        min_score = np.inf
        splitValue = None
        sum_ylogy_l = 0
        sum_ylogy_r = np.sum(y*np.log(y+EPSILON))
        n = len(y)
        n_l = 0
        n_r = n
        sum_y_l = 0
        sum_y_r = np.sum(y)
        node_score = (1/n)*(sum_ylogy_r - (sum_y_r/n_r)*np.log(sum_y_r/n_r)*n_r)
        
        if y_prev is not None:
            sum_ylogy_prev_l = 0
            sum_ylogy_prev_r = np.sum(y_prev*np.log(y_prev+EPSILON))
            sum_y_prev_l = 0
            sum_y_prev_r = np.sum(y_prev)



        for i in range(n-1):
            low = sorted_index[i]
            high = sorted_index[i+1]

            lowValue  = feature[low]
            highValue = feature[high]
            value = (lowValue+highValue)/2

            n_l+=1
            n_r-=1

            sum_y_l +=y[low]  
            sum_y_r -=y[low]    
            
            sum_ylogy_l+=((y[low]+EPSILON)*np.log(y[low]+EPSILON))
            sum_ylogy_r-=((y[low]+EPSILON)*np.log(y[low]+EPSILON))
            
            
            sum_ylogpred_l = ( (sum_y_l/n_l) + EPSILON)*np.log( (sum_y_l/n_l) + EPSILON)*n_l
            sum_ylogpred_r = ((sum_y_r/n_r) + EPSILON)*np.log( (sum_y_r/n_r) + EPSILON)*n_r

            if y_prev is not None:
                sum_y_prev_l += y_prev[low]
                sum_y_prev_r -= y_prev[low]
                sum_ylogy_prev_l+=((y_prev[low]+EPSILON)*np.log(y_prev[low]+EPSILON))
                sum_ylogy_prev_r-=((y_prev[low]+EPSILON)*np.log(y_prev[low]+EPSILON))
                

            if sum_y_l <=0.0000001 or sum_y_r <=0.0000001:
                continue

            if highValue-lowValue <0.00001:
                #print("skipped",highValue,lowValue)
                continue
            if lowValue == feature[sorted_index[n-1]]:
                #print("break")
                break
            #score = (sum_ylogy_l + sum_ylogy_r - sum_ylogpred_l-sum_ylogpred_r)/n
            
            score = ((sum_ylogy_l-sum_ylogpred_l)/n_l) + (sum_ylogy_r - sum_ylogpred_r)/n_r
            if y_prev is not None:
                score += self.lmda* ((sum_ylogy_prev_l-sum_ylogpred_l- (sum_y_prev_l - sum_y_l))/n_l + (sum_ylogy_prev_r-sum_ylogpred_r- (sum_y_prev_r - sum_y_r))/n_r)

            #print(i,highValue,lowValue,sum_ylogy_l ,sum_ylogpred_l,n_l, sum_ylogy_r, sum_ylogpred_r, n_r, score)
            if score<=min_score:
                min_score = score
                splitValue=value
        #print("done")
        return node_score,min_score,splitValue

class Node:
    def __init__(self,left = None,right =  None, prediction= None, feature = None,node_score=None,score = None,value = None,
                    nsamples=None,R = None,depth = None, prediction_var = None) -> None:
        self.left = left
        self.right =  right
        self.prediction= prediction
        self.feature = feature
        self.score = score
        self.node_score = node_score
        self.value = value
        self.nsamples = nsamples
        self.depth = depth
        self.prediction_var = prediction_var
        self.R  = R

    def is_leaf(self):
        return self.left is None and self.right is None       
class Tree:
    def __init__(self,max_depth = None, min_samples_split = 2) -> None:
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.nleaves = None


    def _get_nleaves(self,node):
        if node.is_leaf():
            return 1
        else:
            self.nleaves+=self._get_nleaves(node.left)
            self.nleaves+=self._get_nleaves(node.right)

    def get_nleaves(self):
        if self.nleaves is not None:
            return self.nleaves 
        self._get_nleaves(self.root)
        return self.nleaves 


    def _all_same_label(self,y):
        return np.all(y[0]==y)

    def _all_same_features_values(self,X):
        return np.all(np.apply_along_axis(lambda x:len(np.unique(x)) , 0, X) ==1)

    #def poissondev(self,y,ypred):
    #    return(1/len(y))*np.sum(y*np.log((y+1)/(ypred+1)))

    def poissondev(self,y,ypred):
        return np.mean( (y- ypred)**2)

    def build(self,X,y, depth=0,y_prev = None):
        
        if self.max_depth is not None:
            if self.max_depth<=depth:
                pred = np.mean(y)
                node_score = np.mean((y-pred)**2)
                return Node(prediction=np.mean(y),nsamples=len(y), node_score = node_score)

        if X.shape[0] <1 or y.shape[0]<1:
            return 
        if X.shape[0] <2 or y.shape[0]<2:
            return  Node(prediction=y[0],nsamples=1, node_score=0, depth=depth)
        if X.shape[0] <self.min_samples_split or y.shape[0]<self.min_samples_split: 
            pred = np.mean(y)
            node_score = np.mean((y-pred)**2)
            return  Node(prediction=pred,nsamples=len(y), node_score =node_score,depth=depth)
        if self._all_same_label(y):
            return  Node(prediction=y[0],nsamples=len(y), node_score = 0,depth=depth)
        if self._all_same_features_values(X):
            pred = np.mean(y)
            node_score = np.mean((y-pred)**2)
            return  Node(prediction=pred,nsamples=len(y), node_score = node_score,depth=depth)
       
           
        

        any_split, feature_index, mse, score, value = self.splitter.get_split(X,y,y_prev)
    
        #print(feature_index, mse,value)
        if not any_split:
            pred = np.mean(y)
            return  Node(prediction=pred,nsamples=len(y), node_score = np.mean((y-pred)**2), depth=depth)
      
        mask = X[:,feature_index]<=value
        #n = X.shape[0]
        #n_l = X[mask].shape[0]
        #n_r = X[~mask].shape[0]
        #print("tree",np.mean((y[mask] - np.mean(y[mask]))**2)  + np.mean((y[~mask] - np.mean(y[~mask]))**2), score)
        pred = y.mean()
        node = Node(prediction = pred, feature=feature_index,node_score=mse, score = score ,value=value,nsamples=X.shape[0],R = np.ptp(y),depth=depth)
        y_prev_left = None
        y_prev_right = None
        if y_prev is not None:
            y_prev_left = y_prev[mask]
            y_prev_right = y_prev[~mask]
            current_stab = np.mean((y_prev - pred)**2)
            #if current_stab<mse:
            #    return node
        node.left = self.build(X[mask], y[mask], depth+1, y_prev_left)
        node.right = self.build(X[~mask], y[~mask], depth+1, y_prev_right)

        return node

    def fit(self,X,y):
        self.splitter = PoissonSplitter(np.min(y),np.max(y))
        self.root = self.build(X,y,depth = 0, y_prev=None)
        return self

    def update(self,X,y):
        return self.fit(X,y)

    def predict_obs(self,x):
        node = self.root
        
        while True:
            if x[node.feature]<=node.value:
                node = node.left
            else:
                node = node.right
            if node.is_leaf():
                break

        return node.prediction
        

    
    def predict(self,X):
        predictions = np.zeros(X.shape[0])
        for i,x in enumerate(X):
            #print(i,x)
            predictions[i] = self.predict_obs(x)
        return predictions
    


from matplotlib import pyplot as plt
import numpy as np
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

    if node.is_leaf():
        textstr = ''.join((
        f"value = {node.prediction:.2f}\n",
        #f"impurity: {node.score:.3f}\n",
        f"samples: {node.nsamples}"))
        
        plt.plot(x+10, y-5, alpha=1) 
        plt.plot(x-10, y-5, alpha=1) 
        plt.text(x, y, textstr, fontsize=20, bbox=props,ha='center')
        #plt.text(x, y,f"{node.prediction:.4f}", fontsize=8,ha='center')
        return 
  
    

    textstr = ''.join((
        f"$X_{node.feature} \leq{node.value:.4f}$ \n",
        f"impurity: {node.node_score:.2f}\n",
        f"samples: {node.nsamples}"))

    if node.left != None:
        
        new_x, new_y = x-off_x,y-off_y
        #plt.text(x, y,f"X[{node.feature}] <= {node.value:.4f}", fontsize=8,ha='center')
        #plt.text(x, y-2,f"impurity: {node.score:.3f}", fontsize=8,ha='center')
        #plt.text(x, y-4,f"nsamples: {node.nsamples}", fontsize=8,ha='center')
        plt.text(x, y, textstr, fontsize=20, bbox=props,ha='center')
        plt.annotate("", xy=(new_x, new_y+4), xytext=(x-2, y-1),
            arrowprops=dict(arrowstyle="->"))
        __plot(node.left,new_x, new_y, off_x*0.5,color = color, index= index*2+1, indices=indices)
                
    if node.right != None:
        new_x, new_y = x+off_x,y-off_y
        # plt.text(x, y,f"X[{node.feature}] <= {node.value:.4f}", fontsize=8,ha='center')
        # plt.text(x, y-2,f"impurity: {node.score:.3f}", fontsize=8,ha='center')
        # plt.text(x, y-4,f"nsamples: {node.nsamples}", fontsize=8,ha='center')
        plt.text(x, y, textstr, fontsize=20, bbox=props,ha='center')
        plt.annotate("", xy=(new_x , new_y+4), xytext=(x+2, y-1),
            arrowprops=dict(arrowstyle="->"))
        __plot(node.right, new_x, new_y,off_x*0.5,color = color, index= index*2+2, indices=indices)


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
   

if __name__ == "__main__":
    from sklearn.datasets import make_regression,load_diabetes

    from sklearn.datasets import make_regression,load_diabetes

    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split,KFold


    N = 1000
    np.random.seed(0)
    X = np.random.uniform(0,4, size = (N,1))
    y = np.random.poisson(np.exp(X.ravel()),N)

    t = Tree(max_depth=6,min_samples_split=20).fit(X,y)
    plt.scatter(X,y)
    plt.scatter(X,t.predict(X))

    plt.show()