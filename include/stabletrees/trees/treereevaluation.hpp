#pragma once
#include "tree.hpp"
#include "splitter.hpp"
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort

using namespace std;

class TreeReevaluation: public Tree{
    public:
        TreeReevaluation(double alpha,double delta,int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate,unsigned int random_state);
        TreeReevaluation();
        virtual void update(const dMatrix &X, const dVector &y,const dVector &weights);
        std::vector<double> get_eps();
        std::vector<double> get_mse_ratio();
        std::vector<double> get_obs();
        std::vector<double>  mse_ratio; 
        std::vector<double> epss; 
        std::vector<double> obs; 
    private:
        tuple<Node*, bool> reevaluate_split(Node* node, const dMatrix &X, const dVector &y,const dVector &g, const dVector &h, double delta, int depth, const dVector &weights);
        Node* attempt_split(Node* node, const dMatrix &X, const dVector &y,const dVector &g, const dVector &h, int depth,const  dVector &weights);
        Node* update_rec(Node* node, const dMatrix &X, const dVector &y,const dVector &g, const dVector &h,double delta, int depth, const dVector &weights);
        double hoeffding_bound(double delta, int n);
        int number_of_examples;
        vector<size_t> sort_index(const vector<double> &v);
        double delta;
        double alpha;
        

};

std::vector<double> TreeReevaluation::get_mse_ratio(){
    return mse_ratio;
}
std::vector<double> TreeReevaluation::get_eps(){
    return epss;
}
std::vector<double> TreeReevaluation::get_obs(){
    return obs;
}


TreeReevaluation::TreeReevaluation():Tree(){
    Tree();
    this->alpha = 0.05;
    this->delta = 0.1;
}

TreeReevaluation::TreeReevaluation(double alpha,double delta,int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate,unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
    this->delta = delta;
    this->alpha = alpha;
}

vector<size_t> TreeReevaluation::sort_index(const vector<double> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values 
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

void TreeReevaluation::update(const dMatrix &X, const dVector &y, const dVector &weights){
    if(root == NULL){
        learn(X,y,weights);
    }

    number_of_examples = y.size();
    
    //printf("min_samples_leaf: %d \n", min_samples_leaf);
    splitter = new Splitter(min_samples_leaf,number_of_examples, adaptive_complexity, max_features,learning_rate);


    pred_0 = loss_function->link_function(y.array().mean());
    //pred_0 = 0;
    
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    dVector g = loss_function->dloss(y, pred ).array()*weights.array(); //dVector::Zero(n1,1)
    dVector h = loss_function->ddloss(y, pred ).array()*weights.array();//dVector::Zero(n1,1)

    // dVector g = loss_function->dloss(y,  dVector::Zero(number_of_examples,1));
    // dVector h = loss_function->ddloss(y, dVector::Zero(number_of_examples,1)); 

    root = update_rec(root, X, y, g,h, delta,0,weights);
    root = update_tree_info(X, y, g,h, root,0,weights);

    vector<size_t> sorted_ind = sort_index(obs);

    std::vector<double>  mse_ratio_sorted; 
    std::vector<double> epss_sorted;
    std::vector<double> obs_sorted;
    for (auto i: sorted_ind) {
        mse_ratio_sorted.push_back(mse_ratio[i]);
        epss_sorted.push_back(epss[i]);
        obs_sorted.push_back(obs[i]); 
    }
   
    mse_ratio = mse_ratio_sorted;
    epss = epss_sorted;
    obs = obs_sorted;


} 


Node* TreeReevaluation::update_rec(Node* node, const dMatrix &X,const  dVector &y, const dVector &g, const  dVector &h, double delta, int depth, const dVector &weights){
    if(y.size()<=min_samples_leaf){
        return node;
    }
    if(node->is_leaf()){
        return attempt_split(node, X,y,h,g, depth,weights);//node;//
    }

    bool change;
    tie(node,change) = reevaluate_split(node,X,y,g,h,delta,depth,weights);
    //printf("change %d, depth: %d \n", change, depth);
    if(!change){
    iVector left_mask; iVector right_mask;
    dVector feature = X.col(node->split_feature);
    tie(left_mask,right_mask) = get_masks(feature, node->split_value);
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    dMatrix X_left = X(left_mask,keep_cols); dVector y_left = y(left_mask,1);
    dVector g_left = g(left_mask,1); dVector h_left = h(left_mask,1);
    dVector g_right = g(right_mask,1); dVector h_right = h(right_mask,1);
    dVector weights_left = weights(left_mask,1); dVector weights_right = weights(right_mask,1);
    

    node->left_child = update_rec(node->left_child, X_left, y_left,g_left, h_left, delta,depth+1,weights_left);
    dMatrix X_right= X(right_mask,keep_cols); dVector y_right = y(right_mask,1);
    node->right_child = update_rec(node->right_child, X_right, y_right,g_right,h_right,delta,depth+1,weights_right);
    }
    return node;
}

double TreeReevaluation::hoeffding_bound(double delta, int n){
    return sqrt(log(1/delta)/(2*n));
    //return 
}

Node* TreeReevaluation::attempt_split(Node* node, const dMatrix &X, const dVector &y,const dVector &g, const dVector &h, int depth, const dVector &weights){
    if(max_depth<= depth || y.rows()< min_samples_leaf || y.rows() <2 || y.rows()<min_split_sample){
        node->n_samples = y.rows();
        return node;
    }
    return node; // build_tree(X,y,g,h,depth,node,weights); //
}



tuple<Node*, bool> TreeReevaluation::reevaluate_split(Node* node, const dMatrix &X, const dVector &y,const dVector &g,const dVector &h, double delta, int depth, const dVector &weights){
    bool any_split;
    double w_var;
    double y_var;
    double expected_max_S;
    double new_reduction;
    double split_value;
    int split_feature;
    iVector left_mask; iVector mask_right;
    dVector feature = X.col(node->split_feature );
    tie(left_mask, mask_right) = get_masks(feature, node->split_value);
    double G=g.array().sum(), H=h.array().sum();
    double parent_loss = -((G*G)/H)/(2*g.size());
    double old_reduction = splitter->get_reduction(g,h,left_mask) ; //node->get_split_score();//
    //printf("%f %f %f \n ", parent_loss, (- old_reduction - parent_loss) );
    bool changed = false;
    
    
    tie(any_split, split_feature, split_value, new_reduction, y_var ,w_var,expected_max_S) = find_split(X,y,g,h,node->features_indices);
    new_reduction = new_reduction;
    //printf("old_reduction %f new_reduction %f\n", old_reduction,new_reduction);
    node->n_samples = y.rows();
    double eps = this->hoeffding_bound(delta, node->n_samples);
    //printf("%f / %f = %f\n",new_reduction, old_reduction, new_reduction/old_reduction);
    double ratio = (new_reduction/old_reduction);
    if(ratio > (1.0+alpha) + eps && any_split){
        node = build_tree(X,y,g,h,depth,node,weights);
        // node->split_feature = split_feature;
        // node->split_value = split_value;
        // node->split_score = new_reduction;
        // node->y_var = y_var;
        // node->w_var = w_var*node->parent_expected_max_S;
        // node->left_child->parent_expected_max_S = expected_max_S;
        // node->right_child->parent_expected_max_S = expected_max_S;
        //node = build_tree(X,y,g,h,depth);
        changed = true;
        
    }
    // mse_ratio.push_back(ratio);
    // epss.push_back(1+eps);
    // obs.push_back((double)node->n_samples);
    
    return  tuple<Node*, bool>(node, changed);
}
