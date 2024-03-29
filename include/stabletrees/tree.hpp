#ifndef __TREE_HPP_INCLUDED__

#define __TREE_HPP_INCLUDED__


#include <Eigen/Dense>

#include "node.hpp"
#include "splitter.hpp"
#include "lossfunctions.hpp"


#include <mutex>
#include <vector>
#include <thread>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;


using namespace std;
using namespace Eigen;

#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort


class Tree{

    public:
        Node* root  = NULL;
        explicit Tree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,double learning_rate, unsigned int random_state, double alpha, double beta); 
        explicit Tree(); 
        


        // learning tree
        virtual void learn(const dMatrix  X, const dVector y,const  dVector weights);
        
        
        //update tree 
        virtual void update(const dMatrix X, const dVector y,const dVector weights);
        Node* update_node_obs(const dMatrix &X, Node* node);

        dVector predict_uncertainty(const dMatrix  &X);

        // util
        Node* get_root();
        std::vector<Node*> make_node_list();


        // model prediction
        dVector predict(const dMatrix  &X);

        dMatrix predict_info(const dMatrix &X);

        
        int tree_depth;
        double learning_rate;
        double alpha;
        double beta;
        
        
    protected:
        Splitter* splitter;
        LossFunction* loss_function;
        int max_depth;
        bool adaptive_complexity;
        int _criterion;
        double min_split_sample;
        int min_samples_leaf;
        double total_obs;
        unsigned int random_state;
        double pred_0 = 0;
        int n0;
        double n_delta_fra;
        int max_features;
        int number_of_nodes;
        void make_node_list_rec(Node* node, std::vector<Node*> &l, size_t index );
        int init_random_state;

        virtual Node* build_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, const Node* previuos_tree_node,const dVector &weights);
        virtual tuple<bool,int,double, double,double,double,double> find_split(const dMatrix &X, const dVector &y, const dVector &g, const dVector &h, const std::vector<int> &features_indices);
        tuple<iVector, iVector> get_masks(const dVector &feature, double value);

        double predict_obs(const dVector  &obs);
        dVector predict_info_obs(dVector  &obs);
        

        double predict_uncertainty_obs(const dVector  &obs);


        bool all_same(const dVector &vec);
        bool all_same_features_values(const dMatrix  &X);

        Node* update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h,const dVector gammas, int depth, const Node* previuos_tree_node, const dVector &ypred1, const dVector &weights);
        tuple<bool,int,double, double,double,double,double,double>  find_update_split(const dMatrix &X,const dVector &y, const dVector &g,const dVector &h,const dVector &weights);
};
Tree::Tree(){
    int max_depth = INT_MAX;
    double min_split_sample = 2.0;
    this->_criterion = 0;
    this->adaptive_complexity = false;
    this->min_samples_leaf = 1;
    this->tree_depth = 0;
    this->number_of_nodes = 0;
    this->loss_function = new LossFunction(0);
    this-> max_features =  INT_MAX;
    this->learning_rate = 1;
    this->random_state = 1;
    this->init_random_state = this->random_state ;
    this->alpha = 0;
    this->beta = 0;
}


Tree::Tree(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state, double alpha,double beta){
    //this->splitter = Splitter(_criterion);
    if(max_depth !=NULL){
       this-> max_depth =  max_depth;
    }else{
        this-> max_depth =  INT_MAX;
    }
    this-> min_split_sample = min_split_sample;
    this->min_samples_leaf = min_samples_leaf;
    this->_criterion = _criterion;
    this->adaptive_complexity = adaptive_complexity;
    this->max_features = max_features;
    this->learning_rate = learning_rate;
    this->tree_depth = 0;
    this->number_of_nodes = 0;
    this->loss_function = new LossFunction(_criterion);
    this->random_state = random_state;
    this->init_random_state = this->random_state ;
    this->splitter = new Splitter(min_samples_leaf,0.0, adaptive_complexity,max_features, learning_rate);
    this->alpha = alpha;
    this->beta = beta;
} 

tuple<bool,int,double, double,double,double,double>  Tree::find_split(const dMatrix &X, const dVector &y, const dVector &g, const dVector &h, const std::vector<int> &features_indices){
    return splitter->find_best_split(X, y, g, h,features_indices);
}


bool Tree::all_same(const dVector &vec){
    bool same = true;
    for(int i=0; i< vec.rows(); i++){
        if(vec(i)!=vec(0) ){
            same=false;
            break;
        }
    }
    return same;
}

bool Tree::all_same_features_values(const dMatrix &X){
    bool same = true;
    dVector feature;
    for(int i =0; i<X.cols(); i++){
        feature = X.col(i);
        if(!all_same(feature)){
            same=false;
            break;
        }
    }
    return same;
}

tuple<iVector, iVector> Tree::get_masks(const dVector &feature, double value){
    std::vector<int> left_values;
    std::vector<int> right_values;
    for(int i=0; i<feature.rows();i++){
        if(feature[i]<=value){
            left_values.push_back(i);
        }else{
            right_values.push_back(i);
        }
    }
    iVector left_values_v = Eigen::Map<iVector, Eigen::Unaligned>(left_values.data(), left_values.size());
    iVector right_values_v = Eigen::Map<iVector, Eigen::Unaligned>(right_values.data(), right_values.size());

    return tuple<iVector, iVector> (left_values_v, right_values_v);
}


/**
 * Learn a regression tree from the training set (X, y).
 *
 *
 * @param X Feature matrix.
 * @param y response vector.
 * @param weights Sample weights.
 */
void Tree::learn(const dMatrix  X, const  dVector y, const dVector weights){
    std::lock_guard<std::mutex> lock(mutex); // to make it thread safe when used for parallization in random forest 
    total_obs = y.size();
    n0 = total_obs;
    this->splitter->total_obs = total_obs;
    this->splitter->n0 = n0;

    

    pred_0 = loss_function->link_function(0);

    dVector pred = dVector::Constant(y.size(),0,  pred_0); 

    // compute first and second derivatives for the second-order approximation of the loss
    dVector g = loss_function->dloss(y, pred ).array()*weights.array(); 
    dVector h = loss_function->ddloss(y, pred ).array()*weights.array();

    //build tree recursively
    this->root = build_tree(X, y,g, h, 0, NULL,weights);//
    
}



double Tree::predict_uncertainty_obs(const dVector  &obs){
    Node* node = this->root;
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            return node->w_var;
        }else{
            if(obs(node->split_feature) <= node->split_value){
                node = node->left_child;
            }else{
                node = node->right_child;
            }
        }
    }
    return NULL;
}

dVector Tree::predict_uncertainty(const dMatrix  &X){
    int n = X.rows();
    dVector w_var(n);
    dVector obs(X.cols());
    for(int i =0; i<n; i++){
        dVector obs = X.row(i);
        w_var[i] = predict_uncertainty_obs(obs);
    }
    return w_var;
}


dVector Tree::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(6,1);
    while(node !=NULL){
        if(node->is_leaf()){
            info(0,1) = node->predict();
            
            if(node->w_var <=0){
                node->w_var =0.00001;
            }
            if(node->y_var <=0){
                node->y_var =0.00001;
            }
            if(std::isnan(node->y_var)||std::isnan(node->w_var) || std::isnan((node->y_var/node->w_var)) ){
                    std::cout << "y_var or w_var contains NaN:" << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(node->y_var< 0 || node->w_var <0 || (node->y_var/node->w_var)/node->n_samples<0){
                    std::cout << "y_var or w_var <0: " << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(_criterion ==1){ //poisson only uses prediction variance
                info(1,1) = node->y_var/node->w_var; //Based on experimental tries
                //info(1,1) = 1/(node->w_var/node->n_samples); //based on theory
            }
            else{ //mse uses both response and prediction variance
                //info(1,1) = alpha + beta*(node->y_var/node->w_var/node->n_samples); // alpha + beta* y_var/w_var. equation (x) in article
                double eps = 1.0/100.0;
                info(1,1) = 1/((node->n_samples*node->w_var) +eps ); //(node->y_var/node->w_var); // alpha + beta* y_var/w_var. equation (x) in article
            }
            info(2,1) = 1.0/node->w_var/node->n_samples; ///((double)node->n_samples*(double)node->n_samples);//node->w_var;
            info(3,1) = node->y_var;
            info(4,1) = node->n_samples;
            info(5,1) = 1/node->effect_profiling;
            return info;
        }else{
            if(obs(node->split_feature) <= node->split_value){
                node = node->left_child;
            }else{
                node = node->right_child;
            }
        }
    }
}

dMatrix Tree::predict_info(const dMatrix &X){
    int n = X.rows();
    dMatrix leaf_info(n,6);
    dVector obs(X.cols());
    for(int i =0; i<n; i++){
        dVector obs = X.row(i);
        dVector info =predict_info_obs(obs);
        for (size_t j = 0; j < info.size(); j++)
        {
            leaf_info(i,j) = info(j);
        }
    }
    return leaf_info;
}

double Tree::predict_obs(const dVector  &obs){
    Node* node = this->root;
    while(node !=NULL){
        if(node->is_leaf()){
            return node->predict();
        }else{

            if(obs(node->split_feature) <= node->split_value){
                node = node->left_child;
            }else{
                node = node->right_child;
            }
        }
    }
    return NULL;
}

/**
 * Predict response values for X.
 *
 *
 * @param X Feature matrix.
 */
dVector Tree::predict(const dMatrix  &X){
    int n = X.rows();
    dVector y_pred(n);
    dVector obs(X.cols());
    for(int i =0; i<n; i++){
        dVector obs = X.row(i);
        y_pred[i] = predict_obs(obs);
    }
    return loss_function->inverse_link_function(pred_0 + y_pred.array());//y_pred; //
}

/**
 * Builds a tree based on on X.
 *
 * @param X Feature matrix.
 * @param y response vector.
 * @param g first derivatives.
 * @param h second derivatives.
 * @param depth node depth.
 * @param previuos_tree_node node from previous tree (null if no previous tree).
 * @param weights sample weights.
 */
Node* Tree::build_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, const Node* previuos_tree_node, const dVector &weights){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    if(X.rows()<2 || y.rows()<2){
        printf("X.rows()<2 || y.rows()<2 \n");
        return NULL;
    }

    
    int n = y.size();
    double G = g.array().sum();
    double H = h.array().sum();
    
    double y_sum = (y.array()*weights.array()).sum();
    double sum_weights = weights.array().sum();
    double pred = loss_function->link_function(y_sum/sum_weights) - pred_0;

    //double pred = -G/H;
    
    bool any_split;
    double score;
    
    double split_value;
    double w_var = 1/(double)n;
    double y_var = (double)n*w_var;

    
    
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    double expected_max_S;
    

    std::vector<int> features_indices(X.cols());
    for (int i=0; i<X.cols(); i++){features_indices[i] = i; } 
    
    double loss_parent = (y.array() - pred).square().sum();
    if(all_same(y)){
        //printf("all_same(y) \n");
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    }
    
    
    
    
    
    tie(any_split, split_feature, split_value, score, y_var ,w_var,expected_max_S)  = find_split(X,y, g,h, features_indices);
    
    if(depth>=this->max_depth){
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);

    }
    if(y.rows()< this->min_split_sample){
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    }
    if(!any_split){
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    }


    dVector feature = X.col(split_feature);

    tie(mask_left, mask_right) = get_masks(feature, split_value);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector g_left = g(mask_left,1); dVector h_left = h(mask_left,1);
    dVector g_right = g(mask_right,1); dVector h_right = h(mask_right,1);
    dVector weights_left  = weights(mask_left,1); dVector weights_right = weights(mask_right,1);
    
    Node* node = new Node(split_value, loss_parent/n, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    
    double effect_profiling = (1.0 + expected_max_S)/2.0;
    

    if(previuos_tree_node !=NULL){//only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,previuos_tree_node->left_child,weights_left);
    }else{
        node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,NULL,weights_left);
    }
    if(node->left_child!=NULL){
        node->left_child->w_var*=effect_profiling;
        node->left_child->parent_expected_max_S=expected_max_S;
        node->left_child->effect_profiling=effect_profiling;
        node->left_child->posterior_precision = 1.0/node->left_child->w_var; 
    }
    if(previuos_tree_node !=NULL){ //only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,previuos_tree_node->right_child,weights_right) ;
    }else{
        node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,NULL,weights_right) ;
    }
    if(node->right_child!=NULL){
        node->right_child->w_var *=effect_profiling;
        node->right_child->parent_expected_max_S=expected_max_S;
        node->right_child->effect_profiling=effect_profiling;
        node->right_child->posterior_precision = 1.0/node->right_child->w_var; 
    }
 
    return node;
}


Node* Tree::update_node_obs(const dMatrix &X, Node* node){
    
    node->n_samples = X.rows();
    double eps = 0.0;
    
    if(node->is_leaf()){
        return node;
    }
    dVector feature = X.col(node->split_feature);
    iVector mask_left;
    iVector mask_right;
    tie(mask_left, mask_right) = get_masks(feature, node->split_value);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    dMatrix X_left = X(mask_left,keep_cols); 
    dMatrix X_right = X(mask_right,keep_cols); 
    node->left_child = update_node_obs(X_left,node->left_child) ;
    node->right_child = update_node_obs(X_right,node->right_child) ;
    
    //printf("update %d \n", node->get_split_feature());
    return node;
}


/**
 * Update the tree based on X.
 *
 * @param X Feature matrix.
 * @param y response vector.
 * @param weights sample weights.
 */
void Tree::update(const dMatrix X,const dVector y, const dVector sample_weights){
    random_state = init_random_state;
    std::lock_guard<std::mutex> lock(mutex);

    
    dMatrix info = predict_info(X);
    dVector num_obs_in_nodes = info.col(4).array(); // n_d

    //double min_node_variance = info.col(2).array().minCoeff(); // 1/n sum_i^n w_var/nd
    double average_response_variance = info.col(3).array().mean(); // c^2
    
    n0 = total_obs;
    total_obs = y.size();

    dVector node_variance = info.col(2).array();

    dVector uwr = average_response_variance*info.col(1).array();
    //dVector uwr = info.col(1).array();
   
   

    

    dVector gammas = alpha + beta*uwr.array() ;//*uwr.array(); //gamma_i = y_var_bar(x_i)/(w_var(x_i))
    //std::cout << "min gamma: " <<  gammas.minCoeff() << " mean gamma: " << gammas.array().mean() << " max gamma: " << gammas.maxCoeff() << std::endl;
    pred_0 = loss_function->link_function(0);//
    dVector ypred1 = info.col(0).array() + pred_0; //prediction from previous tree
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    
    dVector g = loss_function->dloss(y.array(), pred,ypred1, gammas,sample_weights ); 
    dVector h = loss_function->ddloss(y.array(), pred,ypred1,gammas,sample_weights); 
    
    splitter->total_obs= total_obs;
    splitter->n0 = n0;

    this->root = update_tree(X, y, g, h, gammas, 0,this->root, ypred1, sample_weights );
}


Node* Tree::update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h,const dVector gammas, int depth, const Node* previuos_tree_node, const dVector &ypred1, const dVector &weights){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    if(X.rows()<2 || y.rows()<2){
        printf("X.rows()<2 || y.rows()<2 \n");
        return NULL;
    }
    
    int n = y.size();
    double G = g.array().sum();
    double H = h.array().sum();
    
    double y_sum = (y.array()*weights.array()).sum();
    
    double ypred1_sum = (ypred1.array() * gammas.array()).sum();
    
    double sum_weights = weights.array().sum();
    double sum_gammas = gammas.array().sum();


    double pred = loss_function->link_function((y_sum+ypred1_sum)/((sum_weights+sum_gammas))) - pred_0;// same as -G/H
    //double pred = -G/H;
    if (abs((-G/H) - pred)>0.0000001){
        std::cout << "-G/H: " <<  -G/H<< " pred: "<< pred << " "<< abs((-G/H) - pred)<< std::endl;
    }
    
    
    
    
    bool any_split;
    double score;
    
    double split_value;
    double w_var = 1/(double)n;
    double y_var = (double)n*w_var;

    if(all_same(y)){
        return new Node(pred, n,y_var,w_var);
    }
    
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    double expected_max_S;
    std::vector<int> features_indices(X.cols());
    for (int i=0; i<X.cols(); i++){features_indices[i] = i; } 


    
    
    tie(any_split, split_feature, split_value, score, y_var ,w_var,expected_max_S)  = find_split(X,y, g,h, features_indices);
    

    if(depth>=this->max_depth){
        return new Node(pred, n, y_var,w_var);
    }
    if(y.rows()< this->min_split_sample){
        return new Node(pred, n, y_var,w_var);
    }
    if(!any_split){

        return new Node(pred ,n, y_var, w_var);
    }



    dVector feature = X.col(split_feature);

    tie(mask_left, mask_right) = get_masks(feature, split_value);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector g_left = g(mask_left,1); dVector h_left = h(mask_left,1);
    dVector g_right = g(mask_right,1); dVector h_right = h(mask_right,1);

    dVector gammas_left = gammas(mask_left,1); dVector gammas_right = gammas(mask_right,1); 
    
    dVector ypred1_left = ypred1(mask_left,1); dVector ypred1_right = ypred1(mask_right,1);

    dVector weights_left  = weights(mask_left,1); dVector weights_right = weights(mask_right,1);

    double loss_parent = (y.array() - pred).square().sum();
    
    Node* node = new Node(split_value, loss_parent/n, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    double effect_profiling = (1.0 + expected_max_S)/2.0; 
    

    node->left_child = update_tree( X_left, y_left, g_left,h_left,gammas_left, depth+1,NULL,ypred1_left,weights_left);
    
    if(node->left_child!=NULL){
        node->left_child->w_var*=effect_profiling;
        node->left_child->parent_expected_max_S=expected_max_S;
        node->left_child->effect_profiling=effect_profiling;     
    }
    
    node->right_child = update_tree(X_right, y_right,g_right,h_right, gammas_right, depth+1,NULL,ypred1_right,weights_right) ;
    
    if(node->right_child!=NULL){
        node->right_child->w_var *=effect_profiling;
        node->right_child->parent_expected_max_S=expected_max_S;
        node->right_child->effect_profiling=effect_profiling;

    }
    return node;
}

Node* Tree::get_root(){
    return this->root;
}

#endif