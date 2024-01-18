#ifndef __TREE_HPP_INCLUDED__

#define __TREE_HPP_INCLUDED__


#include <Eigen/Dense>

#include "node.hpp"
#include "splitter.hpp"
#include "lossfunctions.hpp"
#include "initial_prediction.hpp"


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
        explicit Tree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,double learning_rate, unsigned int random_state); 
        explicit Tree(); 
        bool all_same(const dVector &vec);
        bool all_same_features_values(const dMatrix  &X);
        virtual Node* build_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, const Node* previuos_tree_node,const dVector &weights);
        tuple<iVector, iVector> get_masks(const dVector &feature, double value);
        virtual void learn(const dMatrix  X, const dVector y,const  dVector weights);
        void learn_difference(const dMatrix  X,const dVector y, const dVector g, const dVector h, const dVector weights);
        Node* get_root();
        double predict_obs(const dVector  &obs);
        double predict_uncertainty_obs(const dVector  &obs);
        dVector predict(const dMatrix  &X);
        dVector predict_uncertainty(const dMatrix  &X);
        virtual void update(const dMatrix X, const dVector y,const dVector weights);
        virtual tuple<bool,int,double, double,double,double,double> find_split(const dMatrix &X, const dVector &y, const dVector &g, const dVector &h, const std::vector<int> &features_indices);
        //Node* update_tree_info(dMatrix &X, dVector &y, Node* node, int depth);
        Node* update_tree_info(const dMatrix &X,const  dVector &y, const dVector &g ,const dVector &h, Node* node, int depth, const dVector &weights);
        //~Tree();
        std::vector<Node*> make_node_list();
        //virtual Tree* copy();
        Tree* next_tree = NULL; // only needed for gradient boosting
        int tree_depth;
        double learning_rate; // only needed for gradient boosting (shrinkage)
        // void serialize(Tree* tptr, std::ofstream& f);
        // bool deSerialize(Tree* tptr,  std::ifstream& f);
        
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
        int n1;
        int max_features;
        int number_of_nodes;
        void make_node_list_rec(Node* node, std::vector<Node*> &l, size_t index );
        int init_random_state;
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
}

// ~DecisionTree() {
//         // all automatically recycled
//     }

//  Tree* Tree::copy(){
//     Tree* tree = new Tree(*this);
//     tree->root = root->copy();
//     return tree;
//  }
 
// void Tree::serialize(Tree *tptr, std::ofstream& f)
// {
    
//     int MARKER = -1;
//     // If current tree is NULL, store marker
//     if(tptr == NULL)
//     {
//         f << MARKER;
//         return;
//     }
//     tptr->root->serialize(f);
// }
// bool Tree::deSerialize(Tree *tptr, std::ifstream& f)
// {
//     tptr->root = new Node;
//     const bool success = tptr->root->deSerialize(tptr->root, f);
//     if (!success) {
//         tptr = NULL;
//         return false;
//     }

//     Tree* new_tree = new Tree;
//     bool new_tree_exist = deSerialize(new_tree, f);
//     if(new_tree_exist)
//     {
//         tptr->next_tree = new_tree;
//     }else{
//         tptr->next_tree = NULL;
//     }
    
//     return true;
// }


Tree::Tree(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state){
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
    this->splitter->total_obs = total_obs;
    //printf("min_samples_leaf: %d \n", min_samples_leaf);
    
    n1 = total_obs;

    // for (size_t i = 0; i < weights.size(); i++)
    // {
    //     printf("w %f\n", weights(i));
    //}
    
    //dVector offset =  dVector::Constant(y.size(),0,  0);
    pred_0 = loss_function->link_function(y.mean()) ; //learn_initial_prediction(y,offset,loss_function); //
    //pred_0 = 0;
    //printf("pred_0 %f %f \n", pred_0, loss_function->link_function(y.array().mean()+y.array().mean()/2));
    dVector pred = dVector::Constant(y.size(),0,  pred_0); 

    // compute first and second derivatives for the second-order approximation of the loss
    dVector g = loss_function->dloss(y, pred ).array()*weights.array(); 
    dVector h = loss_function->ddloss(y, pred ).array()*weights.array();

    //build tree recursively
    this->root = build_tree(X, y,g, h, 0, NULL,weights);//
    
}


void Tree::learn_difference(const dMatrix  X, const dVector y, const dVector g, const dVector h,const  dVector weights){
    total_obs = y.size();
    //printf("min_samples_leaf: %d \n", min_samples_leaf);
    splitter = new Splitter(min_samples_leaf,total_obs, adaptive_complexity,max_features, learning_rate );
    n1 = total_obs;
    
    this->root = build_tree(X, y,g, h, 0, NULL,weights);//
}


double Tree::predict_uncertainty_obs(const dVector  &obs){
    Node* node = this->root;
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            return node->w_var;
        }else{
            //printf("feature %d, value %f, obs %f \n", node->split_feature, node->split_value,obs(node->split_feature));
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


double Tree::predict_obs(const dVector  &obs){
    Node* node = this->root;
    while(node !=NULL){
        if(node->is_leaf()){
            //printf("prediction %f \n", node->predict());
            return node->predict();
        }else{
            //printf("feature %d, value %f, obs %f \n", node->split_feature, node->split_value,obs(node->split_feature));
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
    
   
    double eps = 0.0;
    if(_criterion ==1){ // for poisson need to ensure not log(0)
        eps=0.0000000001;
    }

    double y_sum = (y.array()*weights.array()).sum();
    double sum_weights = weights.array().sum();
    double pred = loss_function->link_function(y_sum/sum_weights+eps) - pred_0;
    //printf("%f \n", pred);
    //double pred = -G/H;
    //printf("-g/h = %f, y.mean() = %f, -G/H = %f \n", pred, y.array().mean(),pred_0-G/H);
    if(std::isnan(pred)|| std::isinf(pred)){
        std::cout << "tree_depth: " << tree_depth << std::endl;
        std::cout << "_criterion: " << _criterion << std::endl;
        std::cout << "eps: " << eps << std::endl;
        std::cout << "pred: " << pred << std::endl;
        std::cout << "G: " << G << std::endl;
        std::cout << "H: " << H << std::endl;
        std::cout << "y_sum: " << y_sum << std::endl;

        throw exception("pred is nan or inf: %f \n",pred);

    }
    bool any_split;
    double score;
    
    double split_value;
    double w_var = 1;
    double y_var = 1;

    
    
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    double expected_max_S;
    //iVector features_indices(X.cols(),1);

    std::vector<int> features_indices(X.cols());
    for (int i=0; i<X.cols(); i++){features_indices[i] = i; } 
    // if(previuos_tree_node ==NULL){

    //     if(max_features<INT_MAX){
    //         std::mt19937 gen(random_state);
    //         std::iota(features_indices.begin(), features_indices.end(), 0);
    //         std::shuffle(features_indices.begin(), features_indices.end(), gen);
    //         features_indices.resize(max_features); 
    //         features_indices.erase(features_indices.begin() + max_features, features_indices.end());
            
    //     }
    // }else 
    // if(previuos_tree_node->get_features_indices().size()>0) {
    //     //features_indices.resize(max_features);
    //     features_indices = previuos_tree_node->get_features_indices();
    // }
    // this->random_state +=1;
    double loss_parent = (y.array() - pred).square().sum();
    if(all_same(y)){
        //printf("all_same(y) \n");
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    }
    // if(previuos_tree_node ==NULL){
    //     //for (int i=0; i<X.cols(); i++){features_indices(i) = i; } 
    
    //     if(max_features<INT_MAX){
    //         thread_local std::mt19937 gen(omp_get_thread_num());
    //         std::shuffle(features_indices.data(), features_indices.data() + features_indices.size(), gen);
    //         features_indices = features_indices.block(0,0,max_features,1);
    //         //this->random_state +=1;
    //     }
    // }else 
    // if(previuos_tree_node->get_features_indices().size()>0) {
    //     features_indices = previuos_tree_node->get_features_indices();
    // }


    //printf("%d \n", features_indices.allFinite());
    
    
    
    
    tie(any_split, split_feature, split_value, score, y_var ,w_var,expected_max_S)  = find_split(X,y, g,h, features_indices);
    if(any_split && (std::isnan(y_var)||std::isnan(w_var))){
        double G=g.array().sum(), H=h.array().sum(), G2=g.array().square().sum(), H2=h.array().square().sum(), gxh=(g.array()*h.array()).sum();
        double optimism = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n);

        std::cout << "y_var: " << y_var << std::endl;
        std::cout << "w_var: "<< w_var << std::endl;
        std::cout << "n: "<< n << std::endl;
        std::cout << "optimism: "<< optimism << std::endl;
        std::cout << "expected_max_S: "<< expected_max_S << std::endl;
        
        
        double y_0 = y(0);
        bool same = true;
        std::cout << "y"<<0 <<": "<< y_0 << std::endl;


        for (size_t i = 1; i < y.size(); i++)
        {
            if(y_0 != y(i)){
                same = false;
            }
            if(std::isnan(y_0) ||std::isnan(y(i))  ){
                std::cout << "nan detected: "<< i << std::endl;
            }
            if(std::isnan(g(i))  ){
                std::cout << "g"<<i <<": "<< g(i) << std::endl;
            }
        
        }
        std::cout << "all same: "<< same << std::endl;
        throw exception("something wrong!") ;

    }
    if(depth>=this->max_depth){
        //printf("max_depth: %d >= %d \n", depth,this->max_depth);
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
        //return new Node(pred, n, y_var,w_var,features_indices);
    }
    if(y.rows()< this->min_split_sample){
        //printf("min_split_sample \n");
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    }
    if(!any_split){
        //printf("any_split \n");
        return new Node(split_value, loss_parent, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    }

    // if(depth>=this->max_depth){
    //     //printf("max_depth: %d >= %d \n", depth,this->max_depth);
    //     return new Node(pred, n, y_var,w_var);
    // }
    // if(y.rows()< this->min_split_sample){
    //     //printf("min_split_sample \n");
    //     return new Node(pred, n, y_var,w_var);
    // }
    // if(!any_split){
    //     //printf("any_split \n");
    //     return new Node(pred ,n, y_var, w_var);
    // }

    if(score == std::numeric_limits<double>::infinity()){
        printf("X.size %d y.size %d, reduction %f, expected_max_S %f, min_samples_leaf = %d \n", X.rows(), y.rows(),score,expected_max_S, min_samples_leaf);
        cout<<"\n Two Dimensional Array is : \n";
        for(int r=0; r<X.rows(); r++)
        {
                for(int c=0; c<X.cols(); c++)
                {
                        cout<<" "<<X(r,c)<<" ";
                }
                cout<<"\n";
        }
         cout<<"\n one Dimensional Array is : \n";
        for(int c=0; c<y.size(); c++)
        {
                cout<<" "<<y(c)<<" ";
        }
        cout<<"\n";
    }
   

    dVector feature = X.col(split_feature);

    tie(mask_left, mask_right) = get_masks(feature, split_value);
    
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
    dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
    dVector g_left = g(mask_left,1); dVector h_left = h(mask_left,1);
    dVector g_right = g(mask_right,1); dVector h_right = h(mask_right,1);
    dVector weights_left  = weights(mask_left,1); dVector weights_right = weights(mask_right,1);


    
    //printf("loss_parent %f \n" ,loss_parent);
    // dVector pred_left = dVector::Constant(y_left.size(),0,loss_function->link_function(y_left.array().mean()));
    // dVector pred_right = dVector::Constant(y_right.size(),0,loss_function->link_function(y_right.array().mean()));
    // double loss_left = (y_left.array() - y_left.array().mean()).square().sum();
    // double loss_right = (y_right.array() - y_right.array().mean()).square().sum();
    // printf("score comparison: %f, %f \n", score, (loss_parent - (loss_left+loss_right))/n);
    
    Node* node = new Node(split_value, loss_parent/n, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    
    if(previuos_tree_node !=NULL){//only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,previuos_tree_node->left_child,weights_left);
    }else{
        node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,NULL,weights_left);
    }
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
        node->left_child->parent_expected_max_S=expected_max_S;
    }
    if(previuos_tree_node !=NULL){ //only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,previuos_tree_node->right_child,weights_right) ;
    }else{
        node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,NULL,weights_right) ;
    }
    if(node->right_child!=NULL){
        node->right_child->w_var *=expected_max_S;
        node->right_child->parent_expected_max_S=expected_max_S;
    }

    return node;
}


/**
 * Update leaf predictions.
 *
 * @param X Feature matrix.
 * @param y response vector.
 * @param g first derivatives.
 * @param h second derivatives.
 * @param depth node depth.
 * @param previuos_tree_node node from previous tree (null if no previous tree).
 * @param weights sample weights.
 */
Node* Tree::update_tree_info(const dMatrix &X, const dVector &y, const dVector &g , const dVector &h, Node* node, int depth, const dVector &weights){
    tree_depth = max(tree_depth, depth);
    node->n_samples = y.size();
    double eps = 0.0;
    if(_criterion ==1){ // for poisson need to ensure not log(0)
        eps=0.0000000001;
    }
    if(node->n_samples<1){
        return NULL;
    }else if(node->n_samples<1){
        node->prediction = y[0] - pred_0;
    }else{
        double y_mean = (y.array()*weights.array()).sum()/weights.array().sum();
        double pred = loss_function->link_function(y_mean+eps) - pred_0;
        node->prediction = pred; //-g.array().sum()/h.array().sum();
    }
    if(node->is_leaf()){
        return node;
    }
    dVector feature = X.col(node->split_feature);
    iVector mask_left;
    iVector mask_right;
    tie(mask_left, mask_right) = get_masks(feature, node->split_value);
    if(mask_left.size()<1 || mask_right.size()<1 ){
        //printf("null %d %d \n", mask_left.size()<1, mask_right.size()<1);
        node->left_child = NULL;
        node->right_child = NULL;
    }else{
        iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
        dMatrix X_left = X(mask_left,keep_cols); dVector y_left = y(mask_left,1);
        dMatrix X_right = X(mask_right,keep_cols); dVector y_right = y(mask_right,1);
        dVector g_left = g(mask_left,1); dVector h_left = h(mask_left,1);
        dVector g_right = g(mask_right,1); dVector h_right = h(mask_right,1);
        dVector weights_left  = weights(mask_left,1); dVector weights_right = weights(mask_right,1);

        //printf("update rec %d %d \n", mask_left.size()<1, mask_right.size()<1);

        node->left_child = update_tree_info(X_left, y_left,g_left,h_left, node->left_child, depth+1,weights_left) ;
        node->right_child = update_tree_info(X_right, y_right,g_right,h_right, node->right_child, depth+1,weights_right) ;
    }
    //printf("update %d \n", node->get_split_feature());
    return node;
}


void Tree::make_node_list_rec(Node* node, std::vector<Node*> &l, size_t index ){
    if(node->is_leaf()){
        return;
    }
    size_t left_index = index*2+1;
    size_t right_index = index*2+2;
    l[left_index] = node->left_child;
    l[right_index] = node->right_child;
    make_node_list_rec(node->left_child, l, left_index);
    make_node_list_rec(node->right_child, l, right_index);

}

std::vector<Node*> Tree::make_node_list(){
    size_t index = 0;
    int max_number_of_nodes = pow(2,tree_depth+1)-1;
    std::vector<Node*> l(max_number_of_nodes);
    Node* node = root;
    l[index] = node;
    make_node_list_rec(node,l, index);

    return l;
}


/**
 * Update the tree based on X.
 *
 * @param X Feature matrix.
 * @param y response vector.
 * @param weights sample weights.
 */
void Tree::update(const dMatrix X, const  dVector y,const dVector weights){
    this->random_state = this->init_random_state; //
    this->learn(X,y,weights);
}

Node* Tree::get_root(){
    return this->root;
}

#endif