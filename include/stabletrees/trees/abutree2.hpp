#ifndef __ABUTREE2_HPP_INCLUDED__

#define __ABUTREE2_HPP_INCLUDED__
#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"


#include <mutex>
#include <vector>
#include <thread>

using namespace std;
using namespace Eigen;

class AbuTree2: public Tree{
    public:
        AbuTree2(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state, double alpha, double beta);
        AbuTree2();
        virtual void update(const dMatrix X,const dVector y, const dVector sample_weights);
        dMatrix predict_info(const dMatrix &X);
        tuple<bool,int,double, double,double,double,double,double>  AbuTree2::find_update_split(const dMatrix &X,const dVector &y, const dVector &g,const dVector &h,const dVector &weights);
        Node* update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h,const dVector gammas, int depth, const Node* previuos_tree_node, const dVector &ypred1, const dVector &weights);
        Node* update_node_obs(const dMatrix &X, Node* node);
    private:
        dVector predict_info_obs(dVector  &obs);
        dMatrix sample_X(const dMatrix &X, int n1);
        int bootstrap_seed ;
        int init_random_state ;
        double alpha;
        double beta;
};

AbuTree2::AbuTree2():Tree(){
    Tree(); 
    bootstrap_seed=0;
    this->alpha = 0;
    this->beta = 1.0;
}

AbuTree2::AbuTree2(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, double learning_rate, unsigned int random_state, double alpha,double beta):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf,adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
    bootstrap_seed=0;
    init_random_state = random_state;
    this->alpha = alpha;
    this->beta = beta;
}


dVector AbuTree2::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(5,1);
    while(node !=NULL){
        if(node->is_leaf()){
            info(0,1) = node->predict();
            
            if(node->w_var <=0){
                node->w_var =0.00001;
            }
            if(node->y_var <=0){
                node->y_var =0.00001;
            }
            if(std::isnan(node->y_var)||std::isnan(node->w_var) || std::isnan((node->y_var/node->w_var)/node->n_samples) ){
                    std::cout << "y_var or w_var contains NaN:" << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(node->y_var< 0 || node->w_var <0 || (node->y_var/node->w_var)/node->n_samples<0){
                    std::cout << "y_var or w_var <0: " << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(_criterion ==1){ //poisson only uses prediction variance
                info(1,1) = node->y_var/node->w_var/node->n_samples; //Based on experimental tries
                //info(1,1) = 1/(node->w_var/node->n_samples); //based on theory
            }
            else{ //mse uses both response and prediction variance
                //std::cout << alpha << " " << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                info(1,1) = alpha + beta*(node->y_var/node->w_var/node->n_samples);
            }
            //std::cout << "y_var or w_var contains:" << node->y_var << " " <<node->w_var << " " << node->n_samples<<  " " << n1<<  std::endl;
            info(2,1) = node->w_var;
            info(3,1) = node->y_var;
            info(4,1) = node->n_samples;
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
dMatrix AbuTree2::predict_info(const dMatrix &X){
    int n = X.rows();
    dMatrix leaf_info(n,5);
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

Node* AbuTree2::update_node_obs(const dMatrix &X, Node* node){
    
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


void AbuTree2::update(const dMatrix X,const dVector y, const dVector sample_weights){
    random_state = init_random_state;
    // bootstrap_seed = init_random_state;
    std::lock_guard<std::mutex> lock(mutex);
    //this->root = update_node_obs(X,this->root);
    dMatrix info = predict_info(X);
    dVector gammas = info.col(1).array(); //gamma

    pred_0 = loss_function->link_function(y.array().mean());//
    dVector ypred1 = info.col(0).array() + pred_0; //prediction from previous tree
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    
    total_obs = y.size();
    dVector g = loss_function->dloss(y.array(), pred,ypred1, gammas,sample_weights ); 
    dVector h = loss_function->ddloss(y.array(), pred,ypred1,gammas,sample_weights); 
    splitter->total_obs= total_obs;

    this->root = update_tree(X, y, g, h, gammas, 0,this->root, ypred1, sample_weights );


}


dMatrix AbuTree2::sample_X(const dMatrix &X, int n1){
    std::mt19937 gen(bootstrap_seed);
    std::uniform_int_distribution<size_t>  distr(0, X.rows()-1);
    dMatrix X_sample(n1, X.cols());
    for (size_t i = 0; i < n1; i++)
    {   
        size_t ind = distr(gen);
        for (size_t j = 0; j < X.cols(); j++)
        {   
            double x_b = X(ind,j);
            X_sample(i,j) = x_b;
        } 
    }
    bootstrap_seed+=1;
    return X_sample;
}

Node* AbuTree2::update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h,const dVector gammas, int depth, const Node* previuos_tree_node, const dVector &ypred1, const dVector &weights){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    if(X.rows()<2 || y.rows()<2){
        printf("X.rows()<2 || y.rows()<2 \n");
        return NULL;
    }

    double eps = 0.0;
    if(_criterion ==1){ // for poisson need to ensure not log(0)
        eps=0.0000000001;
    }
    int n = y.size();
    double G = g.array().sum();
    double H = h.array().sum();
    
    double y_sum = (y.array()*weights.array()).sum();
    
    double ypred1_sum = (ypred1.array() * gammas.array()).sum();
    
    // if(ypred1_sum !=0){
    //     printf("%f\n",ypred1_sum );
    // }
    double sum_weights = weights.array().sum();
    double sum_gammas = gammas.array().sum();


    // for (size_t i = 0; i < gammas.size(); i++)
    // {
    //     printf("gamma = %f \n", gammas(i));
    // }

    // if(y_sum+ypred1_sum !=y_sum){
    //     printf("%f\n",ypred1_sum );
    // }
    // if((1+gamma)*sum_weights !=sum_weights){
    //     printf("%f\n",sum_weights );
    // }
    // printf("%f %f %f %f\n",y_sum,ypred1_sum,sum_weights,gamma );
    double pred = loss_function->link_function((y_sum+ypred1_sum)/((sum_weights+sum_gammas)) + eps) - pred_0;
   
    //double pred = -G/H;
    if(std::isnan(pred)|| std::isinf(pred)){//|| abs(pred +G/H)>0.000001
        std::cout << "pred: " << pred << std::endl;
        std::cout << "G: " << G << std::endl;
        std::cout << "H: " << H << std::endl;
        std::cout << "diff: " << abs(pred + G/H)<< std::endl;
        std::cout << "n: " << n << std::endl;
        std::cout << "y_sum: " << y_sum << std::endl;
        std::cout << "ypred1_sum: " << ypred1_sum << std::endl;
        std::cout << "ypred1 size: " << ypred1.size() << std::endl;
        std::cout << "y size: " << y.size() << std::endl;
        std::cout << "y: " << y.array().sum() << std::endl;
        throw exception("pred is nan or inf: %f \n",pred);

    }
    bool any_split;
    double score;
    
    double split_value;
    double w_var = 1;
    double y_var = 1;

    if(all_same(y)){
        //printf("all_same(y) \n");
        return new Node(pred, n,y_var,w_var);
    }
    
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    double expected_max_S;
    std::vector<int> features_indices(X.cols());
    for (int i=0; i<X.cols(); i++){features_indices[i] = i; } 
     if(previuos_tree_node ==NULL){

        if(max_features<INT_MAX){
            std::mt19937 gen(random_state);
            std::iota(features_indices.begin(), features_indices.end(), 0);
            std::shuffle(features_indices.begin(), features_indices.end(), gen);
            features_indices.resize(max_features);
            features_indices.erase(features_indices.begin() + max_features, features_indices.end());
            
            
            // for (int i=0; i<X.cols(); i++){
            //     printf("%d %d\n", features_indices[i], features_indices.size());
            // } 
            // printf("\n");
            
        }
    }else 
    if(previuos_tree_node->get_features_indices().size()>0) {
        //features_indices.resize(max_features);
        //printf("else if %d\n", features_indices.size());
        features_indices = previuos_tree_node->get_features_indices();
    }
    this->random_state +=1;

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
        return new Node(pred, n, y_var,w_var);
    }
    if(y.rows()< this->min_split_sample){
        //printf("min_split_sample \n");
        return new Node(pred, n, y_var,w_var);
    }
    if(!any_split){
        //printf("any_split \n");
        return new Node(pred ,n, y_var, w_var);
    }

    if(score == std::numeric_limits<double>::infinity()){
        printf("X.size %d y.size %d, reduction %f, expected_max_S %f, min_samples_leaf = %d \n", X.rows(), y.rows(),score,expected_max_S, min_samples_leaf);
        // cout<<"\n Two Dimensional Array is : \n";
        // for(int r=0; r<X.rows(); r++)
        // {
        //         for(int c=0; c<X.cols(); c++)
        //         {
        //                 cout<<" "<<X(r,c)<<" ";
        //         }
        //         cout<<"\n";
        // }
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

    dVector gammas_left = gammas(mask_left,1); dVector gammas_right = gammas(mask_right,1); 
    
    dVector ypred1_left = ypred1(mask_left,1); dVector ypred1_right = ypred1(mask_right,1);

    dVector weights_left  = weights(mask_left,1); dVector weights_right = weights(mask_right,1);

    double loss_parent = (y.array() - pred).square().sum();
    //printf("loss_parent %f \n" ,loss_parent);
    // dVector pred_left = dVector::Constant(y_left.size(),0,loss_function->link_function(y_left.array().mean()));
    // dVector pred_right = dVector::Constant(y_right.size(),0,loss_function->link_function(y_right.array().mean()));
    // double loss_left = (y_left.array() - y_left.array().mean()).square().sum();
    // double loss_right = (y_right.array() - y_right.array().mean()).square().sum();
    // printf("score comparison: %f, %f \n", score, (loss_parent - (loss_left+loss_right))/n);
    
    Node* node = new Node(split_value, loss_parent/n, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    
    if(previuos_tree_node !=NULL){//only applicable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->left_child = update_tree( X_left, y_left, g_left,h_left,gammas_left, depth+1,previuos_tree_node->left_child,ypred1_left,weights_left);
    }else{
        node->left_child = update_tree( X_left, y_left, g_left,h_left,gammas_left, depth+1,NULL,ypred1_left,weights_left);
    }
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
        node->left_child->parent_expected_max_S=expected_max_S;
    }
    if(previuos_tree_node !=NULL){ //only applicable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->right_child = update_tree(X_right, y_right,g_right,h_right, gammas_right, depth+1,previuos_tree_node->right_child,ypred1_right,weights_right) ;
    }else{
        node->right_child = update_tree(X_right, y_right,g_right,h_right, gammas_right, depth+1,NULL,ypred1_right,weights_right) ;
    }
    if(node->right_child!=NULL){
        node->right_child->w_var *=expected_max_S;
        node->right_child->parent_expected_max_S=expected_max_S;
    }

    return node;
}
   

#endif
