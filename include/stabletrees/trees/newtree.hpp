#pragma once


#include <Eigen/Dense>

#include "node.hpp"
#include "splitter.hpp"
#include "lossfunctions.hpp"
#include "initial_prediction.hpp"



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


class NewTree{

    public:
        Node* root  = NULL;
        NewTree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,double learning_rate, unsigned int random_state); 
        bool all_same(const dVector &vec);
        bool all_same_features_values(const dMatrix  &X);
        virtual Node* build_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, Node* previuos_tree_node,const dVector &indicator, const dVector &gamma);// )
        tuple<iVector, iVector> get_masks(dVector &feature, double value);
        virtual void learn(dMatrix  &X, dVector &y);
        void learn_difference(dMatrix  &X, dVector &y, dVector &g, dVector &h);
        Node* get_root();
        double predict_obs(dVector  &obs);
        dVector predict(dMatrix  &X);
        virtual void update(dMatrix &X, dVector &y);
        virtual tuple<bool,int,double, double,double,double,double> find_split(const dMatrix &X, const dVector &y, const dVector &g, const dVector &h, const std::vector<int> &features_indices);
        //Node* update_tree_info(dMatrix &X, dVector &y, Node* node, int depth);
        NewTree* next_tree = NULL; // only needed for gradient boosting
        int tree_depth;
        double learning_rate; // only needed for gradient boosting (shrinkage)
        double gradient_descent(double w_0, const dVector &y, const dVector &indicator, const dVector &gamma);
        dMatrix predict_info(dMatrix &X);
    private:
        dVector predict_info_obs(dVector  &obs);
        dMatrix sample_X(const dMatrix &X, int n1);
        int bootstrap_seed = 0;
        
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
};




NewTree::NewTree(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state){
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
    tree_depth = 0;
    number_of_nodes = 0;
    loss_function = new LossFunction(_criterion);
    this->random_state = random_state;

} 

tuple<bool,int,double, double,double,double,double>  NewTree::find_split(const dMatrix &X, const dVector &y, const dVector &g, const dVector &h, const std::vector<int> &features_indices){
    return splitter->find_best_split(X, y, g, h,features_indices);
}


bool NewTree::all_same(const dVector &vec){
    bool same = true;
    for(int i=0; i< vec.rows(); i++){
        if(vec(i)!=vec(0) ){
            same=false;
            break;
        }
    }
    return same;
}

bool NewTree::all_same_features_values(const dMatrix &X){
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

tuple<iVector, iVector> NewTree::get_masks(dVector &feature, double value){
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



void NewTree::learn(dMatrix  &X, dVector &y){
    bootstrap_seed=0;
    total_obs = y.size();
    //printf("min_samples_leaf: %d \n", min_samples_leaf);
    splitter = new Splitter(min_samples_leaf,total_obs, adaptive_complexity,max_features, learning_rate);
    n1 = total_obs;


    dVector offset =  dVector::Constant(y.size(),0,  0);
    pred_0 = loss_function->link_function(y.array().mean());//learn_initial_prediction(y,offset,loss_function); //
    //pred_0 = 0;
    //printf("pred_0 %f %f \n", pred_0, loss_function->link_function(y.array().mean()+y.array().mean()/2));
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    dVector g = loss_function->dloss(y, pred ); //dVector::Zero(n1,1)
    dVector h = loss_function->ddloss(y, pred ); //dVector::Zero(n1,1)
    dVector gamma = dVector::Constant(y.size(),0,  0) ;
    dVector indicator = dVector::Constant(y.size(),0,  1) ;
    
    this->root = build_tree(X, y,g, h, 0, NULL,indicator,gamma);//
    
}




double NewTree::predict_obs(dVector  &obs){
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
    throw exception("no leaf node found, not allowed");
    return NULL;
}

dVector NewTree::predict(dMatrix  &X){
    int n = X.rows();
    dVector y_pred(n);
    dVector obs(X.cols());
    for(int i =0; i<n; i++){
        dVector obs = X.row(i);
        y_pred[i] = predict_obs(obs);
    }
    return loss_function->inverse_link_function(pred_0 + y_pred.array());//y_pred; //
}

double NewTree::gradient_descent(double w_0, const dVector &y, const dVector &indicator, const dVector &gamma){
    double tolerance = 1E-9;
    for (int j = 0; j < 10; j++)
    {
        double G = 0;
        double H = 0;
        for (size_t i = 0; i < y.size(); i++)
        {
            if( (std::isnan(indicator[i])||std::isinf(indicator[i])) || (indicator[i] !=0 && indicator[i] !=1 )){
                std::cout << "indicator[i]: " << indicator[i] << std::endl;
                throw exception("indicator[i] is not 0 or 1");
            }
            double g = loss_function->dloss(y[i], w_0);
            double h = loss_function->ddloss(y[i], w_0);
            if( std::isnan(g)||std::isinf(g) ){
                std::cout << "g: " << g << std::endl;
                throw exception("g is nan or inf");
            }
            if( (std::isnan(gamma[i])||std::isinf(gamma[i]))){
                std::cout << "gamma[i]: " << gamma[i] << std::endl;
                throw exception("gamma[i] is nan or inf");
            }
            G += indicator[i]*g + (1-indicator[i])*2*gamma[i]*(w_0 - y[i] );
            H += indicator[i]*h+ (1-indicator[i])*2*gamma[i];
        }
        printf("%d %f %f \n", j, w_0, y.array().mean());
        w_0-= 0.1*G/H;

        if(std::abs(G) <= tolerance){
            return w_0;
        }
    }
    return w_0;
}
//
Node* NewTree::build_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, Node* previuos_tree_node, const dVector &indicator, const dVector &gamma){
    number_of_nodes +=1;
    tree_depth = max(depth,tree_depth);
    if(X.rows()<2 || y.rows()<2){
        printf("X.rows()<2 || y.rows()<2 \n");
        return NULL;
    }

    
    int n = y.size();
    //printf("n: %d \n", n );
    double G = g.array().sum();
    double H = h.array().sum();
    
    
    
    //printf("%f %f / %f , %d , %f \n", damping_h, (H/(double)n), exp(log(y.array().mean())), n, H);
    //double pred =  loss_function->link_function(y.array().mean()) ;//-G/H;//-damping_h*G/H; // //loss_function->inverse_link_function(0)-g.array().sum()/h.array().sum();//y.array().mean();//pred_0-G/H;//loss_function->link_function(y.array().mean()) - pred_0;//
    //printf("-g/h = %f, y.mean() = %f, -G/H = %f \n", pred, y.array().mean(),pred_0-G/H);

    double n2 = indicator.array().sum();
    double gamma_sum = gamma.array().sum();
    double y_sum = (y.array()*indicator.array()).sum();
    double yprev_sum = (gamma.array()*y.array()*(1-indicator.array())).sum();
    double pred = loss_function->link_function((y_sum+yprev_sum)/(n2+gamma_sum)) - pred_0;

    //double pred =  loss_function->link_function(y.array().mean()) - pred_0;//-G/H;//loss_function->link_function(y.array().mean()) - pred_0;//loss_function->link_function((y_sum+yprev_sum)/(n2+gamma_sum)) - pred_0;
    if(std::isnan(pred)|| std::isinf(pred)){
        std::cout << "pred: " << pred << std::endl;
        std::cout << "G: " << G << std::endl;
        std::cout << "H: " << H << std::endl;
        std::cout << "n2: " << n2 << std::endl;
        std::cout << "gamma_sum: " << gamma_sum << std::endl;
        std::cout << "yprev_sum: " << yprev_sum << std::endl;
        throw exception("pred is nan or inf: %f",pred);
    }
    bool any_split;
    double score;
    
    double split_value;
    double w_var = 1;
    double y_var = 1;

    if(all_same(y)){
        return new Node(pred, n,y_var,w_var);
    }
    
    int split_feature;
    iVector mask_left;
    iVector mask_right;
    double expected_max_S;
    std::vector<int> features_indices(X.cols(),1);
    for (int i=0; i<X.cols(); i++){features_indices[i] = i; } 
    // if(previuos_tree_node ==NULL){
    //     //for (int i=0; i<X.cols(); i++){features_indices(i) = i; } 
    
    //     if(max_features<INT_MAX){
    //         std::mt19937 gen(random_state);
    //         std::shuffle(features_indices.data(), features_indices.data() + features_indices.size(), gen);
    //         features_indices = features_indices.block(0,0,max_features,1);
    //         this->random_state +=1;
    //     }
    // }else if(previuos_tree_node->get_features_indices().size()>0) {
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
        //pred = gradient_descent(pred, y.array()  ,indicator,gamma);

        return new Node(pred, n, y_var,w_var);
    }
    if(y.rows()< this->min_split_sample){
        //pred = gradient_descent(pred, y.array()  ,indicator,gamma);
        return new Node(pred, n, y_var,w_var);
    }
    if(!any_split){
        //pred = gradient_descent(pred, y.array()  ,indicator,gamma);
        return new Node(pred ,n, y_var, w_var);
    }

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


    dVector indicator_left = indicator(mask_left,1); dVector gamma_left = gamma(mask_left,1);
    dVector indicator_right = indicator(mask_right,1); dVector gamma_right = gamma(mask_right,1);

    // printf("indicator_left %d indicator_right %d \n ", indicator_left.size(), indicator_right.size());
    // printf("gamma_left %d  gamma_right %d \n ", gamma_left.size(), gamma_right.size());
    
    double loss_parent = (y.array() - pred).square().sum();
    //printf("loss_parent %f \n" ,loss_parent);
    // dVector pred_left = dVector::Constant(y_left.size(),0,loss_function->link_function(y_left.array().mean()));
    // dVector pred_right = dVector::Constant(y_right.size(),0,loss_function->link_function(y_right.array().mean()));
    // double loss_left = (y_left.array() - y_left.array().mean()).square().sum();
    // double loss_right = (y_right.array() - y_right.array().mean()).square().sum();
    // printf("score comparison: %f, %f \n", score, (loss_parent - (loss_left+loss_right))/n);
    
    Node* node = new Node(split_value, loss_parent/n, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    
    if(previuos_tree_node !=NULL){//only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        //node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,previuos_tree_node->left_child);
        node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,previuos_tree_node->left_child, indicator_left,gamma_left);
    }else{
        //node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,NULL);
        node->left_child = build_tree( X_left, y_left, g_left,h_left, depth+1,NULL,indicator_left,gamma_left);
    }
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
        node->left_child->parent_expected_max_S=expected_max_S;
    }
    if(previuos_tree_node !=NULL){ //only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
       //node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,previuos_tree_node->right_child) ;
        node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,previuos_tree_node->right_child,indicator_right,gamma_right) ;
    }else{
        //node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,NULL) ;
        node->right_child = build_tree(X_right, y_right,g_right,h_right, depth+1,NULL,indicator_right,gamma_right) ;
    }
    if(node->right_child!=NULL){
        node->right_child->w_var *=expected_max_S;
        node->right_child->parent_expected_max_S=expected_max_S;
    }

    return node;
}



Node* NewTree::get_root(){
    return this->root;
}

dVector NewTree::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(2,1);
    while(node !=NULL){
        if(node->is_leaf()){
            info(0,1) = node->predict();
            if(std::isnan(node->y_var)||std::isnan(node->w_var) || std::isnan((node->y_var/node->w_var)/node->n_samples) ){
                    std::cout << "y_var or w_var contains NaN:" << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(node->y_var< 0 || node->w_var <0 || (node->y_var/node->w_var)/node->n_samples<0){
                    std::cout << "y_var or w_var <0: " << node->y_var << " " <<node->w_var << " " << node->n_samples<< std::endl;
                }
            if(node->w_var <=0){
                node->w_var =0.00001;
            }
            if(node->y_var <=0){
                node->y_var =0.00001;
            }
            if(_criterion ==1){ //poisson only uses prediction variance
                //info(1,1) = (node->y_var/node->w_var)/node->n_samples;
                info(1,1) = 1/(node->w_var/node->n_samples);
            }
            else{ //mse uses both response and prediction variance
                
                info(1,1) = (node->y_var/node->w_var)/node->n_samples;
            }
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
dMatrix NewTree::predict_info(dMatrix &X){
    int n = X.rows();
    dMatrix leaf_info(n,2);
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


void NewTree::update(dMatrix &X, dVector &y){
    //printf("%d\n", n1);
    dMatrix Xb = sample_X(X,n1);
    dMatrix info = predict_info(Xb);
    dVector weights = info.col(1);//array().min(1000).max(0);
    dVector yb = info.col(0);
    // for (size_t i = 0; i < yb.size(); i++)
    // {
    //     printf("yb = %f \n", yb(i));
    // }

    // complete the squares 
    dVector hb = 2*weights.array();
    dVector gb = -1*hb.array().cwiseProduct(yb.array());
    dVector indicator_b = dVector::Constant(yb.size(),0,  0) ;
    yb = yb.array()+pred_0;

    pred_0 = loss_function->link_function(y.array().mean()+y.array().mean()/2);
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;

    // for (size_t i = 0; i < pred.size(); i++)
    // {
    //     printf("pred = %f \n", pred(i));
    // }

    dVector g = loss_function->dloss(y, pred ); //dVector::Zero(n1,1)
    dVector h = loss_function->ddloss(y, pred ); //dVector::Zero(n1,1)
    dVector gamma = dVector::Constant(y.size(),0,  0) ;
    dVector indicator = dVector::Constant(y.size(),0,  1) ;
    
    // dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1)); 
    // dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1) );
    dMatrix X_concat(X.rows()+Xb.rows(), X.cols());
    dVector y_concat(y.rows()+yb.rows(), 1);
    dVector g_concat(g.rows() + gb.rows(), 1); 
    dVector h_concat(h.rows() + hb.rows(), 1); 
    
    dVector gamma_concat(y.rows() + y.rows(), 1); 
    dVector indicator_concat(y.rows() + y.rows(), 1);
    

    // for (int i = 0; i < yb.size(); i++) {
    //     if (std::isnan(yb(i))) {
    //         std::cout << "yb contains NaN at index " << i << std::endl;
    //     }
    // }
     for (int i = 0; i < weights.size(); i++) {
        if (std::isnan(weights(i)) || weights(i)<=0) {
            std::cout << "weights contains NaN at index " << i <<" - "<< weights(i) << std::endl;
        }
    }


    g_concat <<g,gb ;
    h_concat <<h,hb;
    X_concat <<X,Xb;
    y_concat <<y, loss_function->inverse_link_function(yb.array());
    gamma_concat <<gamma, weights;
    indicator_concat <<indicator, indicator_b;

    // for (size_t i = 0; i < y.size(); i++)
    // {
    //     printf("g = %f \n", y(i));
    // }
    // for (size_t i = 0; i < g.size(); i++)
    // {
    //     printf("g = %f \n", g(i));
    // }
  
    // for (size_t i = 0; i < g_concat.size(); i++)
    // {
    //     printf("g_concat = %f \n", g_concat(i));
    // }
  
    total_obs = X_concat.rows();
    splitter = new Splitter(min_samples_leaf,total_obs, adaptive_complexity, max_features,learning_rate);

    this->root = build_tree(X_concat, y_concat, g_concat, h_concat, 0,this->root,indicator_concat, gamma_concat);//
    n1 = total_obs;
}


dMatrix NewTree::sample_X(const dMatrix &X, int n1){
    std::mt19937 gen(0);
    std::uniform_int_distribution<size_t>  distr(0, X.rows()-1);
    dMatrix X_sample(n1, X.cols());
    for (size_t i = 0; i < n1; i++)
    {   
        size_t ind = distr(gen);
        for (size_t j = 0; j < X.cols(); j++)
        {
            X_sample(i,j) = X(ind,j);
        } 
    }
    bootstrap_seed+=1;
    return X_sample;
}
