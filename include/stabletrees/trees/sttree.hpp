#pragma once
#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"

class STTree: public Tree{
    public:
        STTree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state);
        STTree();
        virtual void update(dMatrix &X, dVector &y, dMatrix &X_u);
        dMatrix predict_info(dMatrix &X);
        tuple<bool,int,double, double,double,double,double,double>  STTree::find_update_split(const dMatrix &X,const dVector &y, const dVector &g,const dVector &h,const dVector &weights);
        Node* update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, Node* previuos_tree_node,const dVector &indicator, const dVector &gamma);
    private:
        dVector predict_info_obs(dVector  &obs);
        dMatrix sample_X(const dMatrix &X, int n1);
        int bootstrap_seed ;
};

STTree::STTree():Tree(){
    Tree(); 
    bootstrap_seed=0;
}

STTree::STTree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, double learning_rate, unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf,adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
    bootstrap_seed=0;
}


dVector STTree::predict_info_obs(dVector  &obs){
    Node* node = this->root;
    dVector info = dVector::Zero(3,1);
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
                info(1,1) = (node->y_var/node->w_var)/node->n_samples; //Based on experimental tries
                //info(1,1) = 1/(node->w_var/node->n_samples); //based on theory
            }
            else{ //mse uses both response and prediction variance
                
                info(1,1) = (node->y_var/node->w_var)/node->n_samples;
            }
            info(2,1) = node->w_var;
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
dMatrix STTree::predict_info(dMatrix &X){
    int n = X.rows();
    dMatrix leaf_info(n,3);
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


void STTree::update(dMatrix &X, dVector &y, dMatrix &X_u){

    dMatrix info = predict_info(X_u);

    dVector weights = info.col(1); //array().min(1000).max(0);
    dVector w_wars = info.col(2);
    std::mt19937 gen(bootstrap_seed);

    dVector yb = info.col(0);
    for (size_t i = 0; i < yb.size(); i++)
    {
        std::normal_distribution<double> distribution(0.0, sqrt(w_wars(i)));
        double noise = distribution(gen);
        yb(i)+= noise;
    }

    // complete the squares 
    dVector hb = 2*weights.array();
    dVector gb = -1*hb.array().cwiseProduct(yb.array());
    dVector indicator_b = dVector::Constant(yb.size(),0,  0) ;
    yb = yb.array()+pred_0;

    pred_0 = loss_function->link_function(y.array().mean());//
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    // for (size_t i = 0; i < pred.size(); i++)
    // {
    //     printf("pred = %f \n", pred(i));
    // }

    dVector g = loss_function->dloss(y, pred ); 
    dVector h = loss_function->ddloss(y, pred );
    dVector gamma = dVector::Constant(y.size(),0,  0) ;
    dVector indicator = dVector::Constant(y.size(),0,  1) ;
    
    // dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1)); 
    // dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1) );
    dMatrix X_concat(X.rows()+X_u.rows(), X.cols());
    dVector y_concat(y.rows()+yb.rows(), 1);
    dVector g_concat(g.rows() + gb.rows(), 1); 
    dVector h_concat(h.rows() + hb.rows(), 1); 

    dVector gamma_concat(y.rows() + yb.rows(), 1); 
    dVector indicator_concat(y.rows() + yb.rows(), 1);
    

     for (int i = 0; i < weights.size(); i++) {
        if (std::isnan(weights(i)) || weights(i)<=0) {
            std::cout << "weights contains NaN at index " << i <<" - "<< weights(i) << std::endl;
        }
    }


    g_concat <<g,gb ;
    h_concat <<h,hb;
    X_concat <<X,X_u;
    y_concat <<y, loss_function->inverse_link_function(yb.array());
    gamma_concat <<gamma, weights;
    indicator_concat <<indicator, indicator_b;
    
  
    total_obs = X_concat.rows();
    splitter = new Splitter(min_samples_leaf,total_obs, adaptive_complexity, max_features,learning_rate);
    this->root = update_tree(X_concat, y_concat, g_concat, h_concat, 0,this->root,indicator_concat, gamma_concat);
    n1 = total_obs;
}


// dMatrix AbuTree::sample_X(const dMatrix &X, int n1){
//     std::mt19937 gen(bootstrap_seed);
//     std::uniform_int_distribution<size_t>  distr(0, X.rows()-1);
//     dMatrix X_sample(n1, X.cols());
//     for (size_t i = 0; i < n1; i++)
//     {   
//         size_t ind = distr(gen);
//         for (size_t j = 0; j < X.cols(); j++)
//         {   
//             double x_b = X(ind,j);
//             X_sample(i,j) = x_b;
//         } 
//     }
//     bootstrap_seed+=1;
//     return X_sample;
// }

Node* STTree::update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, Node* previuos_tree_node,const dVector &indicator, const dVector &gamma){
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
    double n2 = indicator.array().sum();
    double gamma_sum = gamma.array().sum();
    double y_sum = (y.array()*indicator.array()).sum();
    double yprev_sum = (gamma.array()*y.array()*(1-indicator.array())).sum();
    double pred = loss_function->link_function((y_sum+yprev_sum)/(n2+gamma_sum)+eps) - pred_0;
    //double pred = -G/H;
    //printf("-g/h = %f, y.mean() = %f, -G/H = %f \n", pred, y.array().mean(),pred_0-G/H);
    if(std::isnan(pred)|| std::isinf(pred)|| indicator.size()<=0){
        std::cout << "_criterion: " << _criterion << std::endl;
        std::cout << "eps: " << eps << std::endl;
        std::cout << "pred: " << pred << std::endl;
        std::cout << "G: " << G << std::endl;
        std::cout << "H: " << H << std::endl;
        
        std::cout << "n2: " << n2 << std::endl;
        std::cout << "gamma_sum: " << gamma_sum << std::endl;
        std::cout << "y_sum: " << y_sum << std::endl;
        std::cout << "yprev_sum: " << yprev_sum << std::endl;
        std::cout << "y: " << y.array().sum() << std::endl;
        std::cout << "indicator: " << indicator.size() << std::endl;
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
    
    double loss_parent = (y.array() - pred).square().sum();
    
    
    Node* node = new Node(split_value, loss_parent/n, score, split_feature, y.rows() , pred, y_var, w_var,features_indices);
    
    if(previuos_tree_node !=NULL){//only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->left_child = update_tree( X_left, y_left, g_left,h_left, depth+1,previuos_tree_node->left_child, indicator_left,gamma_left);
    }else{
        node->left_child = update_tree( X_left, y_left, g_left,h_left, depth+1,NULL,indicator_left,gamma_left);
    }
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
        node->left_child->parent_expected_max_S=expected_max_S;
    }
    if(previuos_tree_node !=NULL){ //only applivable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->right_child = update_tree(X_right, y_right,g_right,h_right, depth+1,previuos_tree_node->right_child,indicator_right,gamma_right) ;
    }else{
        node->right_child = update_tree(X_right, y_right,g_right,h_right, depth+1,NULL,indicator_right,gamma_right) ;
    }
    if(node->right_child!=NULL){
        node->right_child->w_var *=expected_max_S;
        node->right_child->parent_expected_max_S=expected_max_S;
    }

    return node;
}


