#pragma once
#include "tree.hpp"

class StableLossTree: public Tree{
    public:
        StableLossTree(double gamma, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf,bool adaptive_complexity,int max_features,double learning_rate,unsigned int random_state);
        StableLossTree();
        virtual void update(const dMatrix X, const dVector y, const dVector weights);
        Node* update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, const Node* previuos_tree_node, const dVector &ypred1, const dVector &weights);
    private:
        double gamma;
        
};

StableLossTree::StableLossTree():Tree(){
    Tree();
    this->gamma = 0.5;
}

StableLossTree::StableLossTree(double gamma, int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate, unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth, min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
    this->gamma = gamma;
    this->init_random_state = random_state;
}

void StableLossTree::update(const dMatrix X, const dVector y, const dVector weights){
    //printf("sl \n");
    this->random_state = this->init_random_state;
    if(this->root == NULL){
        this->learn(X,y,weights);
    }else{
        std::lock_guard<std::mutex> lock(mutex);
        dVector ypred1 = this->predict(X);
        dVector ypred1_linked = loss_function->link_function(ypred1);

        // dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1), ypred1, lambda); 
        // dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1), ypred1, lambda);
        double original_mean = (y.array()).mean() ;

        // // a quick fix for SL, since for some updates some of the prediction become extremely large (inf). fix by unsuring log lambda is is at least 0.
        // if(_criterion ==1){ // if poisson loss,
        //     original_mean = max(original_mean,exp(1)); // one need to ensure that $\bar{y}$ is sufficiently large,
        //                                             // so that the Poisson distribution can be approximated by a normal distribution with mean $\lambda$ and variance $\lambda$.
        //                                              //  If $\bar{y}$ is small, then the Poisson distribution is better approximated by a gamma distribution.
        // }
        

        pred_0 = loss_function->link_function(original_mean);
        
        dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
        pred+=weights;
        dVector g = loss_function->dloss(y.array(), pred,ypred1, gamma,weights ); //dVector::Zero(n1,1)
        dVector h = loss_function->ddloss(y.array(), pred,ypred1,gamma,weights); //dVector::Zero(n1,1)

        total_obs = y.size();
        splitter->total_obs= total_obs;
        this->root = update_tree(X, y, g, h, 0,this->root, ypred1, weights );
    }     
}



Node* StableLossTree::update_tree(const dMatrix  &X, const dVector &y, const dVector &g, const dVector &h, int depth, const Node* previuos_tree_node, const dVector &ypred1, const dVector &weights){
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
    double ypred1_sum = (ypred1.array()*gamma).sum();
    // if(ypred1_sum !=0){
    //     printf("%f\n",ypred1_sum );
    // }
    double sum_weights = weights.array().sum();

    // if(y_sum+ypred1_sum !=y_sum){
    //     printf("%f\n",ypred1_sum );
    // }
    // if((1+gamma)*sum_weights !=sum_weights){
    //     printf("%f\n",sum_weights );
    // }
    // printf("%f %f %f %f\n",y_sum,ypred1_sum,sum_weights,gamma );
    double pred = loss_function->link_function((y_sum+ypred1_sum)/((sum_weights+gamma*n)) +eps) - pred_0;

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
        node->left_child = update_tree( X_left, y_left, g_left,h_left, depth+1,previuos_tree_node->left_child,ypred1_left,weights_left);
    }else{
        node->left_child = update_tree( X_left, y_left, g_left,h_left, depth+1,NULL,ypred1_left,weights_left);
    }
    if(node->left_child!=NULL){
        node->left_child->w_var*=expected_max_S;
        node->left_child->parent_expected_max_S=expected_max_S;
    }
    if(previuos_tree_node !=NULL){ //only applicable for random forest for remembering previous node sub features when updating a tree (or if max_features are less then total number of features)
        node->right_child = update_tree(X_right, y_right,g_right,h_right, depth+1,previuos_tree_node->right_child,ypred1_right,weights_right) ;
    }else{
        node->right_child = update_tree(X_right, y_right,g_right,h_right, depth+1,NULL,ypred1_right,weights_right) ;
    }
    if(node->right_child!=NULL){
        node->right_child->w_var *=expected_max_S;
        node->right_child->parent_expected_max_S=expected_max_S;
    }

    return node;
}










