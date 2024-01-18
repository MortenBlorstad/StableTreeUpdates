#pragma once
#include "tree.hpp"
#include "naiveupdate.hpp"
#include "treereevaluation.hpp"
#include "stabilityregularization.hpp"
#include "abutree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"
#include <omp.h>
#include <random>

using namespace std;
class RandomForest{
    public:
        explicit RandomForest(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,int method );
        void update(dMatrix &X, dVector &y);
        void learn(dMatrix &X, dVector &y);
        dVector predict(dMatrix &X);
        tuple<dMatrix,dVector> RandomForest::sample_X_y(const dMatrix &X,const dVector &y);
        

    protected:
        int max_depth;
        bool adaptive_complexity;
        int _criterion;
        double min_split_sample;
        int min_samples_leaf;
        double total_obs;
        int max_features;
        int n_estimator;
        double initial_pred;
        std::vector<AbuTree*> forest;
        unsigned int random_state;
        int method;
        double gamma;
        double delta;
        double alpha;
        Tree* create_tree(int method,int random_state_);
        iMatrix sample_indices(int start, int end);
        //iMatrix bootstrap_indices;

};

RandomForest::RandomForest(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, int method ){
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
    this->n_estimator = n_estimator;
    std::vector<AbuTree*> forest;
    thread_local unsigned int random_state = 0;
    this->method = method;
    this->gamma = 0.25;
    this->delta = 0.1;
    this->alpha = 0.05;
}

Tree* RandomForest::create_tree(int method, int random_state_){
    switch (method) {
        case 0:
            return new Tree(this->_criterion,this->max_depth, this->min_split_sample, this->min_samples_leaf, this->adaptive_complexity, this->max_features, 1, 0);
        case 1:
            return new NaiveUpdate(this->_criterion,this->max_depth, this->min_split_sample, this->min_samples_leaf, this->adaptive_complexity, this->max_features, 1, 0);
        case 2:
            return new TreeReevaluation(alpha,delta,this->_criterion,this->max_depth, this->min_split_sample, this->min_samples_leaf, this->adaptive_complexity, this->max_features, 1, 0);
        case 3:
            return new StabilityRegularization(gamma,  _criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, max_features, 1, 0);
        case 4:
            return new AbuTree( _criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, max_features, 1, 0);
        default:
            throw exception("Invalid method");
            break;
    }
}

void RandomForest::learn(dMatrix &X, dVector &y){
    //printf("learn \n");
    random_state = 0;
    //iMatrix bootstrap_indices = sample_indices(0, y.size());
    // for (int b = 0; b < bootstrap_indices.cols(); b++) {
    //     for (int i = 0; i < bootstrap_indices.rows(); i++) {
    //         int val = bootstrap_indices(i,b);
    //         if (isnan((double)val) || isinf((double)val)|| val<0 || val> (y.size()-1) ) {
    //             std::cout << "weights contains NaN at index "<< std::endl;
    //         }
    //     }
    // }
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    int max_threads = omp_get_num_procs();
    this->forest.clear();
   // printf("creating trees \n");
    for (size_t i = 0; i < n_estimator; i++)
    {   
        AbuTree* tree =  new AbuTree( _criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, max_features, 1, 0);//create_tree(method,i);
        this->forest.push_back(tree);     
    }
    //printf("trees created\n");
    #pragma omp parallel for num_threads(max_threads)
    for (size_t i = 0; i < n_estimator; i++)
    {   
        // iVector ind = bootstrap_indices.col(i);
        // dMatrix X_b = X(ind,keep_cols);
        // dVector y_b = y(ind);
        //tie(X_b,y_b) = sample_X_y(X,y);
     
        this->forest[i]->learn(X,y);
        //printf("learned\n");
    }


}

void RandomForest::update(dMatrix &X, dVector &y){
    random_state = 0;
    //iMatrix combined(X.rows(), n_estimator);
    
    //iMatrix bootstrap_indices_new = sample_indices(X.rows()-bootstrap_indices.rows(), X.rows());
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    
    //combined << bootstrap_indices, bootstrap_indices_new;
    //bootstrap_indices = combined;
    //iMatrix bootstrap_indices =  sample_indices(0, y.size());

    // for (int b = 0; b < bootstrap_indices.cols(); b++) {
    //     for (int i = 0; i < bootstrap_indices.rows(); i++) {
    //         int val = bootstrap_indices(i,b);
    //         if (isnan((double)val) || isinf((double)val)) {
    //             std::cout << "weights contains NaN at index "<< std::endl;
    //         }
    //     }
    // }

    int max_threads = omp_get_num_procs();
    #pragma omp parallel for num_threads(max_threads)
    for (size_t i = 0; i < n_estimator; i++)
    {   
        // iVector ind = bootstrap_indices.col(i);
        // dMatrix X_b = X(ind,keep_cols);
        // dVector y_b = y(ind);
        this->forest[i]->update(X,y);
        //printf("updated\n");
    }
}

dVector RandomForest::predict(dMatrix &X){
    
    dVector predictions = dVector::Zero(X.rows(),1);
    int max_threads = omp_get_num_procs();
    #pragma omp parallel for num_threads(max_threads) shared(predictions)
    for (size_t i = 0; i < n_estimator; i++)
    {   
        dVector pred = this->forest[i]->predict(X);
        #pragma omp critical
        predictions = predictions + pred;
    }
    
    return predictions.array()/n_estimator;
}
        
        

tuple<dMatrix,dVector> RandomForest::sample_X_y(const dMatrix &X,const dVector &y){
    printf("random_state %d\n", random_state);
    std::mt19937 gen(random_state);
    size_t n = y.size();
    std::uniform_int_distribution<size_t>  distr(0, n-1);
    dMatrix X_sample(n, X.cols());
    dVector y_sample(n,1);
    printf("n %d\n", n);
    for (size_t i = 0; i < n; i++)
    {
        size_t ind = distr(gen);
        printf("ind %d\n", ind);
        for (size_t j = 0; j < X.cols(); j++){
            X_sample(i,j) = X(ind,j);
        }
        y_sample(i,0) = y(ind,0);
    }   
    
    return tuple<dMatrix,dVector>(X_sample,y_sample);
}


iMatrix RandomForest::sample_indices(int start, int end){
    //printf("start end %d %d \n", start, end);
    std::uniform_int_distribution<int>  distr(start, end-1);
    iMatrix bootstrap_indices_(end-start,this->n_estimator);
    int max_threads = omp_get_num_procs();
    //printf("max_threads %d\n", max_threads);
    #pragma omp parallel for num_threads(max_threads) 
    for (int b = 0; b < n_estimator; b++) {
        std::mt19937 gen(b);
        for (int i = 0; i < end-start; i++) {
            int index = distr(gen);
            bootstrap_indices_(i,b) = index;
        }
    }
    

    
    
    return bootstrap_indices_;
}
