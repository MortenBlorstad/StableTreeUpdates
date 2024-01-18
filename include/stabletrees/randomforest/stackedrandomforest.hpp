#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"
#include <omp.h>
#include <random>

using namespace std;
class StackedRandomForest{
    public:
        explicit StackedRandomForest(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, double gamma, double learning_rate);
        virtual void update(const dMatrix X, const dVector y, const dVector weights);
        virtual void learn(const dMatrix X, const dVector y, const dVector weights);
        dVector predict(const dMatrix X);
        iMatrix sample_indices(int start, int end);
        void learn_tree_weights(const dMatrix &X, const dVector &y_true,const dVector &prev_pred,const dVector  &weights );
        

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
        std::vector<Tree> forest;
        std::vector<double> tree_weights;
        unsigned int random_state;
        LossFunction* loss_functions;
        double gamma;
        double learning_rate;

};




void StackedRandomForest::learn_tree_weights(const dMatrix &X, const dVector &y_true,const dVector &prev_pred, const dVector  &weights ){
    
    for (size_t epoch = 0; epoch < 100; epoch++)
    {   
        dVector preds = predict(X);
        double pred_max = y_true.array().maxCoeff();
        double pred_min= y_true.minCoeff();
        dVector preds_norm = (preds.array() -pred_min) / (pred_max - pred_min);
        dVector y_true_norm = (y_true.array() -pred_min) / (pred_max - pred_min);
        dVector prev_pred_norm = (prev_pred.array() -pred_min) / (pred_max - pred_min);
        dVector g_ = loss_functions->dloss(y_true_norm,preds_norm,prev_pred_norm, gamma, weights); // get gradients
        std::vector<double> g(g_.data(), g_.data() + g_.size()); // put gradient in std::vector as dVector is not thread safe
        dVector loss = loss_functions->loss(y_true_norm,preds_norm,prev_pred_norm, gamma);
        printf("loss %f \n", loss.array().mean());
        for (size_t i = 0; i < n_estimator; i++)
        {
            double nabla = 0;
            dVector tree_preds = (forest[i].predict(X).array() - pred_min)/(pred_max - pred_min);
            for (size_t j = 0; j < tree_preds.size(); j++)
            {
                nabla+=tree_preds(j)*g[j];
            }
            nabla/=tree_preds.size();
            //printf("%f\n",nabla);
            tree_weights[i] =tree_weights[i] -learning_rate*(nabla) ;
        }
    } 
    printf("\n");  
}


StackedRandomForest::StackedRandomForest(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, double gamma, double learning_rate){
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
    thread_local unsigned int random_state = 0;
    loss_functions = new LossFunction(_criterion);
    this->gamma = gamma;
    this->learning_rate = learning_rate;
}

void StackedRandomForest::learn(const dMatrix X, const dVector y, const dVector weights){
    forest.resize(n_estimator);
    tree_weights.resize(n_estimator);
    for (size_t i = 0; i < tree_weights.size(); i++)
    {
        tree_weights[i] = 1/(double)n_estimator;
        //printf("%f\n", tree_weights[i]);
    }
    

    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    iMatrix bootstrap_indices = sample_indices(0, y.size());
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs)
    for (int i = 0; i < n_estimator; i++) {
        forest[i]= Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity,  max_features,1.0,  i);
    }

    #pragma omp parallel for num_threads(num_procs)
    for (int i = 0; i < n_estimator; ++i) {
        iVector ind = bootstrap_indices.col(i);
        dMatrix X_b = X(ind,keep_cols);
        dVector y_b = y(ind);
        dVector weights_b = weights(ind);
        //printf("l %d %d  %f %d %d %d %f %i \n", _criterion, max_depth, min_split_sample , min_samples_leaf,  adaptive_complexity,  max_features,1.0,  i);
        forest[i].learn(X_b,y_b, weights_b);
    }
}

void StackedRandomForest::update(const dMatrix X,const dVector y, const dVector weights){
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    iMatrix bootstrap_indices = sample_indices(0, y.size());
    dVector prev_preds = predict(X);
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs)
    for (int i = 0; i < n_estimator; i++) {
        iVector ind = bootstrap_indices.col(i);
        dMatrix X_b = X(ind,keep_cols);
        dVector y_b = y(ind);
        dVector weights_b = weights(ind);
        forest[i].update(X_b,y_b, weights_b);
    }
    learn_tree_weights(X, y, prev_preds,weights );
    // for (size_t i = 0; i < tree_weights.size(); i++)
    // {
    //     printf("%f\n", tree_weights[i]);
    // }
}



dVector StackedRandomForest::predict(const dMatrix X){
    dVector prediction(X.rows()); 
    prediction.setConstant(0);
    //printf("%d \n",forest.size());
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs)
    for (int i = 0; i < n_estimator; i++) {
        dVector pred = forest[i].predict(X).array()*tree_weights[i];
        #pragma omp critical
        {
            prediction= prediction.array()+pred.array();
        }
    }
    return prediction;
}

iMatrix StackedRandomForest::sample_indices(int start, int end){
    //printf("start end %d %d \n", start, end);
    std::uniform_int_distribution<int>  distr(start, end-1);
    iMatrix bootstrap_indices_(end-start,this->n_estimator);
    //printf("max_threads %d\n", max_threads);
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs)
    for (int b = 0; b < n_estimator; b++) {
        std::mt19937 gen(b);
        for (int i = 0; i < end-start; i++) {
            int index = distr(gen);
            bootstrap_indices_(i,b) = index;
        }
    }
    return bootstrap_indices_;
}


