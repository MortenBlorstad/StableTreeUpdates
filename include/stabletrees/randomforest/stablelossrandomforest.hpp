
#include "stabilityregularization.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"
#include <omp.h>
#include <random>

using namespace std;
class RandomForestSL{
    public:
        explicit RandomForestSL(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, double gamma);
        void update(const dMatrix X, const dVector y, const dVector weights);
        void learn(const dMatrix X, const dVector y, const dVector weights);
        dVector predict(const dMatrix X);
        iMatrix sample_indices(int start, int end);
        
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
        std::vector<StabilityRegularization> forest;
        unsigned int random_state;
        double gamma;
        iMatrix bootstrap_indices;

};

RandomForestSL::RandomForestSL(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features, double gamma){
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
    this->gamma = gamma;

}


void RandomForestSL::learn(const dMatrix X, const dVector y, const dVector weights){
    forest.resize(n_estimator);
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    bootstrap_indices = sample_indices(0, y.size());
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs)
    for (int i = 0; i < n_estimator; i++) {
        forest[i] = StabilityRegularization(gamma,  _criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, max_features, 1, i);
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


void RandomForestSL::update(const dMatrix X,const dVector y, const dVector weights){
    iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
    //iMatrix bootstrap_indices = sample_indices(0, y.size());
    iMatrix bootstrap_indices_new = sample_indices(X.rows()-bootstrap_indices.rows()-1, X.rows());
    iMatrix combined(X.rows(), n_estimator);
    combined << bootstrap_indices, bootstrap_indices_new;
    bootstrap_indices = combined;
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs)
        for (int i = 0; i < n_estimator; i++) {
            iVector ind = bootstrap_indices.col(i);
            dMatrix X_b = X(ind,keep_cols);
            dVector y_b = y(ind);
            dVector weights_b = weights(ind);
            forest[i].update(X_b,y_b, weights_b);
        }
}

// void RandomForestSL::learn(const dMatrix X, const dVector y, const dVector weights){
//     std::lock_guard<std::mutex> lock(mutex);
//     forest.resize(n_estimator);
//     iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
//     iMatrix bootstrap_indices = sample_indices(0, y.size());
//     int num_procs = omp_get_num_procs();
//     #pragma omp parallel for num_threads(num_procs)
//     for (int i = 0; i < n_estimator; i++) {
//         forest[i] = StabilityRegularization(gamma,  _criterion, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, max_features, 1, i);
//     }
//     #pragma omp parallel for num_threads(num_procs)
//     for (int i = 0; i < n_estimator; ++i) {
//         iVector ind = bootstrap_indices.col(i);
//         dMatrix X_b = X(ind,keep_cols);
//         dVector y_b = y(ind);
//         dVector weights_b = weights(ind);
//         //printf("l %d %d  %f %d %d %d %f %i \n", _criterion, max_depth, min_split_sample , min_samples_leaf,  adaptive_complexity,  max_features,1.0,  i);
//         forest[i].learn(X_b,y_b, weights_b);
//     }
// }


// void RandomForestSL::update(const dMatrix X,const dVector y, const dVector weights){
//     std::lock_guard<std::mutex> lock(mutex);
//     iVector keep_cols = iVector::LinSpaced(X.cols(), 0, X.cols()-1).array();
//     iMatrix bootstrap_indices = sample_indices(0, y.size());
//     int num_procs = omp_get_num_procs();
//     #pragma omp parallel for num_threads(num_procs)
//         for (int i = 0; i < n_estimator; i++) {
//             iVector ind = bootstrap_indices.col(i);
//             dMatrix X_b = X(ind,keep_cols);
//             dVector y_b = y(ind);
//             dVector weights_b = weights(ind);
//             forest[i].update(X_b,y_b, weights_b);
//         }
// }

dVector RandomForestSL::predict(const dMatrix X){
    dVector prediction(X.rows()); 
    prediction.setConstant(0);
    //printf("%d \n",forest.size());
    int num_procs = omp_get_num_procs();
    #pragma omp parallel for num_threads(num_procs)
    for (int i = 0; i < n_estimator; i++) {
        dVector pred = forest[i].predict(X);
        #pragma omp critical
        {
            prediction= prediction.array()+pred.array();
        }

    }
    return prediction.array()/n_estimator;
}


iMatrix RandomForestSL::sample_indices(int start, int end){
    //printf("start end %d %d \n", start, end);
    std::lock_guard<std::mutex> lock(mutex);
    std::uniform_int_distribution<int>  distr(start, end-1);
    iMatrix bootstrap_indices_(end-start,this->n_estimator);
    //printf("max_threads %d\n", max_threads);
    int num_procs = omp_get_num_procs();
    //#pragma omp parallel for num_threads(num_procs)
    for (int b = 0; b < n_estimator; b++) {
        std::mt19937 gen(b);
        for (int i = 0; i < end-start; i++) {
            int index = distr(gen);
            bootstrap_indices_(i,b) = index;
        }
    }
    return bootstrap_indices_;
}

