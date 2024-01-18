
#ifndef __GMGTB_HPP_INCLUDED__
#define __GMGTB_HPP_INCLUDED__

// External
#include "agtboost\external_rcpp.hpp"

#include <iostream>
#include <fstream>
#include <functional>
#include <iomanip>
// Internal
#include "optimism\cir.hpp"
#include "optimism\gumbel.hpp"
#include "agtboost\agtnode.hpp"
#include "agtboost\boosttree.hpp"
#include "agtboost\ensemble.hpp"
#include "agtboost\agt_loss_functions.hpp"
#include "agtboost\agt_stable_loss_functions.hpp"
#include "agtboost\agt_initial_prediction.hpp"

using namespace std;
#endif // __GMGTB_HPP_INCLUDED__

// ---------------- ENSEMBLE ----------------
ENSEMBLE::ENSEMBLE(){
    this->first_tree = NULL;
    this->nrounds = 5000;
    this->learning_rate=0.01;
    this->extra_param = 0.0;
    this->loss_function = "mse";
    this->gamma = 0.5;
}

ENSEMBLE::ENSEMBLE(double learning_rate_){
    this->first_tree = NULL;
    this->nrounds = 5000;
    this->learning_rate=learning_rate_;
    this->extra_param = 0.0;
    this->loss_function = "mse";
    this->gamma = 0.5;
}

void ENSEMBLE::set_param(int nrounds_, double learning_rate_, double extra_param_, std::string loss_function_, double gamma)
{
    this->nrounds = nrounds_;
    this->learning_rate = learning_rate_;
    this->extra_param = extra_param_;
    this->loss_function = loss_function_;
    this->gamma = gamma;
}

int ENSEMBLE::get_nrounds(){
    return this->nrounds;
}

double ENSEMBLE::get_learning_rate(){
    return this->learning_rate;
}

double ENSEMBLE::get_extra_param(){
    return this->extra_param;
}

std::string ENSEMBLE::get_loss_function(){
    return this->loss_function;
}

void ENSEMBLE::serialize(ENSEMBLE *eptr, std::ofstream& f)
{
    // If current ENSEMBLE is NULL, return
    if(eptr == NULL)
    {
        //Rcpp::Rcout << "Trying to save NULL pointer" << std::endl;
        return;
    }
    
    f << std::fixed << eptr->nrounds << "\n";
    f << std::fixed << eptr->learning_rate << "\n";
    f << std::fixed << eptr->extra_param << "\n";
    f << std::fixed << eptr->initialPred << "\n";
    f << std::fixed << eptr->initial_score << "\n";
    f << eptr->loss_function << "\n";
    
    eptr->first_tree->serialize(eptr->first_tree, f);
    f.close();
}

void ENSEMBLE::deSerialize(ENSEMBLE *eptr, std::ifstream& f)
{
    // Check stream
    std::streampos oldpos = f.tellg();
    int val;
    int MARKER = -1;
    if( !(f >> val) || val==MARKER ){
        return;   
    }
    f.seekg(oldpos);
    
    // Read from stream
    f >> eptr->nrounds >> eptr->learning_rate >> eptr->extra_param >>
        eptr->initialPred >> eptr->initial_score >> eptr->loss_function >> std::ws;

    eptr->first_tree = new GBTREE;
    eptr->first_tree->deSerialize(eptr->first_tree, f);
}

void ENSEMBLE::save_model(std::string filepath)
{
    std::ofstream f;
    f.open(filepath.c_str());
    this->serialize(this, f);
    f.close();
}

void ENSEMBLE::load_model(std::string filepath)
{
    std::ifstream f;
    f.open(filepath.c_str());
    this->deSerialize(this, f);
    f.close();
}
 
double ENSEMBLE::initial_prediction(Tvec<double> &y, std::string loss_function, Tvec<double> &w){
    
    double pred=0;
    double pred_g_transform = y.sum()/w.sum(); // should be optim given weights...
    
    if(loss_function=="mse"){
        pred = pred_g_transform;
    }else if(loss_function=="logloss"){
        //double pred_g_transform = (y*w).sum()/n; // naive probability
        pred = log(pred_g_transform) - log(1 - pred_g_transform);
    }else if(loss_function=="poisson"){
        //double pred_g_transform = (y*w).sum()/n; // naive intensity
        pred = log(pred_g_transform);
    }else if(loss_function=="gamma::neginv"){
        //double pred_g_transform = (y*w).sum()/n;
        pred = - 1.0 / pred_g_transform;
    }else if(loss_function=="gamma::log"){
        pred = log(pred_g_transform);
    }else if(loss_function=="negbinom"){
        pred = log(pred_g_transform);
    }
    
    return pred;
}


void verbose_output(int verbose, int iteration, int nleaves, double tr_loss, double gen_loss){
    // Print output-information to user
    if(verbose>0){
        if(iteration % verbose == 0){
            std::cout  <<
                std::setprecision(4) <<
                    "it: " << iteration << 
                        "  |  n-leaves: " << nleaves << 
                            "  |  tr loss: " << tr_loss <<
                                "  |  gen loss: " << gen_loss << 
                                    std::endl;
        }
    }
}


// Loss functions defined in Ensemble class
double ENSEMBLE::loss(Tvec<double> &y, Tvec<double> &pred, Tvec<double> &w){
    return loss_functions::loss(y, pred, loss_function, w, extra_param);
}


Tvec<double> ENSEMBLE::dloss(Tvec<double> &y, Tvec<double> &pred){
    return loss_functions::dloss(y, pred, loss_function, extra_param);
}


Tvec<double> ENSEMBLE::ddloss(Tvec<double> &y, Tvec<double> &pred){
    return loss_functions::ddloss(y, pred, loss_function, extra_param);
}


double ENSEMBLE::link_function(double pred_observed){
    return loss_functions::link_function(pred_observed, loss_function);
}


double ENSEMBLE::inverse_link_function(double pred){
    return loss_functions::inverse_link_function(pred, loss_function);
}


//stable loss
double ENSEMBLE::stableloss(Tvec<double> &y, Tvec<double> &pred, Tvec<double> &w, Tvec<double> &prev_pred, double gamma){
    return stable_loss_functions::loss(y, pred, prev_pred, gamma, loss_function, w, extra_param);
}

Tvec<double> ENSEMBLE::dstableloss(Tvec<double> &y, Tvec<double> &pred, Tvec<double> &prev_pred, double gamma){
    return stable_loss_functions::dloss(y, pred, prev_pred, gamma, loss_function, extra_param);
}


Tvec<double> ENSEMBLE::ddstableloss(Tvec<double> &y, Tvec<double> &pred, Tvec<double> &prev_pred, double gamma){
    return stable_loss_functions::ddloss(y, pred, prev_pred, gamma, loss_function, extra_param);
}


void ENSEMBLE::train(
        Tvec<double> &y, 
        Tmat<double> &X, 
        int verbose, 
        bool greedy_complexities, 
        bool force_continued_learning, // Default: False
        Tvec<double> &w, Tvec<double> &offset // Defaults to a zero-vector
    ){
    set_seed(1);
    using namespace std::placeholders;

    // Set initials and declare variables
    int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double ensemble_training_loss;
    double ensemble_approx_training_loss;
    double ensemble_optimism;
    Tvec<double> pred(n), g(n), h(n);
    Tmat<double> cir_sim = cir_sim_mat(100, 100,1); // nsim=100, nobs=100
    
    // Initial constant prediction: arg min l(y,constant)
    this->initialPred = learn_initial_prediction(
        y, 
        offset, 
        std::bind(&ENSEMBLE::dloss, this, _1, _2),
        std::bind(&ENSEMBLE::ddloss, this, _1, _2),
        std::bind(&ENSEMBLE::link_function, this, _1),
        std::bind(&ENSEMBLE::inverse_link_function, this, _1),
        verbose
        );
    pred.setConstant(this->initialPred);
    pred += offset;
    this->initial_score = loss_functions::loss(y, pred, loss_function, w, extra_param);
    
    // First tree
    g = dloss(y, pred) * w;
    h = ddloss(y, pred) * w;
    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = tree_expected_test_reduction(current_tree, learning_rate);
    verbose_output(
        verbose,
        1,
        current_tree->getNumLeaves(),
        loss(y, pred, w),
        this->estimate_generalization_loss(1)
    );
    
    // Consecutive trees
    for(int i=2; i<(MAXITER+1); i++){
        // Calculate gradients
        g = dloss(y, pred) * w;
        h = ddloss(y, pred) * w;
        // Check for perfect fit
        if(((g.array())/h.array()).matrix().maxCoeff() < 1e-12){
            // Every perfect step is below tresh
            break;
        }
        // Train a new tree
        GBTREE* new_tree = new GBTREE();
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate);
        // Update ensemble-predictions
        pred = pred + learning_rate * (new_tree->predict_data(X));
        // Calculate expected generalization loss for tree
        expected_loss = tree_expected_test_reduction(new_tree, learning_rate);
        // Update ensemble training loss and ensemble optimism for iteration k-1
        ensemble_training_loss = loss_functions::loss(y, pred, loss_function, w, extra_param);
        ensemble_approx_training_loss = this->estimate_training_loss(i-1) + 
            new_tree->getTreeScore() * (-2)*learning_rate*(learning_rate/2 - 1);
        ensemble_optimism = this->estimate_optimism(i-1) + 
            learning_rate * new_tree->getTreeOptimism();
        // Optionally output information to user
        verbose_output(
            verbose,
            i,
            new_tree->getNumLeaves(),
            ensemble_training_loss,
            ensemble_training_loss + ensemble_optimism + expected_loss
        );
        // Stopping criteria
        if(!force_continued_learning){
            // Check criterion
            if(expected_loss > EPS){
                break;
            }
            
        }
        // Passed criterion or force passed: Update ensemble
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
        // Check for non-linearity
        if(std::abs(ensemble_training_loss-ensemble_approx_training_loss)>1E-5){
            std::cerr << "Warning: Loss-function deviating from gradient boosting approximation. Try smaller learning_rate."<< std::endl;
        }
    }
}

void ENSEMBLE::update(
         Tvec<double> &y,
        Tvec<double> &prev_pred,  
        Tmat<double> &X, 
        int verbose, 
        bool greedy_complexities,
        bool force_continued_learning, 
        Tvec<double> &w, 
        Tvec<double> &offset
    ){
    set_seed(1);
    using namespace std::placeholders;
    // Set initials and declare variables
    int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double ensemble_training_loss;
    double ensemble_approx_training_loss;
    double ensemble_optimism;
    Tvec<double> pred(n), g(n), h(n);
    Tmat<double> cir_sim = cir_sim_mat(100, 100,1); // nsim=100, nobs=100
    
    // Initial constant prediction: arg min l(y,constant)
    this->initialPred = learn_initial_prediction(
        y, 
        offset, 
        std::bind(&ENSEMBLE::dloss, this, _1, _2),
        std::bind(&ENSEMBLE::ddloss, this, _1, _2),
        std::bind(&ENSEMBLE::link_function, this, _1),
        std::bind(&ENSEMBLE::inverse_link_function, this, _1),
        verbose
        );
    pred.setConstant(this->initialPred);
    pred += offset;
    this->initial_score = stable_loss_functions::loss(y, pred,prev_pred,this->gamma, loss_function, w, extra_param);
    
    // First tree
    g = dstableloss(y, pred, prev_pred, this->gamma) * w;
    h = ddstableloss(y, pred, prev_pred, this->gamma) * w;
    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = tree_expected_test_reduction(current_tree, learning_rate);
    verbose_output(
        verbose,
        1,
        current_tree->getNumLeaves(),
        stableloss(y, pred, w,prev_pred, this->gamma),
        this->estimate_generalization_loss(1)
    );
    
    // Consecutive trees
    for(int i=2; i<(MAXITER+1); i++){
        // Calculate gradients 
        g = dstableloss(y, pred, prev_pred, this->gamma) * w;
        h = ddstableloss(y, pred, prev_pred, this->gamma) * w;
        // Check for perfect fit
        if(((g.array())/h.array()).matrix().maxCoeff() < 1e-12){
            // Every perfect step is below tresh
            break;
        }
        // Train a new tree
        GBTREE* new_tree = new GBTREE();
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate);
        // Update ensemble-predictions
        pred = pred + learning_rate * (new_tree->predict_data(X));
        // Calculate expected generalization loss for tree
        expected_loss = tree_expected_test_reduction(new_tree, learning_rate);
        // Update ensemble training loss and ensemble optimism for iteration k-1
        ensemble_training_loss = loss_functions::loss(y, pred, loss_function, w, extra_param);
        ensemble_approx_training_loss = this->estimate_training_loss(i-1) + 
            new_tree->getTreeScore() * (-2)*learning_rate*(learning_rate/2 - 1);
        ensemble_optimism = this->estimate_optimism(i-1) + 
            learning_rate * new_tree->getTreeOptimism();
        // Optionally output information to user
        verbose_output(
            verbose,
            i,
            new_tree->getNumLeaves(),
            ensemble_training_loss,
            ensemble_training_loss + ensemble_optimism + expected_loss
        );
        // Stopping criteria
        if(!force_continued_learning){
            // Check criterion
            if(expected_loss > EPS){
                break;
            }
            
        }
        // Passed criterion or force passed: Update ensemble
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
        // Check for non-linearity
        // if(std::abs(ensemble_training_loss-ensemble_approx_training_loss)>1E-5){
        //     std::cerr << "Warning: Loss-function deviating from gradient boosting approximation. Try smaller learning_rate."<< std::endl;
        // }
    }
}

void ENSEMBLE::train_from_preds(Tvec<double> &pred, Tvec<double> &y, Tmat<double> &X, int verbose, bool greedy_complexities, Tvec<double> &w){
    // Set init -- mean
    int MAXITER = nrounds;
    int n = y.size(); 
    double EPS = 1E-9;
    double expected_loss;
    double learning_rate_set = this->learning_rate;
    Tvec<double> g(n), h(n);
    
    // Initial prediction
    g = loss_functions::dloss(y, pred, loss_function, extra_param)*w;
    h = loss_functions::ddloss(y, pred, loss_function, extra_param)*w;
    this->initialPred = - g.sum() / h.sum();
    pred = pred.array() + this->initialPred;
    this->initial_score = loss_functions::loss(y, pred, loss_function, w, extra_param); //(y - pred).squaredNorm() / n;
    
    // Prepare cir matrix
    Tmat<double> cir_sim = cir_sim_mat(100, 100,1);
    
    // First tree
    g = loss_functions::dloss(y, pred, loss_function, extra_param)*w;
    h = loss_functions::ddloss(y, pred, loss_function, extra_param)*w;
    this->first_tree = new GBTREE;
    this->first_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
    GBTREE* current_tree = this->first_tree;
    pred = pred + learning_rate * (current_tree->predict_data(X)); // POSSIBLY SCALED
    expected_loss = (current_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
        learning_rate_set * current_tree->getTreeOptimism();
    
    if(verbose>0){
        std::cout  <<
            std::setprecision(4) <<
                "it: " << 1 << 
                    "  |  n-leaves: " << current_tree->getNumLeaves() <<
                        "  |  tr loss: " << loss_functions::loss(y, pred, loss_function, w, extra_param) <<
                            "  |  gen loss: " << this->estimate_generalization_loss(1) << 
                                std::endl;
    }
    
    
    
    for(int i=2; i<(MAXITER+1); i++){
        
    
        // TRAINING
        GBTREE* new_tree = new GBTREE();
        g = loss_functions::dloss(y, pred, loss_function, extra_param)*w;
        h = loss_functions::ddloss(y, pred, loss_function, extra_param)*w;
        new_tree->train(g, h, X, cir_sim, greedy_complexities, learning_rate_set);
        
        // EXPECTED LOSS
        expected_loss = (new_tree->getTreeScore()) * (-2)*learning_rate_set*(learning_rate_set/2 - 1) + 
            learning_rate_set * new_tree->getTreeOptimism();
        
        // Update preds -- if should not be updated for last iter, it does not matter much computationally
        pred = pred + learning_rate * (current_tree->predict_data(X));
        
        // iter: i | num leaves: T | iter train loss: itl | iter generalization loss: igl | mod train loss: mtl | mod gen loss: mgl "\n"
        if(verbose>0){
            if(i % verbose == 0){
                std::cout  <<
                    std::setprecision(4) <<
                        "it: " << i << 
                            "  |  n-leaves: " << current_tree->getNumLeaves() << 
                                "  |  tr loss: " << loss_functions::loss(y, pred, loss_function, w, extra_param) <<
                                    "  |  gen loss: " << this->estimate_generalization_loss(i-1) + expected_loss << 
                                        std::endl;
                
            }
        }
        
        
        if(expected_loss < EPS){ // && NUM_BINTREE_CONSECUTIVE < MAX_NUM_BINTREE_CONSECUTIVE){
            current_tree->next_tree = new_tree;
            current_tree = new_tree;
        }else{
            break;
        }
    }
}

Tvec<double> ENSEMBLE::importance(int ncols)
{
    // Vector with importance
    Tvec<double> importance_vector(ncols);
    importance_vector.setZero();
    
    // Go through each tree to fill importance vector
    GBTREE* current = this->first_tree;
    while(current != NULL)
    {
        current->importance(importance_vector, this->learning_rate);
        current = current->next_tree;
    }
    
    // Scale and return percentwise
    Tvec<double> importance_vec_percent = importance_vector.array()/importance_vector.sum();
    
    return importance_vec_percent;
}

Tvec<double> ENSEMBLE::predict(Tmat<double> &X, Tvec<double> &offset){
    int n = X.rows();
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    pred += offset;
    GBTREE* current = this->first_tree;
    while(current != NULL){
        pred = pred + (this->learning_rate) * (current->predict_data(X));
        current = current->next_tree;
    }
    return pred;
}

Tvec<double> ENSEMBLE::predict2(Tmat<double> &X, int num_trees){
    int n = X.rows();
    int tree_num = 1;
    
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    GBTREE* current = this->first_tree;
    
    
    if(num_trees < 1){
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            pred = pred + (this->learning_rate) * (current->predict_data(X));
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    
    return pred;
}


double ENSEMBLE::estimate_optimism(int num_trees){
    // Return optimism approximated from 2'nd order GB loss-approximation
    // And assuming no-influence / influence adjustment
    double optimism = 0.0;
    int tree_num = 1;
    GBTREE* current = this->first_tree;
    if(num_trees<1){
        while(current != NULL){
            optimism += current->getTreeOptimism();
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            optimism += current->getTreeOptimism();
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    optimism = learning_rate * optimism;
    return optimism;
    
}


double ENSEMBLE::estimate_training_loss(int num_trees){
    // Return training loss approximated from 2'nd order GB loss-approximation
    double training_loss = 0.0;
    int tree_num = 1;
    double total_observed_reduction = 0.0;
    GBTREE* current = this->first_tree;
    if(num_trees<1){
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    training_loss = 
        this->initial_score + 
        total_observed_reduction * 
        (-2)*learning_rate*(learning_rate/2 - 1);
    return training_loss;
}


double ENSEMBLE::estimate_generalization_loss(int num_trees){
    
    int tree_num = 1;
    double total_observed_reduction = 0.0;
    double total_optimism = 0.0;
    GBTREE* current = this->first_tree;
    if(num_trees<1){
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            total_optimism += current->getTreeOptimism();
            current = current->next_tree;
        }
    }else{
        while(current != NULL){
            total_observed_reduction += current->getTreeScore();
            total_optimism += current->getTreeOptimism();
            current = current->next_tree;
            tree_num++;
            if(tree_num > num_trees) break;
        }
    }
    return (this->initial_score) + total_observed_reduction * (-2)*learning_rate*(learning_rate/2 - 1) + 
        learning_rate * total_optimism;
}

int ENSEMBLE::get_num_trees(){
    int num_trees = 0;
    GBTREE* current = this->first_tree;
    
    while(current != NULL){
        num_trees++;
        current = current->next_tree;
    }
    
    return num_trees;
}

Tvec<double> ENSEMBLE::get_num_leaves(){
    int num_trees = this->get_num_trees();
    Tvec<double> num_leaves(num_trees);
    GBTREE* current = this->first_tree;
    for(int i=0; i<num_trees; i++){
        num_leaves[i] = current->getNumLeaves();
        current = current->next_tree;
    }
    return num_leaves;
}

Tvec<double> ENSEMBLE::convergence(Tvec<double> &y, Tmat<double> &X){
    
    // Number of trees
    int K = this->get_num_trees();
    Tvec<double> loss_val(K+1);
    loss_val.setZero();
    
    // Prepare prediction vector
    int n = X.rows();
    Tvec<double> pred(n);
    pred.setConstant(this->initialPred);
    
    // Unit weights
    Tvec<double> w(n);
    w.setOnes();
    
    // After each update (tree), compute loss
    loss_val[0] = loss_functions::loss(y, pred, this->loss_function, w, extra_param);
    
    GBTREE* current = this->first_tree;
    for(int k=1; k<(K+1); k++)
    {
        // Update predictions with k'th tree
        pred = pred + (this->learning_rate) * (current->predict_data(X));
        
        // Compute loss
        loss_val[k] = loss_functions::loss(y, pred, this->loss_function, w, extra_param);
        
        // Update to next tree
        current = current->next_tree;
        
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
    }
    
    return loss_val;
}

Tvec<int> ENSEMBLE::get_tree_depths(){
    // Return vector of ints with individual tree-depths
    int number_of_trees = this->get_num_trees();
    Tvec<int> tree_depths(number_of_trees);
    GBTREE* current = this->first_tree;
    
    for(int i=0; i<number_of_trees; i++){
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
        // get tree depth
        tree_depths[i] = current->get_tree_depth();
        // Update to next tree
        current = current->next_tree;
    }
    return tree_depths;
}

double ENSEMBLE::get_max_node_optimism(){
    // Return the minimum loss-reduction in ensemble
    double max_node_optimism = 0.0;
    double tree_max_node_optimism;
    int number_of_trees = this->get_num_trees();
    GBTREE* current = this->first_tree;
    for(int i=0; i<number_of_trees; i++){
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
        // get minimum loss reduction in tree
        tree_max_node_optimism = current->get_tree_max_optimism();
        if(tree_max_node_optimism > max_node_optimism){
            max_node_optimism = tree_max_node_optimism;
        }
        // Update to next tree
        current = current->next_tree;
    }
    return max_node_optimism;
}

double ENSEMBLE::get_min_hessian_weights(){
    double min_hess_weight = std::numeric_limits<double>::infinity();
    double tree_min_hess_weight;
    int number_of_trees = this->get_num_trees();
    GBTREE* current = this->first_tree;
    for(int i=0; i<number_of_trees; i++){
        // Check if NULL ptr
        if(current == NULL)
        {
            break;
        }
        // get minimum loss reduction in tree
        tree_min_hess_weight = current->get_tree_min_hess_sum();
        if(tree_min_hess_weight < min_hess_weight){
            min_hess_weight = tree_min_hess_weight;
        }
        // Update to next tree
        current = current->next_tree;
    }
    return min_hess_weight;
}






