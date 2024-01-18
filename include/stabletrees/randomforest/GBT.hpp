#pragma once
#include "GTB_tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"


class GBT{
    public:
        explicit GBT(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,double learning_rate);
        explicit GBT();
        void update(dMatrix &X, dVector &y, double gamma = 0.5);
        void learn(dMatrix &X, dVector &y);
        dVector predict(dMatrix &X);
        GTBTREE* first_tree;
        GTBTREE* current_tree;
        GTBTREE* new_tree;
        LossFunction* loss_function ;
    protected:
        int max_depth;
        bool adaptive_complexity;
        int _criterion;
        double min_split_sample;
        int min_samples_leaf;
        double total_obs;
        double learning_rate;
        int n_estimator;
        double initial_pred;
        unsigned int random_state;
};

GBT::GBT(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,double learning_rate){
    if(max_depth !=NULL){
       this-> max_depth =  max_depth;
    }else{
        this-> max_depth =  INT_MAX;
    }
    this-> min_split_sample = min_split_sample;
    this->min_samples_leaf = min_samples_leaf;
    this->_criterion = _criterion;
    this->adaptive_complexity = adaptive_complexity;
    this->learning_rate = learning_rate;
    this->n_estimator = n_estimator;
    this->random_state = 1;
    loss_function = new LossFunction(_criterion);
}

void GBT::learn(dMatrix &X, dVector &y){
    initial_pred = loss_function->link_function(y.array().mean());
    dVector pred = dVector::Zero(y.size());
    pred.setConstant(initial_pred);

    dVector g = loss_function->dloss(y,  pred);
    dVector h = loss_function->ddloss(y, pred); 

    this->first_tree = new GTBTREE(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX,learning_rate,this->random_state);
    first_tree->learn(X,y,g,h);
    pred = pred + first_tree->learning_rate* loss_function->link_function(first_tree->predict(X));
    GTBTREE* current_tree = this->first_tree;
    for(int i = 0; i<n_estimator;i++){
        g = loss_function->dloss(y,  pred);
        h = loss_function->ddloss(y,  pred);
        GTBTREE* new_tree = new GTBTREE(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX, learning_rate,this->random_state);
        new_tree->learn(X,y,g,h);
        pred = pred + new_tree->learning_rate* loss_function->link_function(new_tree->predict(X));
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
    }


}

dVector GBT::predict(dMatrix &X){
    GTBTREE* current_tree = this->first_tree;
    dVector pred =  dVector::Zero(X.rows());
    pred.setConstant(initial_pred);
    while(current_tree !=NULL){
        pred = pred + learning_rate*loss_function->link_function( current_tree->predict(X));
        current_tree = current_tree->next_tree;

    }
    return loss_function->inverse_link_function(pred);
}

void GBT::update(dMatrix &X, dVector &y, double gamma){
    
    dVector weights = dVector::Constant(y.size(),1 );
    
    
    initial_pred = loss_function->link_function(y.array().mean());
    dVector pred = dVector::Zero(y.size());
    pred.setConstant(initial_pred);
    dVector pred1 = loss_function->link_function(this->predict(X));
    dVector g = loss_function->dloss(y,pred, pred1, gamma,weights); 
    dVector h = loss_function->ddloss(y, pred, pred1, gamma,weights);
    
    this->first_tree = new GTBTREE(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX,learning_rate,this->random_state);
    first_tree->learn(X,y,g,h);
    pred = pred + first_tree->learning_rate* loss_function->link_function(first_tree->predict(X));
    GTBTREE* current_tree = this->first_tree;
    for(int i = 0; i<n_estimator;i++){
        g = loss_function->dloss(y, pred, pred1, gamma,weights); 
        h = loss_function->ddloss(y, pred, pred1, gamma,weights);
        GTBTREE* new_tree = new GTBTREE(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX, learning_rate,this->random_state);
        new_tree->learn(X,y,g,h);
        pred = pred + new_tree->learning_rate*loss_function->link_function(new_tree->predict(X));
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
    }

    


   
}