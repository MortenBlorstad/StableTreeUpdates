#pragma once
#include "tree.hpp"
#include "utils.hpp"
#include "lossfunctions.hpp"
#include "GBT"


class ABUGTB: public GBT{
    public:
        ABUGTB(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,double learning_rate);
        ABUGTB();
        virtual void update(dMatrix &X, dVector &y);
        virtual void learn(dMatrix &X, dVector &y);
    
};

ABUGTB::ABUGTB(int _criterion,int n_estimator,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,double learning_rate):GBT(_criterion, n_estimator, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, learning_rate){
    GBT(_criterion, n_estimator, max_depth,  min_split_sample, min_samples_leaf,  adaptive_complexity, learning_rate);
    
}
void GBT::update(dMatrix &X, dVector &y){
    LossFunction* loss_function = new LossFunction(_criterion);
    initial_pred = loss_function->link_function(y.array().mean());
    dVector pred = dVector::Zero(y.size());
    pred.setConstant(initial_pred);
    dVector pred1 = this->predict(X);
    dVector g = loss_function->dloss(y, dVector::Zero(X.rows(),1), pred, 0.25); 
    dVector h = loss_function->ddloss(y, dVector::Zero(X.rows(),1), pred, 0.25);

    this->first_tree = new Tree(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX,learning_rate);
    first_tree->learn_difference(X,y,g,h);
    pred = pred + first_tree->learning_rate* first_tree->predict(X);
    Tree* current_tree = this->first_tree;
    for(int i = 0; i<n_estimator;i++){
        g = loss_function->dloss(y, pred, pred1, 0.25); 
        h = loss_function->ddloss(y, pred, pred1, 0.25);
        Tree* new_tree = new Tree(_criterion,max_depth, min_split_sample, min_samples_leaf, adaptive_complexity, INT_MAX, learning_rate);
        new_tree->learn_difference(X,y,g,h);
        pred = pred + new_tree->learning_rate*new_tree->predict(X);
        current_tree->next_tree = new_tree;
        current_tree = new_tree;
    }

    


   
}