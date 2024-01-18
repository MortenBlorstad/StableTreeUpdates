#pragma once
#include "tree.hpp"
#include "node.hpp"


class NaiveUpdate: public Tree{
    public:
        NaiveUpdate(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity,int max_features,double learning_rate,unsigned int random_state);
        NaiveUpdate();
        virtual void update(const dMatrix &X, const dVector &y, const dVector &weights);
};

NaiveUpdate::NaiveUpdate():Tree(){
    Tree();
}

NaiveUpdate::NaiveUpdate(int _criterion, int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,double learning_rate,unsigned int random_state):Tree(_criterion, max_depth,  min_split_sample,min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state){
    Tree(_criterion, max_depth,  min_split_sample, min_samples_leaf, adaptive_complexity,max_features,learning_rate,random_state);
}



void NaiveUpdate::update(const dMatrix &X, const dVector &y, const dVector &weights){
    if(root == NULL){
        learn(X,y,weights);
    }
    pred_0 = loss_function->link_function(y.array().mean());
    //pred_0 = 0;
    
    dVector pred = dVector::Constant(y.size(),0,  pred_0) ;
    dVector g = loss_function->dloss(y, pred ).array()*weights.array(); //dVector::Zero(n1,1)
    dVector h = loss_function->ddloss(y, pred ).array()*weights.array();//dVector::Zero(n1,1)
    root = update_tree_info(X, y,g,h, root,0,weights);

    //root = update_tree_info(X, y, root,0);
} 