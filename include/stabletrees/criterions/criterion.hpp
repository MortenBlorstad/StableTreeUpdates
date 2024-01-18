#pragma once
#ifndef __CRITERION_HPP_INCLUDED__

#define __CRITERION_HPP_INCLUDED__



#include <Eigen/Dense>

using namespace std;

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;
using dArray = Eigen::Array<double,Eigen::Dynamic,1>;

class Criterion{
    public:
        explicit Criterion();
        virtual void init(double n, const dVector &y,const dVector &yprev);
        virtual void init(double n, const dVector &y);


        virtual void update(double y_i, double yp_i);
        virtual void update(double y_i);

        virtual void reset();

        virtual double node_impurity(const dVector &y);
        virtual bool should_skip(int min_samples_leaf);
        void set_lambda(double lambda);

        double get_score();
        double get_reg();
        double node_score;
        double observed_reduction;
        double score_reg;

        double num_splits = 0;
        double optimism;
        double local_opt_l;
        double local_opt_r;
        double lambda; //only used in Regulazation criterions
        virtual ~Criterion();
        double n_l;
        double n_r;
        double n;
        double G;
        double H;
        
        
    protected:
        double sum_y_l;
        double sum_y_r;
        double sum_y;
        
        double G_l;
        double H_l;
        double G2; double H2; double gxh;
        double pred;
        
        
        
        double reg;
        double score;
        double eps = 0.000000000000001;    

};

Criterion::Criterion(){

}

Criterion::~Criterion(){
    node_score = NULL;
    observed_reduction = NULL;
    num_splits = NULL;
    optimism = NULL;
    lambda = NULL; //only used in Regulazation criterions
    sum_y_l = NULL;
    sum_y_r = NULL;
    sum_y = NULL;
    n = NULL;
    n_l = NULL;
    n_r = NULL;
    G = NULL;  G_l = NULL;
    H = NULL;  H_l = NULL;
    G2 = NULL;  H2 = NULL;  gxh = NULL;
    pred = NULL;
    score = NULL;
    eps = NULL; 
}

void Criterion::set_lambda(double lambda){
    this->lambda = lambda;
}

bool Criterion::should_skip(int min_samples_leaf){
    // if(n_l < (double)min_samples_leaf || n_r < (double)min_samples_leaf){
    //     printf("%f %f %d\n", n_l, n_r,min_samples_leaf);
    // }
    return n_l < (double)min_samples_leaf || n_r < (double)min_samples_leaf;
}

double Criterion::node_impurity(const dVector &y){
    return 0;
}

double Criterion::get_reg(){
    return reg;
}

void Criterion::update(double y_i){
    sum_y_l+= y_i;
    sum_y_r-=y_i;
    n_l+=1;
    n_r-=1;
}

void Criterion::update(double y_i, double yp_i){

}




void Criterion::reset(){
    sum_y_l = 0;
    sum_y_r = sum_y;
    n_r = n;
    n_l = 0;
    G_l = 0; H_l = 0; 
    
}

double Criterion::get_score(){
    return score;
}
void Criterion::init(double _n, const dVector &y){

    n = _n;
    sum_y = y.array().sum();
    sum_y_l = 0;
    sum_y_r = sum_y;
    n_l = 0;
    n_r = n;
    pred = sum_y/n;
    observed_reduction = 0.0;

}

void Criterion::init(double _n, const dVector &y, const dVector &yprev){

}

#endif