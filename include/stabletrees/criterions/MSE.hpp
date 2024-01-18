#pragma once
#ifndef __MSE_HPP_INCLUDED__

#define __MSE_HPP_INCLUDED__



#include <Eigen/Dense>
#include "criterion.hpp"

using namespace std;


class MSE : public Criterion{ 
    public:
        void init(double _n, const dVector &y);
        void update(double y_i);
        void reset();
        double node_impurity(const dVector &y);
        double get_score();
        double lamda; //only used in Regulazation criterions
        ~MSE();
    protected:
        double y_sum_squared;
        double optimism_;
};

MSE::~MSE(){
    Criterion::~Criterion();
    y_sum_squared = NULL;
    lamda = NULL;
}

double MSE::get_score(){
    return score;
}

void MSE::init(double _n, const dVector &y){
    
    Criterion::init(_n,y);
    y_sum_squared = y.array().square().sum();
    score = 0;
    node_score = (y_sum_squared - n*(pred*pred))/n;
}

double MSE::node_impurity(const dVector &y){
    if(node_score >=0.0){
        return node_score;
    }
    double pred = y.array().mean();
    node_score = (y.array() - pred).square().mean();
    return node_score;
}


    
void MSE::update(double y_i){
    
    Criterion::update(y_i);
    //printf("MSE::update called \n");
    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;
    double SSE_L = n_l*( pow(y_bar_l,2.0) );
    double SSE_R = n_r*( pow(y_bar_r,2.0) );
    score = (y_sum_squared - SSE_L-SSE_R)/n;
    // std::cout << "y_sum_squared: " <<  y_sum_squared<< std::endl;
    // std::cout << "SSE_L: " <<  SSE_L << std::endl;
    // std::cout << "SSE_R:  " <<  SSE_R  <<std::endl;
    // std::cout << "n:  " <<  n << std::endl;
    // std::cout << "nl:  " <<  n_l << std::endl;
    // std::cout << "nr:  " <<  n_r << std::endl;
    // std::cout <<"\n" << std::endl;
}

void MSE::reset(){
    Criterion::reset();
}


class MSEABU : public MSE{ 
     public:
        MSEABU(int min_sample_leaf);
        void init(double _n, const dVector &y, const dVector &weights);
        void update(double y_i, double w_i);
        
        bool should_skip(int min_sample_leaf);
        void reset();
        ~MSEABU();
    protected:
        double y1_sum_squared;
        double y1_sum_squared_l;
        double y1_sum_squared_r;
        double n1;
        double n1_l;
        double n1_r;
        double sum_y1_l;
        double sum_y1_r;
        double sum_y1;
        double y1_var_l;
        double y1_var_r;
        double y1_var;
        double w_var_l;
        double w_var_r;
        double w_var;
        double gamma;
        double gamma_r;
        double gamma_l;
        int j;
        
        int min_sample_leaf;
        double sum_wxy;
        double sum_wxy_l;
        double sum_wxy_r;
        
        double sum_w;
        double sum_w_l;
        double sum_w_r;
};
bool MSEABU::should_skip(int min_sample_leaf){
    return MSE::should_skip(min_sample_leaf);
}
MSEABU::MSEABU(int min_sample_leaf):MSE(){
    MSE();
    this->min_sample_leaf = min_sample_leaf;
}

MSEABU::~MSEABU(){
    MSE::~MSE();
    y1_sum_squared = NULL;
    y1_sum_squared_l = NULL;
    y1_sum_squared_r = NULL;
    
    n1 = NULL;
    n1_l = NULL;
    n1_r = NULL;

    sum_y1 = NULL;
    sum_y1_l = NULL;
    sum_y1_r = NULL;

    gamma = NULL;
    gamma_r = NULL;
    gamma_l = NULL;
    // y1_var = NULL;
    // y1_var_r = NULL;
    // y1_var_l = NULL;

    // w_var = NULL;
    // w_var_l = NULL;
    // w_var_l = NULL;

    j = NULL;
}
void MSEABU::init(double _n, const dVector &y, const dVector &weights){
    n = _n;
    sum_y = y.array().sum();
    sum_y_l = 0;
    sum_y_r = sum_y;
    sum_wxy = (y.array()*weights.array()).sum();
    sum_wxy_l = 0;
    sum_wxy_r = sum_wxy;
    n_l = 0;
    n_r = n;
    pred = y.array().mean();
    observed_reduction = 0.0;


    sum_w = weights.array().sum();
    sum_w_l = 0;
    sum_w_r = sum_w;


    y_sum_squared = (y.array().square()*weights.array()).sum();
    score = 0;
    node_score = (y.array()- pred).square().mean();

}

void MSEABU::reset(){
    sum_wxy_l = 0;
    sum_wxy_r = sum_wxy;
    sum_w_l = 0;
    sum_w_r = sum_w;
    sum_y_l = 0;
    sum_y_r = sum_y;
    n_l = 0;
    n_r = n;
}

void MSEABU::update(double y_i, double w_i){
    sum_y_l+= y_i;
    sum_y_r-= y_i;
    sum_wxy_l+= y_i*w_i;
    sum_wxy_r-=y_i*w_i;
    sum_w_l += w_i;
    sum_w_r -= w_i;

    n_l+=1;
    n_r-=1;
    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;

    double SSE_L =( pow(y_bar_l,2.0) )*sum_w_l;
    double SSE_R =( pow(y_bar_r,2.0) )*sum_w_r;

    score = (y_sum_squared -2*sum_wxy_l*y_bar_l -2*sum_wxy_r*y_bar_r  + SSE_L + SSE_R)/n;
    // std::cout << "score: " <<  score<< std::endl;
    // std::cout << "y_sum_squared: " <<  y_sum_squared<< std::endl;
    // std::cout << "sum_y:  " <<  sum_y << std::endl;
    // std::cout << "sum_y_l:  " <<  sum_y_l << std::endl;
    // std::cout << "sum_y_r:  " <<  sum_y_r << std::endl;
    // std::cout << "sum_w:  " <<  sum_w << std::endl;
    // std::cout << "sum_w_l:  " <<  sum_w_l << std::endl;
    // std::cout << "sum_w_r:  " <<  sum_w_r << std::endl;
    // std::cout << "sum_wxy:  " <<  sum_wxy << std::endl;
    // std::cout << "sum_wxy_l:  " <<  sum_wxy_l << std::endl;
    // std::cout << "sum_wxy_r:  " <<  sum_wxy_r << std::endl;
    // std::cout << "SSE_L: " <<  SSE_L << std::endl;
    // std::cout << "SSE_R:  " <<  SSE_R  <<std::endl;
    // std::cout << "2*sum_wxy_l*y_bar_l:  " <<  2*sum_wxy_l*y_bar_l << std::endl;
    // std::cout << "2*sum_wxy_r*y_bar_r:  " <<  2*sum_wxy_r*y_bar_r << std::endl;
    // std::cout << "n:  " <<  n << std::endl;
    // std::cout << "nl:  " <<  n_l << std::endl;
    // std::cout << "nr:  " <<  n_r << std::endl;
    // std::cout <<"\n" << std::endl;

}


class MSEReg : public MSE{ 
    public:
        void init(double _n, const dVector &y,const dVector &yprev);
        void update(double y_i, double yp_i);
        void reset();
        ~MSEReg();
    protected:
        double yprev_sum_squared;
        double sum_yprev_l;
        double sum_yprev_r;
        double sum_yprev;
        double node_stability;
};

MSEReg::~MSEReg(){
    MSE::~MSE();
    yprev_sum_squared = NULL;
    sum_yprev_l = NULL;
    sum_yprev_r = NULL;
    sum_yprev = NULL;
    node_stability = NULL;
}

void MSEReg::reset(){
    MSE::reset();
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;  
}




void MSEReg::init(double _n, const dVector &y,const dVector &yprev){
    MSE::init(_n,y);
    yprev_sum_squared = yprev.array().square().sum();
    sum_yprev = yprev.array().sum();
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;
    node_stability = (yprev.array() - pred).square().mean();
    //printf("node_score : %f %f \n", node_score, node_score + lambda*node_stability);
    node_score = node_score + lambda*node_stability;

    //printf("G : %f %f \n", G, G + 2*lambda*(pred - yprev.array()).sum());
    G += 2*lambda*(pred - yprev.array()).sum();
    //printf("H : %f %f \n", H, H + 2*lambda*n);
    H += 2*lambda*n;
    
    //printf("G2 : %f %f \n", G2, (2*(pred - y.array()) + 2*lambda*(pred - yprev.array())).square().sum());
    
    G2 = (2*(pred - y.array()) + 2*lambda*(pred - yprev.array()) ).square().sum();
    //printf("H2 : %f %f \n", H2, pow(2 + 2*lambda,2)*n);
    H2 = pow(2 + 2*lambda,2)*n;
    //printf("gxh : %f %f \n", gxh, ( (2*(pred - y.array()) + 2*lambda*(pred - yprev.array()) )*(2+2*lambda)  ).sum());
    gxh = ( (2*(pred - y.array()) + 2*lambda*(pred - yprev.array()) )*(2+2*lambda)  ).sum();
    
    //printf("optimism : %f %f \n", optimism, 2*(G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n));
    optimism = 2*(G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n);
    //printf("%f \n", lambda);
}

void MSEReg::update(double y_i, double yp_i){
    MSE::update(y_i);
    sum_yprev_l+= yp_i;
    sum_yprev_r-=yp_i;

    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;

    G_l += 2*lambda*(y_bar_l  -  yp_i) ; H_l += 2*lambda;
    double G_r = G - G_l; double H_r = H-H_l;

    double reg_ = (n_l*pow(y_bar_l,2.0) + n_r*pow(y_bar_r,2.0) - 2*sum_yprev_l*y_bar_l - 2*sum_yprev_r*y_bar_r + yprev_sum_squared)/n;
    reg = ( (1+ score) /(1+ node_score) )   +   (lambda) * ((1+reg_)/(1+node_stability));
    //printf("%f %f %f \n",lambda, score/node_score,reg/node_stability);
    score = score  +   lambda * reg_;
    
    
}

#endif