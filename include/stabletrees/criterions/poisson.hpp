#pragma once
#ifndef __POISSON_HPP_INCLUDED__

#define __POISSON_HPP_INCLUDED__



#include <Eigen/Dense>
#include "criterion.hpp"
using namespace std;


class Poisson : public Criterion{ 
    public:
        ~Poisson();
        void init(double _n, const dVector &y);
        void update(double y_i);
        void reset();
        double node_impurity(const dVector &y);
        double get_score();
        bool should_skip(int min_samples_leaf);
    protected:
        double sum_ylogy_l;
        double sum_ylogy_r;
        double sum_ylogy;

};

Poisson::~Poisson(){
    Criterion::~Criterion();
    sum_ylogy_l = NULL;
    sum_ylogy_r = NULL;
    sum_ylogy = NULL;
}


double Poisson::get_score(){
    return score;
}

void Poisson::init(double _n, const dVector &y){
    Criterion::init(_n,y);
    // sum_ylogy_l = 0;
    // sum_ylogy = (y.array()*log(y.array()+ eps)).sum();
    // sum_ylogy_r = sum_ylogy;

    node_score = node_impurity(y);
    score = 0;
}

double Poisson::node_impurity(const dVector &y){
    double pred = y.array().mean();
    return 2*((y.array()+eps)*log((y.array()+eps)/(pred+eps)) - (y.array()-pred)).mean();
}

bool Poisson::should_skip(int min_samples_leaf){
    bool skip = false;
    if(Criterion::should_skip(min_samples_leaf)){
        skip = true;
    }
    if(sum_y_l<=eps || sum_y_r<=eps){
        skip = true;
    }
    
    return skip;
}
    
void Poisson::update(double y_i){
    Criterion::update(y_i);
    


    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;

    double sum_ylogpred_l = sum_y_l*log(y_bar_l);
    double sum_ylogpred_r = sum_y_r*log(y_bar_r);
    score =  (y_bar_l + y_bar_r - sum_ylogpred_l  - sum_ylogpred_r)/n;
    
}

void Poisson::reset(){
    Criterion::reset();
    sum_ylogy_l = 0;
    sum_ylogy_r = sum_ylogy;

}


class PoissonABU : public Poisson{ 
    public:
        void init(double _n, const dVector &y, const dVector &weights);
        void update(double y_i,double w_i);
        void reset();
        ~PoissonABU();

    protected:
        double sum_wxy;
        double sum_wxy_l;
        double sum_wxy_r;
        
        double sum_w;
        double sum_w_l;
        double sum_w_r;
        double y_sum_squared;
        double n1;
        double n2;
        double n2_l;
        double n2_r;
        iVector get_y2_mask(const dVector &y, const dVector &weight);
        double sum_y2;
        double sum_y2_l;
        double sum_y2_r;


};

PoissonABU::~PoissonABU(){
    Poisson::~Poisson();
}

iVector PoissonABU::get_y2_mask(const dVector &y, const dVector &weight){
    std::vector<int> mask;
    for(int i=0; i<y.rows();i++){
        if(weight[i]<=0)
            mask.push_back(i);
    }
    iVector mask_v = Eigen::Map<iVector, Eigen::Unaligned>(mask.data(), mask.size());
    return mask_v;
}

void PoissonABU::init(double _n, const dVector &y, const dVector &weights){
    Poisson::init(_n,y);
    sum_wxy = ((y.array()+eps).log()*weights.array()).sum();
    sum_wxy_l = 0;
    sum_wxy_r = sum_wxy;

    iVector y2_mask = get_y2_mask(y, weights);

    sum_y2 = y(y2_mask).array().sum();
    sum_y2_l = 0;
    sum_y2_r = sum_y2;

    sum_w = weights.array().sum();
    sum_w_l = 0;
    sum_w_r = sum_w;

    n2 = y2_mask.size();
    n2_l = 0;
    n2_r = n2;

    y_sum_squared = ((y.array()+eps).log().square()*weights.array()).sum();
}


void PoissonABU::reset(){
    Poisson::reset();
    sum_wxy_l = 0;
    sum_wxy_r = sum_wxy;
    sum_w_l = 0;
    sum_w_r = sum_w;
    n1 = 0;
    n2 = 0;
    n2_l = 0;
    n2_l = n2;
    sum_y2_l = 0;
    sum_y2_r = sum_y2;

}

void PoissonABU::update(double y_i,double w_i){
    Criterion::update(y_i);
    
    
    if(w_i==0){
        n2_l+=1;
        n2_r-=1;
        sum_y2_l += y_i;
        sum_y2_r -= y_i;
    }
    double y_bar_l = log((sum_y2_l+eps)/n2_l);
    double y_bar_r = log((sum_y2_r+eps)/n2_r);    

    double sum_ylogpred_l = sum_y2_l*y_bar_l;
    double sum_ylogpred_r = sum_y2_r*y_bar_r;
    score =  (sum_y2_l+ sum_y2_r - sum_ylogpred_l  - sum_ylogpred_r);

    if(w_i!=0){
        n1+=1;
        sum_wxy_l+= log(y_i+eps)*w_i;
        sum_wxy_r-=log(y_i+eps)*w_i;
        sum_w_l += w_i;
        sum_w_r -= w_i;
    }
        
    double SSE_L =( pow(y_bar_l,2.0) )*sum_w_l;
    double SSE_R =( pow(y_bar_r,2.0) )*sum_w_r;

    double reg = (y_sum_squared -2*sum_wxy_l*y_bar_l -2*sum_wxy_r*y_bar_r  + SSE_L + SSE_R);
    // std::cout << "ybar_l: " <<  (sum_y2_l+eps)/n2_l<< std::endl;
    // std::cout << "sum_ylogpred_l: " <<  sum_ylogpred_l<< std::endl;
    // std::cout << "n2_l: " <<  n2_l<< std::endl;
    // std::cout << "sum_ylogpred_r: " <<  sum_ylogpred_r<< std::endl;
    // std::cout << "score: " <<  score<< std::endl;
    // std::cout << "reg: " <<  reg<< std::endl;
    // score= (score + reg) /n;
    // std::cout << "score: " <<  score<< std::endl;
    // std::cout << "y_sum_squared: " <<  y_sum_squared<< std::endl;
    //  std::cout << "sum_y2:  " <<  sum_y2 << std::endl;
    // std::cout << "sum_y2_l:  " <<  sum_y2_l << std::endl;
    // std::cout << "sum_y2_r:  " <<  sum_y2_r << std::endl;
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


class PoissonReg : public Poisson{ 
    public:
        void init(double _n, const dVector &y, const dVector &yprev);
        void update(double y_i,double yp_i);
        void reset();
        ~PoissonReg();

    protected:
        double sum_prev_ylogy_l;
        double sum_prev_ylogy_r;
        double sum_prev_ylogy;
        double sum_yprev_l;
        double sum_yprev_r;
        double sum_yprev;

};

PoissonReg::~PoissonReg(){
    Poisson::~Poisson();
    sum_prev_ylogy_l = NULL;
    sum_prev_ylogy_r = NULL;
    sum_prev_ylogy = NULL;
    sum_yprev_l = NULL;
    sum_yprev_r = NULL;
    sum_yprev = NULL;
}
void PoissonReg::init(double _n, const dVector &y, const dVector &yprev){
    Poisson::init(_n,y);
    sum_prev_ylogy = (yprev.array()*log(yprev.array()+ eps)).sum();
    sum_prev_ylogy_l = 0;
    sum_prev_ylogy_r = sum_prev_ylogy;

    sum_yprev = yprev.array().sum();
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;
}


void PoissonReg::reset(){
    Poisson::reset();
    sum_prev_ylogy_l = 0;
    sum_prev_ylogy_r = sum_prev_ylogy;
    sum_yprev_l = 0;
    sum_yprev_r = sum_yprev;
}

void PoissonReg::update(double y_i,double yp_i){
    Poisson::update(y_i);
    sum_yprev_l+=yp_i;
    sum_yprev_r-=yp_i;
    sum_prev_ylogy_l += (yp_i+eps)*log(yp_i+eps);
    sum_prev_ylogy_r -= (yp_i+eps)*log(yp_i+eps);

    double y_bar_l = sum_y_l/n_l;
    double y_bar_r = sum_y_r/n_r;
    double sum_prev_ylogpred_l = sum_yprev_l*log(y_bar_l+eps);
    double sum_prev_ylogpred_r = sum_yprev_r*log(y_bar_r+eps);
   

    double reg = ( (sum_prev_ylogy_l - sum_prev_ylogpred_l - (sum_yprev_l - n_l*y_bar_l) ) + (sum_prev_ylogy_r - sum_prev_ylogpred_r  - (sum_yprev_r - n_r*y_bar_r) ) )/n;
    score+=reg;
}


#endif