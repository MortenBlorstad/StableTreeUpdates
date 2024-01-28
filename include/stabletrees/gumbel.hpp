/* gumbel.hpp
*  From https://github.com/Blunde1/agtboost/blob/master/R-package/inst/include/gumbel.hpp
*/
#ifndef __GUMBEL_HPP_INCLUDED__
#define __GUMBEL_HPP_INCLUDED__

#include <Eigen/Dense>

#define _USE_MATH_DEFINES
 
#include <cmath>


using Eigen::Dynamic;

using dVector = Eigen::Matrix<double,Dynamic,1>;
using dArray = Eigen::Array<double,Eigen::Dynamic,1>;
using dMatrix = Eigen::Matrix<double,Dynamic,Dynamic>;

const double PI = 3.14159265358979323846;

// Distribution function templated
template<class T>
T pgumbel(double q, T location, T scale, bool lower_tail, bool log_p){
    
    T z = (q-location)/scale;
    T log_px = -exp(-z); // log p(X <= x)
    T res;
    
    if(lower_tail && log_p){
        res = log_px;
    }else if(lower_tail && !log_p){
        res = exp(log_px);
    }else if(!lower_tail && log_p){
        res = log(1.0 - exp(log_px));
    }else{
        res = 1.0 - exp(log_px);
    }
    
    if( std::isnan(res) ){
        return 1.0;
    }else{
        return res;
    }
    
}

// Gradient of estimating equation for scale
double grad_scale_est_obj(double scale, dArray &x){
    
    int n = x.size();
    dArray exp_x_beta = (-1.0*x.array()/scale).exp();
    double f = scale + (x*exp_x_beta).sum()/exp_x_beta.sum() - x.sum()/n;
    double grad = 2.0*f* ( 1.0 + 
                         ( (x*x*exp_x_beta).sum() * exp_x_beta.sum() - 
                         pow((x*exp_x_beta).sum(),2.0) ) / 
                         pow(scale*exp_x_beta.sum(), 2.0));
    return grad;
    
}

// ML Estimate of scale
double scale_estimate(dArray &x){
    
    // Start in variance estimate -- already pretty good
    int n = x.size();
    int mean = x.sum()/n;
    double var = 0.0;
    for(int i=0; i<n; i++){
        var += (x[i]-mean)*(x[i]-mean)/n;
    }
    double scale_est = sqrt(var*6.0)/PI;
    
    // do some gradient iterations to obtain ML estimate
    int NITER = 50; // max iterations
    double EPS = 1e-2; // precision
    double step_length = 0.2; //conservative
    double step;
    for(int i=0; i<NITER; i++){
        
        // gradient descent
        step = - step_length * grad_scale_est_obj(scale_est, x);
        scale_est += step;
        
        //Rcpp::Rcout << "iter " << i << ", step: " << std::abs(step) << ", estimate: " <<  scale_est << std::endl;
        
        // check precision
        if(std::abs(step) <= EPS){
            break;
        }
        
    }
    
    return scale_est;
    
}

// ML Estimates
dVector par_gumbel_estimates(dArray &x){
    
    int n = x.size();
    
    double scale_est = scale_estimate(x);
    double location_est = scale_est * ( log((double)n) - log( (-1.0*x/scale_est).exp().sum() ) );
    
    dVector res(2);
    res << location_est, scale_est;
    
    return res;
    
}

#endif