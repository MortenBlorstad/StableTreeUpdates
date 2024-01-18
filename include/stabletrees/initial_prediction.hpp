#ifndef __INITIAL_PREDICTION_HPP_INCLUDED__
#define __INITIAL_PREDICTION_HPP_INCLUDED__

#include <iostream>

using namespace std;
//#include "agtboost\agt_loss_functions.hpp"

double learn_initial_prediction(
        dVector &y, 
        dVector &offset,
        LossFunction* loss_function
    ){
    // Newton opt settings
    double tolerance = 1E-9;
    double step_length = 0.01;
    double step=0.0;
    int niter = 50; // Max iterations
    // Data specific settings
    double y_average = y.array().mean();
    double initial_prediction = loss_function->link_function(1.5*y_average);
    std::cout << "initial_prediction " << initial_prediction << std::endl;
    dVector pred = offset.array() + initial_prediction;
    //std::cout << "pred " << pred<< std::endl;
    // Iterate until optimal starting point found
    for(int i=0; i<niter; i++){
        // Gradient descent
        step = - step_length * loss_function->dloss(y, pred).sum() / loss_function->ddloss(y, pred).sum();
        //std::cout << "step " << pred<< std::endl;
        initial_prediction += step;
        pred = pred.array() + step;
        // Check precision
        if(std::abs(step) <= tolerance){
            break;
        }
    }
    std::cout << "return initial_prediction " << initial_prediction<< std::endl;
    // Retun optimal starting point
    return initial_prediction;
}


#endif
