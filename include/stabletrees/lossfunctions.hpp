
#pragma once
#ifndef __LOSSFUNCTIONS_HPP_INCLUDED__
#define __LOSSFUNCTIONS_HPP_INCLUDED__
#include <Eigen/Dense>
#include <vector>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;

using namespace std;


class LossFunction{

    public:
        LossFunction();
        LossFunction(int _citerion);
        ~LossFunction();
        dVector link_function(dVector y_pred );
        double link_function(double y_pred );
        dVector inverse_link_function(dVector y_pred_transformed );
        double inverse_link_function(double y_pred_transformed );
        dVector loss(dVector y_true,dVector y_pred );
        dVector dloss(dVector y_true,dVector y_pred );
        double dloss(double y_true,double y_pred );
        dVector ddloss(dVector y_true,dVector y_pred );
        double ddloss(double y_true,double y_pred );
        dVector loss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma);
        dVector dloss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma, dVector weights );
        dVector ddloss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma, dVector weights );

        dVector dloss(dVector y_true,dVector y_pred2, dVector y_pred1 , dVector gamma, dVector weights );
        dVector ddloss(dVector y_true,dVector y_pred2, dVector y_pred1 , dVector gamma, dVector weights );
        int citerion;
        
};


LossFunction::LossFunction(){
    this->citerion = 0; //"mse"
}
LossFunction::LossFunction(int citerion){
    this->citerion = citerion; 
}

dVector LossFunction::link_function(dVector y_pred ){
    if(citerion ==1) //poisson
         y_pred = y_pred.array().log();
    if(citerion ==2) //multiclassification/categorical distribution - link = softmax
        return y_pred.array()/y_pred.array().sum();

    return y_pred;

}



double LossFunction::link_function(double y_pred ){
    if(citerion ==1)
         y_pred = log(y_pred);

    return y_pred;

}
dVector LossFunction::inverse_link_function(dVector y_pred_transformed ){
    if(citerion ==1)
        y_pred_transformed = y_pred_transformed.array().exp();

    return y_pred_transformed;

}
double LossFunction::inverse_link_function(double y_pred_transformed ){
    if(citerion ==1)
        y_pred_transformed = exp(y_pred_transformed);
    return y_pred_transformed;
}


dVector LossFunction::loss(dVector y_true, dVector y_pred  ){
    if(citerion ==0)
        return (y_true.array() - y_pred.array()).square();

    if(citerion ==1)
        return (y_pred.array().exp() - y_true.array()*y_pred.array());

    throw exception("asdada");
}

dVector LossFunction::dloss(dVector y_true,dVector y_pred  ){
    if(citerion ==0){
        return 2*(y_pred.array() - y_true.array());
    }
        

    if(citerion ==1){
        return y_pred.array().exp() - y_true.array(); //1- y_true.array(); //
    }
        
    throw exception("asdada");
}

double LossFunction::dloss(double y_true,double y_pred ){
    if(citerion ==0){
        return 2*(y_pred- y_true);
    }
        

    if(citerion ==1){
        return exp(y_pred) - y_true; //1- y_true.array(); //
    }
        
    throw exception("asdada");

}

dVector LossFunction::ddloss(dVector y_true,dVector y_pred  ){
    if(citerion ==0){
        //printf("%d \n ", citerion);
        return dVector::Constant(y_true.size(),0,2.0);
    }

    if(citerion ==1){
       // printf("%d \n ", citerion);
        return y_pred.array().exp();
    }
        

    throw exception("asdada");
}
double LossFunction::ddloss(double y_true,double y_pred  ){
    if(citerion ==0){
        return 2;
    }

    if(citerion ==1){
        return exp(y_pred); //y_pred.array()/ y_true.array().square() ;
    }
        

    throw exception("asdada");
}

dVector LossFunction::loss(dVector y_true,dVector y_pred2, dVector y_pred1, double lambda){
    if(citerion ==0)
        return (y_true.array() - y_pred2.array()).square() + lambda*(y_pred1.array() - y_pred2.array()).square();

    if(citerion ==1)
        return (y_pred2.array().exp() - y_true.array()*y_pred2.array()) + lambda*(y_pred2.array().exp() - y_pred1.array()*y_pred2.array());

    throw exception("asdada");
}



dVector LossFunction::dloss(dVector y_true,dVector y_pred2, dVector y_pred1, double gamma, dVector weights ){
    // printf("loss_with_reg , %f %f %f %f\n", lambda, y_true.array().sum(),y_pred2.array().sum(), y_pred1.array().sum());
    // printf("%f %f %f\n",(y_pred2.array() - y_true.array()).sum(), (y_pred2.array()- y_pred1.array()).sum(),  ((y_pred2.array()- y_true.array())+ (y_pred2.array()- y_pred1.array())).sum());
        
    if(citerion ==0){
       return 2*(y_pred2.array()- y_true.array())*weights.array()+ gamma*2*(y_pred2.array()- y_pred1.array());
    }
        

    if(citerion ==1){
        return ((y_pred2.array().exp() - y_true.array()))*weights.array() +gamma*(y_pred2.array().exp() - y_pred1.array()); //gamma*(y_pred2.array().exp() - y_pred1.array());
    }
        

    throw exception("asdada");
}

dVector LossFunction::ddloss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma, dVector weights){
    if(citerion ==0){
        return 2.0*weights.array() + dVector::Constant(y_true.size(),0, 2.0*gamma ).array() ;
    }

    if(citerion ==1){
        return (y_pred2.array().exp()*weights.array()) +y_pred2.array().exp()*gamma; // gamma*(y_pred2.array().exp()); //y_pred.array()/ y_true.array().square() ;
    }
        
    throw exception("asdada");
}












dVector LossFunction::dloss(dVector y_true,dVector y_pred2, dVector y_pred1, dVector gamma, dVector weights ){
  
    if(citerion ==0){
       return 2*(y_pred2.array()- y_true.array())*weights.array()+ gamma.array()*2*(y_pred2.array()- y_pred1.array());
    }
        

    if(citerion ==1){
        return ((y_pred2.array().exp() - y_true.array()))*weights.array() +gamma.array()*(y_pred2.array().exp() - y_pred1.array()); //gamma*(y_pred2.array().exp() - y_pred1.array());
    }
        

    throw exception("asdada");
}

dVector LossFunction::ddloss(dVector y_true,dVector y_pred2, dVector y_pred1 , dVector gamma, dVector weights){
    if(citerion ==0){
        return 2.0*weights.array() + 2.0*gamma.array();
    }

    if(citerion ==1){
        return (y_pred2.array().exp()*weights.array()) +y_pred2.array().exp()*gamma.array(); // gamma*(y_pred2.array().exp()); //y_pred.array()/ y_true.array().square() ;
    }
        
    throw exception("asdada");
}

#endif


// #pragma once
// #ifndef __LOSSFUNCTIONS_HPP_INCLUDED__
// #define __LOSSFUNCTIONS_HPP_INCLUDED__
// #include <Eigen/Dense>


// using Eigen::Dynamic;
// using dVector = Eigen::Matrix<double, Dynamic, 1>;
// using bVector = Eigen::Matrix<bool, Dynamic, 1>;
// using iVector = Eigen::Matrix<int, Dynamic, 1>;
// using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
// using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;

// using namespace std;


// class LossFunction{

//     public:
//         LossFunction();
//         LossFunction(int _citerion);
//         ~LossFunction();
//         dVector link_function(dVector y_pred );
//         double link_function(double y_pred );
//         dVector inverse_link_function(dVector y_pred_transformed );
//         double inverse_link_function(double y_pred_transformed );
//         dVector loss(dVector y_true,dVector y_pred );
//         dVector dloss(dVector y_true,dVector y_pred );
//         double dloss(double y_true,double y_pred );
//         dVector ddloss(dVector y_true,dVector y_pred );
//         double ddloss(double y_true,double y_pred );
//         dVector loss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma);
//         dVector dloss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma, dVector weights );
//         dVector ddloss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma, dVector weights );
//         int citerion;
        
// };

// LossFunction::LossFunction(){
//     this->citerion = 0; //"mse"
// }
// LossFunction::LossFunction(int citerion){
//     this->citerion = citerion; 
// }

// dVector LossFunction::link_function(dVector y_pred ){
//     if(citerion ==1)
//          y_pred = y_pred.array().log();

//     return y_pred;

// }
// double LossFunction::link_function(double y_pred ){
//     if(citerion ==1)
//          y_pred = log(y_pred);

//     return y_pred;

// }
// dVector LossFunction::inverse_link_function(dVector y_pred_transformed ){
//     if(citerion ==1)
//         y_pred_transformed = y_pred_transformed.array().exp();

//     return y_pred_transformed;

// }
// double LossFunction::inverse_link_function(double y_pred_transformed ){
//     if(citerion ==1)
//         y_pred_transformed = exp(y_pred_transformed);
//     return y_pred_transformed;
// }


// dVector LossFunction::loss(dVector y_true, dVector y_pred  ){
//     if(citerion ==0)
//         return (y_true.array() - y_pred.array()).square();

//     if(citerion ==1)
//         return (y_pred.array().exp() - y_true.array()*y_pred.array());

//     throw exception("asdada");
// }

// dVector LossFunction::dloss(dVector y_true,dVector y_pred  ){
//     if(citerion ==0){
//         return 2*(y_pred.array() - y_true.array());
//     }
        

//     if(citerion ==1){
//         return y_pred.array().exp() - y_true.array(); //1- y_true.array(); //
//     }
        
//     throw exception("asdada");
// }

// double LossFunction::dloss(double y_true,double y_pred ){
//     if(citerion ==0){
//         return 2*(y_pred- y_true);
//     }
        

//     if(citerion ==1){
//         return exp(y_pred) - y_true; //1- y_true.array(); //
//     }
        
//     throw exception("asdada");

// }

// dVector LossFunction::ddloss(dVector y_true,dVector y_pred  ){
//     if(citerion ==0){
//         //printf("%d \n ", citerion);
//         return dVector::Constant(y_true.size(),0,2.0);
//     }

//     if(citerion ==1){
//        // printf("%d \n ", citerion);
//         return y_pred.array().exp();
//     }
        

//     throw exception("asdada");
// }
// double LossFunction::ddloss(double y_true,double y_pred  ){
//     if(citerion ==0){
//         return 2;
//     }

//     if(citerion ==1){
//         return exp(y_pred); //y_pred.array()/ y_true.array().square() ;
//     }
        

//     throw exception("asdada");
// }

// dVector LossFunction::loss(dVector y_true,dVector y_pred2, dVector y_pred1, double lambda){
//     if(citerion ==0)
//         return (y_true.array() - y_pred2.array()).square() + lambda*(y_pred1.array() - y_pred2.array()).square();

//     if(citerion ==1)
//         return (y_pred2.array().exp() - y_true.array()*y_pred2.array()) + lambda*(y_pred2.array().exp() - y_pred1.array()*y_pred2.array());

//     throw exception("asdada");
// }



// dVector LossFunction::dloss(dVector y_true,dVector y_pred2, dVector y_pred1, double gamma, dVector weights ){
//     // printf("loss_with_reg , %f %f %f %f\n", lambda, y_true.array().sum(),y_pred2.array().sum(), y_pred1.array().sum());
//     // printf("%f %f %f\n",(y_pred2.array() - y_true.array()).sum(), (y_pred2.array()- y_pred1.array()).sum(),  ((y_pred2.array()- y_true.array())+ (y_pred2.array()- y_pred1.array())).sum());
        
//     if(citerion ==0){
//        return 2*(y_pred2.array()- y_true.array())*weights.array()+ gamma*2*(y_pred2.array()- y_pred1.array());
//     }
        

//     if(citerion ==1){
//         return ((y_pred2.array().exp() - y_true.array()))*weights.array() +gamma*(y_pred2.array().exp() - y_pred1.array()); //gamma*(y_pred2.array().exp() - y_pred1.array());
//     }
        

//     throw exception("asdada");
// }

// dVector LossFunction::ddloss(dVector y_true,dVector y_pred2, dVector y_pred1 , double gamma, dVector weights){
//     if(citerion ==0){
//         return 2.0*weights.array() + dVector::Constant(y_true.size(),0, 2.0*gamma ).array() ;
//     }

//     if(citerion ==1){
//         return (y_pred2.array().exp()*weights.array()) +y_pred1.array().exp()*gamma; // gamma*(y_pred2.array().exp()); //y_pred.array()/ y_true.array().square() ;
//     }
        
//     throw exception("asdada");
// }




// #endif