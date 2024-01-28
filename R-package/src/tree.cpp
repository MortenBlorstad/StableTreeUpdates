#include "tree.hpp"


// Expose the classes

RCPP_MODULE(TreeModule) {
    using namespace Rcpp;
    
    class_<Tree>("Tree")
        .default_constructor("Default constructor")
        .constructor<int, int , double,int, bool, int ,double, unsigned int, double,double>()
        .method("learn", &Tree::learn)
        .method("get_root", &Tree::get_root)
        .method("predict", &Tree::predict)
        .method("predict_uncertainty", &Tree::predict_uncertainty)
        .method("update", &Tree::update)
    ;
    
}