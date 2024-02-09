
#include <pybind11\pybind11.h>
#include <pybind11\eigen.h>
#include <cstdio>
#include <omp.h>


using namespace std;


#include <pybind11/stl.h>


#include "node.hpp"

#include "tree.hpp"
//#include "trees\abutree.hpp"



#include "cir.hpp"



#include <string>
#include <sstream>
#include <iostream>
#include <fstream>


#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(_stabletrees, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: stable_trees
        .. autosummary::
           :toctree: _generate

           get_predictions

    )pbdoc";





    py::class_<Node>(m, "Node")
        .def(py::init<double,double, double, int, int, double,double,double,std::vector<int>>())
        .def(py::init<double, int>())
        .def("is_leaf", &Node::is_leaf)
        .def("set_left_node", &Node::set_left_node)
        .def("set_right_node", &Node::set_right_node)
        .def("get_right_node", &Node::get_right_node)
        .def("get_left_node", &Node::get_left_node)
        .def("predict", &Node::predict)
        .def("nsamples", &Node::nsamples)
        .def("get_split_score", &Node::get_split_score)
        .def("get_impurity", &Node::get_impurity)
        .def("get_split_feature", &Node::get_split_feature)
        .def("get_split_value", &Node::get_split_value)
        .def("toString", &Node::toString)
        .def("get_features_indices", &Node::get_features_indices)
        

        .def_readwrite("split_feature", &Node::split_feature)
        .def_readwrite("prediction", &Node::prediction)
        .def_readwrite("n_samples", &Node::n_samples)
        .def_readwrite("split_score", &Node::split_score)
        .def_readwrite("split_value", &Node::split_value)
        .def_readwrite("impurity", &Node::impurity)
        .def_readwrite("y_var", &Node::y_var)
        .def_readwrite("w_var", &Node::w_var);

    
    py::class_<Tree>(m, "Tree")
        .def(py::init<int, int , double,int, bool, int ,double, unsigned int, double,double>())
        .def("learn", &Tree::learn)
        .def("get_root", &Tree::get_root)
        .def("predict", &Tree::predict)
        .def("predict_uncertainty", &Tree::predict_uncertainty)
        .def("predict_info", &Tree::predict_info)
        .def("update", &Tree::update);




    // py::class_<AbuTree>(m, "AbuTree")
    //      .def(py::init<int, int, double,int,bool, int, double,unsigned int, double,double>())
    //         .def("learn", &AbuTree::learn)
    //         .def("predict", &AbuTree::predict)
    //         .def("update", &AbuTree::update)
    //         .def("predict_uncertainty", &AbuTree::predict_uncertainty)
    //         .def("predict_info", &AbuTree::predict_info)
    //         .def("get_root", &AbuTree::get_root);
        
    py::class_<Splitter>(m, "Splitter")
        .def(py::init<int, double,bool, int, double>())
            .def("find_best_split", &Splitter::find_best_split)
            .def("get_reduction", &Splitter::get_reduction);




    



    m.def("rnchisq", &rnchisq);
    m.def("cir_sim_vec",&cir_sim_vec);
    m.def("cir_sim_mat",&cir_sim_mat);
    


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}