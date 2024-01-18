
#include <pybind11\pybind11.h>
#include <pybind11\eigen.h>
#include <cstdio>
#include <omp.h>


using namespace std;


#include <pybind11/stl.h>


#include "node.hpp"
#include "splitters\splitter.hpp"
#include "criterions\criterion.hpp"
#include "trees\tree.hpp"
#include "trees\sttree.hpp"
#include "trees\newtree.hpp"
#include "trees\abutree.hpp"
#include "trees\abutree2.hpp"
#include "criterions\MSE.hpp"
#include "criterions\Poisson.hpp"
#include "optimism\cir.hpp"
#include "trees\naiveupdate.hpp"
#include "trees\stabilityregularization.hpp"
#include "trees\treereevaluation.hpp"
#include "randomforest\GBT.hpp"
#include "randomforest\randomforest.hpp"
#include "randomforest\stablelossrandomforest.hpp"
#include "randomforest\naiveupdaterandomforest.hpp"
#include "randomforest\treereevaluationrandomforest.hpp"
#include "randomforest\baburandomforest.hpp"
#include "randomforest\aburandomforest.hpp"
#include "randomforest\stackedrandomforest.hpp"
#include "agtboost\agtboost.hpp"
#include "test_parallel_function.hpp"
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

// std::string serialize(Node *node)
// {   
//     if (node == nullptr) {
//         return "null";
//     }
//     std::ostringstream oss;
//     // Else, store information on node
//     oss << node->split_feature << ','
//         << node->prediction << ','
//         << node->n_samples << ','
//         << node->split_score << ','
//         << node->split_value << ','
//         << node->impurity << ','
//         << node->y_var << ','
//         << node->w_var << ','
//         << node->parent_expected_max_S << '\n';
//     if (node->left_child != nullptr) {
//         oss << serialize(node->left_child);
//     } else {
//         oss << "null\n";
//     }
//     if (node->right_child != nullptr) {
//         oss << serialize(node->right_child);
//     } else {
//         oss << "null\n";
//     }
//     return oss.str();
    
// }

// Node* deserialize(std::istringstream &iss) {
//     std::string token;
//     std::getline<char>(iss, token, ',');
//     if (token == "null") {
//         return nullptr;
//     }
//     int split_feature = std::stoi(token);

//     Node *node = new Node;
//     node->split_feature = split_feature;
//     iss >> node->prediction >> node->n_samples >> node->split_score
//         >> node->split_value >> node->impurity >> node->y_var
//         >> node->w_var >> node->parent_expected_max_S;

//     node->left_child = deserialize(iss);
//     node->right_child = deserialize(iss);

//     return node;
// }

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

    py::class_<ParallelSum>(m, "ParallelSum")
        .def(py::init<const std::vector<double>&>())
        .def("sum", &ParallelSum::sum)
        .def("slowsum", &ParallelSum::slowsum)
        .def("learn", &ParallelSum::learn)
        .def("learnslow", &ParallelSum::learnslow);



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
        // .def("copy", &Node::copy)
        .def("toString", &Node::toString)
        .def("get_features_indices", &Node::get_features_indices)
        // .def(py::pickle(
        //     // Pickle function
        //     [](Node* obj) {
        //         return serialize(obj);
        //     },
        //     // Unpickle function
        //     [](std::istringstream &data) {
        //         return deserialize(data);
        //     }
        // ))
        //  .def(py::pickle(
        // [](const Node& node) { // dump
        //     return py::make_tuple(node.split_value,  node.split_score,
        //      node.n_samples, node.split_feature, node.left_child , node.right_child, node.prediction, node.y_var, node.w_var, node.features_indices );
        // },
        // [](py::tuple t) { // load
        //     return Node{t[0].cast<double>(), t[1].cast<double>(), t[2].cast<double>(), t[3].cast<int>(), t[4].cast<int>(), t[5].cast<double>(), t[6].cast<double>(),
        //                 t[7].cast<double>(), t[8].cast<iVector>()};
        // }))

        .def_readwrite("split_feature", &Node::split_feature)
        .def_readwrite("prediction", &Node::prediction)
        .def_readwrite("n_samples", &Node::n_samples)
        .def_readwrite("split_score", &Node::split_score)
        .def_readwrite("split_value", &Node::split_value)
        .def_readwrite("impurity", &Node::impurity)
        .def_readwrite("y_var", &Node::y_var)
        .def_readwrite("w_var", &Node::w_var);

    
    py::class_<Tree>(m, "Tree")
        .def(py::init<int, int , double,int, bool, int ,double, unsigned int>())
        .def("all_same", &Tree::all_same)
        .def("all_same_features_values", &Tree::all_same_features_values )
        .def("get_masks", &Tree::get_masks)
        .def("build_tree", &Tree::build_tree)
        .def("learn", &Tree::learn)
        .def("learn_difference", &Tree::learn_difference)
        .def("get_root", &Tree::get_root)
        .def("predict", &Tree::predict)
        .def("predict_uncertainty", &Tree::predict_uncertainty)
        .def("update", &Tree::update)
        .def("make_node_list", &Tree::make_node_list);

    py::class_<ENSEMBLE>(m, "agtboost")
        .def(py::init<>())
        .def("learn", &ENSEMBLE::train)
        .def("predict", &ENSEMBLE::predict)
        .def("update", &ENSEMBLE::update)
        .def("set_param", &ENSEMBLE::set_param)
        ;
    
    
    
     py::class_<NaiveUpdate>(m, "NaiveUpdate")
        .def(py::init<int, int, double,int, bool, int,double,unsigned int>())
            .def("learn", &NaiveUpdate::learn)
            .def("predict", &NaiveUpdate::predict)
            .def("update", &NaiveUpdate::update)
            .def("get_root", &NaiveUpdate::get_root);
        
    py::class_<TreeReevaluation>(m, "TreeReevaluation")
        .def(py::init<double, double,int, int, double,int, bool, int,double,unsigned int>())
            .def("learn", &TreeReevaluation::learn)
            .def("predict", &TreeReevaluation::predict)
            .def("update", &TreeReevaluation::update)
            .def("get_root", &TreeReevaluation::get_root)
            .def("get_mse_ratio", &TreeReevaluation::get_mse_ratio)
            .def("get_eps", &TreeReevaluation::get_eps)
            .def("get_obs", &TreeReevaluation::get_obs);



    py::class_<StabilityRegularization>(m, "StabilityRegularization")
         .def(py::init<double, int, int, double, int,bool, int,double,unsigned int>())
            .def("learn", &StabilityRegularization::learn)
            .def("predict", &StabilityRegularization::predict)
            .def("update", &StabilityRegularization::update)
            .def("get_root", &StabilityRegularization::get_root);

    py::class_<AbuTree2>(m, "AbuTree2")
         .def(py::init<int, int, double,int,bool, int, double,unsigned int, double,double>())
            .def("learn", &AbuTree2::learn)
            .def("predict", &AbuTree2::predict)
            .def("update", &AbuTree2::update)
            .def("predict_uncertainty", &AbuTree2::predict_uncertainty)
            .def("predict_info", &AbuTree2::predict_info)
            .def("get_root", &AbuTree2::get_root);
        

    py::class_<AbuTree>(m, "AbuTree")
        .def(py::init<int, int, double,int,bool, int, double,unsigned int>())
            .def("learn", &AbuTree::learn)
            .def("predict", &AbuTree::predict)
            .def("update", &AbuTree::update)
            .def("predict_uncertainty", &AbuTree::predict_uncertainty)
            .def("predict_info", &AbuTree::predict_info)
            .def("get_root", &AbuTree::get_root);
    
    py::class_<STTree>(m, "STTree")
        .def(py::init<int, int, double,int,bool, int, double,unsigned int>())
            .def("learn", &STTree::learn)
            .def("update", &STTree::update)
            .def("predict", &STTree::predict)
            .def("predict_uncertainty", &STTree::predict_uncertainty)
            .def("predict_info", &STTree::predict_info)
            .def("get_root", &STTree::get_root);

    py::class_<GBT>(m, "GBT")
        .def(py::init<int,int, int , double,int, bool ,double>())
        .def("learn", &GBT::learn)
        .def("predict", &GBT::predict)
        .def("update", &GBT::update);
    

    py::class_<RandomForest>(m, "RandomForest")
    .def(py::init<int,int, int , double,int, bool ,int>())
    .def("learn", &RandomForest::learn,py::call_guard<py::gil_scoped_release>())
    .def("predict", &RandomForest::predict)
    .def("update", &RandomForest::update);

    py::class_<StackedRandomForest>(m, "StackedRandomForest")
    .def(py::init<int,int, int , double,int, bool ,int,double,double>())
    .def("learn", &StackedRandomForest::learn,py::call_guard<py::gil_scoped_release>())
    .def("predict", &StackedRandomForest::predict)
    .def("update", &StackedRandomForest::update);

    py::class_<RandomForestSL>(m, "RandomForestSL")
    .def(py::init<int,int, int , double,int, bool ,int, double>())
    .def("learn", &RandomForestSL::learn,py::call_guard<py::gil_scoped_release>())
    .def("predict", &RandomForestSL::predict)
    .def("update", &RandomForestSL::update);

    py::class_<RandomForestNU>(m, "RandomForestNU")
    .def(py::init<int,int, int , double,int, bool ,int>())
    .def("learn", &RandomForestNU::learn,py::call_guard<py::gil_scoped_release>())
    .def("predict", &RandomForestNU::predict)
    .def("update", &RandomForestNU::update);
       
    py::class_<RandomForestTR>(m, "RandomForestTR")
    .def(py::init<int,int, int , double,int, bool ,int,double,double>())
    .def("learn", &RandomForestTR::learn,py::call_guard<py::gil_scoped_release>())
    .def("predict", &RandomForestTR::predict)
    .def("update", &RandomForestTR::update);

    py::class_<RandomForestABU>(m, "RandomForestABU")
    .def(py::init<int,int, int , double,int, bool ,int>())
    .def("learn", &RandomForestABU::learn,py::call_guard<py::gil_scoped_release>())
    .def("predict", &RandomForestABU::predict)
    .def("update", &RandomForestABU::update);

    py::class_<RandomForestBABU>(m, "RandomForestBABU")
    .def(py::init<int,int, int , double,int, bool ,int,int>())
    .def("learn", &RandomForestBABU::learn,py::call_guard<py::gil_scoped_release>())
    .def("predict", &RandomForestBABU::predict)
    .def("update", &RandomForestBABU::update);


    py::class_<MSE>(m, "MSE")
        .def(py::init<>())
            .def("get_score", &MSE::get_score)
            .def("init", &MSE::init)
            .def("update", &MSE::update)
            .def("get_root", &MSE::reset)
            .def("node_impurity", &MSE::node_impurity)
            .def("reset", &MSE::reset);

    py::class_<Poisson>(m, "Poisson")
        .def(py::init<>())
            .def("get_score", &Poisson::get_score)
            .def("init", &Poisson::init)
            .def("update", &Poisson::update)
            .def("get_root", &Poisson::reset)
            .def("node_impurity", &Poisson::node_impurity)
            .def("reset", &Poisson::reset);

    py::class_<MSEReg>(m, "MSEReg")
        .def(py::init<>())
            .def("get_score", &MSEReg::get_score)
            .def("init", &MSEReg::init)
            .def("update", &MSEReg::update)
            .def("get_root", &MSEReg::reset)
            .def("node_impurity", &MSEReg::node_impurity)
            .def("reset", &MSEReg::reset);

    py::class_<MSEABU>(m, "MSEABU")
    .def(py::init<int>())
        .def("get_score", &MSEABU::get_score)
        .def("init", &MSEABU::init)
        .def("update", &MSEABU::update)
        .def("get_root", &MSEABU::reset)
        .def("node_impurity", &MSEABU::node_impurity)
        .def("reset", &MSEABU::reset);
    
    py::class_<PoissonABU>(m, "PoissonABU")
    .def(py::init<>())
        .def("get_score", &PoissonABU::get_score)
        .def("init", &PoissonABU::init)
        .def("update", &PoissonABU::update)
        .def("get_root", &PoissonABU::reset)
        .def("node_impurity", &PoissonABU::node_impurity)
        .def("reset", &PoissonABU::reset);


    py::class_<PoissonReg>(m, "PoissonReg")
        .def(py::init<>())
            .def("get_score", &PoissonReg::get_score)
            .def("init", &PoissonReg::init)
            .def("update", &PoissonReg::update)
            .def("get_root", &PoissonReg::reset)
            .def("node_impurity", &PoissonReg::node_impurity)
            .def("reset", &PoissonReg::reset);



    py::class_<Splitter>(m, "Splitter")
        .def(py::init<int, double,bool, int, double>())
            .def("find_best_split", &Splitter::find_best_split)
            .def("get_reduction", &Splitter::get_reduction);

    py::class_<NewTree>(m, "NewTree")
        .def(py::init<int, int , double,int, bool, int ,double, unsigned int>())
            .def("build_tree", &NewTree::build_tree)
            .def("learn", &NewTree::learn)
            .def("get_root", &NewTree::get_root)
            .def("predict", &NewTree::predict)
            .def("update", &NewTree::update);


    



    m.def("rnchisq", &rnchisq);
    m.def("cir_sim_vec",&cir_sim_vec);
    m.def("cir_sim_mat",&cir_sim_mat);
    


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif

}