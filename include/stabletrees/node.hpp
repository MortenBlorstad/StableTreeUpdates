
#ifndef __NODE_HPP_INCLUDED__

#define __NODE_HPP_INCLUDED__

#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Eigen;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
class Node{
    public:
        Node* left_child;
        Node* right_child;
        double split_value= NULL; // split value
        double prediction= NULL; // node's prediction 
        int n_samples= NULL; // number of sample in node
        int split_feature= NULL; //  index of feature to split
        double split_score = NULL; // split reduduction
        double impurity; // node impurity
        double y_var;// variance to the response variable in a node
        double w_var;// variance to the prediction in a node
        double parent_expected_max_S; 
        Node();
        Node(double _split_value,double _impurity, double _split_score, int _split_feature, int _n_samples, double _prediction);
        Node(double _split_value,double _impurity, double _split_score, int _split_feature, int _n_samples, double _prediction, double y_var, double w_var);
        Node(double _split_value,double _impurity, double _split_score, int _split_feature, int _n_samples, double _prediction, double y_var, double w_var, std::vector<int> &features_indices);
        Node(double _prediction, int _n_samples, double y_var, double w_var);
        Node(double _prediction, int _n_samples);


        Node* get_left_node();
        Node* get_right_node();
        void set_left_node(Node* node);
        void set_right_node(Node* node); 
        bool is_leaf();
        double predict();
        int nsamples();
        int get_split_feature();
        double get_split_value();
        double get_split_score();
        double get_impurity();
        std::vector<int> get_features_indices() const;
        // std::string serialize();
        // Node* deSerialize(std::ifstream& f);
        //Node* copy();
        //~Node();
        std::string toString();

        std::vector<int> features_indices;

    // private:
    //     Node* copy_rec(Node* node);
    //     void delete_node(Node* node);

};

    std::string Node::toString() {
        if(is_leaf()){
            return "(prediction: " + std::to_string(predict())+ ", nsamples: " + std::to_string(nsamples()) + ")";
        }else{
            return "(feature: " + std::to_string(get_split_feature()) + " <= " + std::to_string(get_split_value()) + ", nsamples: " + std::to_string(nsamples()) + ")";
        }
}
// void Node::set_state(py::dict state) {
//     split_feature = state["split_feature"].cast<int>();
//     prediction = state["prediction"].cast<double>();
//     n_samples = state["n_samples"].cast<int>();
//     split_score = state["split_score"].cast<double>();
//     split_value = state["split_value"].cast<double>();
//     impurity = state["impurity"].cast<double>();
//     y_var = state["y_var"].cast<double>();
//     w_var = state["w_var"].cast<double>();
//     parent_expected_max_S = state["parent_expected_max_S"].cast<double>();
// }

// std::string Node::serialize()
// {   
//     std::ostringstream f;
//     // Check if file is open
   
//     // Check for null
//     int MARKER = -1;
//     if(this == NULL)
//     {
//         f << MARKER << "\n";
//         return "null";
//     }
//     // Else, store information on node
//     f << std::fixed << this->split_feature << " ";
//     f << std::fixed << this->prediction << " ";
//     f << std::fixed << this->n_samples << " ";
//     f << std::fixed << this->split_score << " ";
//     f << std::fixed << this->split_value << " ";
//     f << std::fixed << this->impurity << " ";
//     f << std::fixed << this->y_var << " ";
//     f << std::fixed << this->w_var << " ";
//     f << std::fixed << this->parent_expected_max_S << "\n";
//     // Recurrence
//     f <<this->left_child->serialize();
//     f <<this->right_child->serialize();
//     return f.str();
    
// }

// Node* Node::deSerialize(std::istringstream& f)
// {
    
//     int MARKER = -1;

//     std::string stemp;
//     if(!std::getline(f,stemp)){
//         return false;
//     }
    
//     // Check stemp for MARKER
//     std::istringstream istemp(stemp);
//     int val;
//     istemp >> val;
//     if(val == MARKER){
//         return nullptr;
//     }
//     Node *node = new Node;
//     // Load node
//     node->split_feature = val;
//     istemp >> node->prediction >> node->n_samples >> node->split_score >>
//         node->split_value >> node->impurity >> node->y_var >>
//         node->w_var >> node->parent_expected_max_S;
    
//     node->left_child->deSerialize(istemp);
//     node->right_child->deSerialize(istemp);

//     return node;
// }


    // Node* Node::copy_rec(Node* node){

    //     if(node==NULL){
    //         return NULL;
    //     }
    //     if(node->is_leaf()){
    //         Node* leaf = new Node(node->prediction, node->n_samples);
    //         leaf->left_child = copy_rec(node->left_child);
    //         leaf->right_child = copy_rec(node->right_child);
    //         return leaf;
    //     }
    //     Node* current = new Node(node->split_value, node->impurity,node->split_score, node->split_feature, node->n_samples, node->prediction);
    //     current->left_child = copy_rec(node->left_child);
    //     current->right_child = copy_rec(node->right_child);

    //     return current;    
    // }


    // Node* Node::copy(){
    //    return copy_rec(this);
    // }




// void Node::delete_node(Node* node){
//     if(node == NULL) return;
//     delete_node(node->left_child);
//     delete_node(node->right_child);
//     split_value = NULL;
//     impurity = NULL;
//     split_score = NULL;
//     n_samples = NULL;
//     split_feature = NULL;
//     left_child=NULL;
//     right_child=NULL;
//     prediction = NULL;
//     node = NULL;
// }
// Node::~Node(){
//     delete_node(this);
// }

Node::Node(){
}

Node::Node(double _split_value, double _impurity, double _split_score, int _split_feature,int _n_samples, double _prediction){
    split_value = _split_value;
    impurity = _impurity;
    split_score = _split_score;
    n_samples = _n_samples;
    split_feature = _split_feature;
    left_child=NULL;
    right_child=NULL;
    prediction = _prediction;


}
Node::Node(double _split_value, double _impurity, double _split_score, int _split_feature,int _n_samples, double _prediction, double y_var,double w_var){
    split_value = _split_value;
    impurity = _impurity;
    split_score = _split_score;
    n_samples = _n_samples;
    split_feature = _split_feature;
    left_child=NULL;
    right_child=NULL;
    prediction = _prediction;
    this->y_var = y_var;
    this->w_var = w_var;

}
Node::Node(double _split_value,double _impurity, double _split_score, int _split_feature, int _n_samples, double _prediction, double y_var, double w_var, std::vector<int> &features_indices){
    split_value = _split_value;
    impurity = _impurity;
    split_score = _split_score;
    n_samples = _n_samples;
    split_feature = _split_feature;
    left_child=NULL;
    right_child=NULL;
    prediction = _prediction;
    this->y_var = y_var;
    this->w_var = w_var;
    this->features_indices = features_indices;
}
    
Node::Node(double _prediction, int _n_samples, double y_var,double w_var){
    n_samples = _n_samples;
    prediction = _prediction;
    left_child=NULL;
    right_child=NULL;
    this->y_var = y_var;
    this->w_var = w_var;
}

Node::Node(double _prediction, int _n_samples){
    n_samples = _n_samples;
    prediction = _prediction;
    left_child=NULL;
    right_child=NULL;
}

std::vector<int> Node::get_features_indices() const{
    return features_indices;
}

void Node::set_left_node(Node* node){
    this->left_child = node;
}
void Node::set_right_node(Node* node){
    this->right_child = node;
}

double Node::predict(){
    if(std::isnan(this->prediction)|| std::isinf(this->prediction)){
        throw exception("node prediction is nan or inf: %f\n", this->prediction);
    }
    return this->prediction;
}
int Node::get_split_feature(){
    return this->split_feature;
}


double Node::get_split_value(){
    return this->split_value;
}

Node* Node::get_left_node(){
    return this->left_child;
}
Node* Node::get_right_node(){

    return this->right_child ;
}

int Node::nsamples(){
    return this->n_samples;
}

double Node::get_split_score(){
    return this-> split_score;
}

double Node::get_impurity()
{
    return this-> impurity;
}

bool Node::is_leaf(){
    return this->left_child== NULL && this->right_child== NULL;
    
}


#endif