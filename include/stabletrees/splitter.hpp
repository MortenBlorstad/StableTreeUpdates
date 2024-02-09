#pragma once
#ifndef __Splitter_HPP_INCLUDED__

#define __SLITTER_HPP_INCLUDED__

//#include <C:\Users\mb-92\OneDrive\Skrivebord\studie\StableTrees\cpp\thirdparty\eigen\Eigen/Dense>
#include <Eigen/Dense>
#include <unordered_set>
#include "cir.hpp"
#include "gumbel.hpp"
#include "utils.hpp"
#include <omp.h>
#include <vector>

using Eigen::Dynamic;
using dVector = Eigen::Matrix<double, Dynamic, 1>;
using bVector = Eigen::Matrix<bool, Dynamic, 1>;
using dMatrix = Eigen::Matrix<double, Dynamic, Dynamic>;
using iMatrix = Eigen::Matrix<int, Dynamic, Dynamic>;
using iVector = Eigen::Matrix<int, Dynamic, 1>;
using dArray = Eigen::Array<double,Eigen::Dynamic,1>;

using namespace std;



class Splitter{

    public:
        Splitter();
        Splitter(int min_samples_leaf, double _total_obs, bool _adaptive_complexity,int max_features, double learning_rate);
        virtual tuple<bool,int,double,double,double,double,double>  find_best_split(const dMatrix  &X, const dVector  &y,const dVector &g,const dVector &h,const std::vector<int> &features_indices);
        dMatrix cir_sim;
        virtual ~Splitter();
        double get_reduction(const dVector &g,const dVector &h, const iVector mask_left);
        double total_obs;
        
    protected:
        
        int grid_size = 101;
        dVector grid;
        dArray gum_cdf_mmcir_grid;
        bool adaptive_complexity;
        int min_samples_leaf;
        int max_features;
        double learning_rate;

};

Splitter::Splitter(){
    adaptive_complexity = false;
    this->min_samples_leaf = 1;
    cir_sim = cir_sim_mat(500,500,1);
}

Splitter::Splitter(int min_samples_leaf,double _total_obs, bool _adaptive_complexity, int max_features, double learning_rate){
    total_obs = _total_obs;
    adaptive_complexity =_adaptive_complexity;
    this->min_samples_leaf = min_samples_leaf;
    this->max_features = max_features;

    cir_sim = cir_sim_mat(500,500,1);
    this->learning_rate = learning_rate;
}

Splitter::~Splitter(){

    total_obs = NULL;
    adaptive_complexity = NULL;
    min_samples_leaf = NULL;
    grid_size = NULL;
}

double Splitter::get_reduction(const dVector &g,const dVector &h, const iVector mask_left){
    double G=g.array().sum(), H=h.array().sum();
    double Gl=g(mask_left).array().sum(), Hl=h(mask_left).array().sum();
    double Gr = G - Gl, Hr = H - Hl;
    double n = g.size();
    double reduction= ((Gl*Gl)/Hl + (Gr*Gr)/Hr - (G*G)/H)/(2*n);
    return reduction;
}

tuple<bool,int,double,double,double,double,double> Splitter::find_best_split(const dMatrix  &X, const dVector  &y, const dVector &g, const dVector &h,const std::vector<int> &features_indices){
    int n = y.size(); //number of obs in node
    double observed_reduction = -std::numeric_limits<double>::infinity();
    double score;
    double split_value;
    int split_feature;
    bool any_split = false;
    
    
    


    iMatrix X_sorted_indices = sorted_indices(X);
    dVector feature;
    double G=g.array().sum(), H=h.array().sum(), G2=g.array().square().sum(), H2=h.array().square().sum(), gxh=(g.array()*h.array()).sum();
    double Gl_final; double Hl_final;
    double grid_end = 1.5*cir_sim.maxCoeff();
    dVector grid = dVector::LinSpaced( grid_size, 0.0, grid_end );
    gum_cdf_mmcir_grid = dArray::Ones(grid_size);
    dVector gum_cdf_mmcir_complement(grid_size);
    int num_splits;
    dVector u_store((int)n);
    double prob_delta = 1.0/n;
    dArray gum_cdf_grid(grid_size);
    double optimism = (G2 - 2.0*gxh*(G/H) + G*G*H2/(H*H)) / (H*n); //equation 30, lunde et al
    double expected_max_S;
    double conditional_variance_without_profiling = optimism/(H/n); //equation 19, lunde et al  //((n/total_obs))*
    
    double w_var = conditional_variance_without_profiling;
    //double w_var =  (n/total_obs)*(g.array() + h.array()*(-G/H) ).square().sum()/(H*H); //conditional_variance_without_profiling*n; // later multiplied by the effect of profiling (tree.hpp, in build_tree and update_tree method, e.g line 553 and 562)
    //(g.array() + h.array()*(-G/H) ).square().sum()/(H2) ; //

    
    //double y_var = n*conditional_variance_without_profiling; 
    double y_var = (y.array() - y.array().mean()).square().mean(); 
    //double w_var = y_var/n;
    //double w_var = total_obs*(n/total_obs)*(optimism/(H));
    //double y_var =  n * (n/total_obs) * total_obs * (optimism / H ); //(y.array() - y.array().mean()).square().mean();


   
    //for(int j = 0; j<X.cols();j++){
    for(int j = 0; j<features_indices.size(); j++){
        
        int col_index = features_indices[j];//j; //
        //printf("%d, %d\n",j, col_index);
        int nl = 0; int nr = n;
        double Gl = 0, Gl2 = 0, Hl=0, Hl2=0, Gr=G, Gr2 = G2, Hr=H, Hr2 = H2;
        feature = X.col(col_index);
        num_splits = 0;
        iVector sorted_index = X_sorted_indices.col(col_index);
        double largestValue = feature(sorted_index[n-1]);
        u_store = dVector::Zero(n);
        for (int i = 0; i < n-1; i++) {
            int low = sorted_index[i];
            int high = sorted_index[i+1];
            double lowValue = feature[low];
            double hightValue = feature[high];
            double middle =  (lowValue+hightValue)/2;
            
            // increment g and h values -------------
            double g_i = g(low);
            double h_i = h(low);
            Gl += g_i; Hl += h_i;
            Gl2 += g_i*g_i; Hl2 += h_i*h_i;
            Gr = G - Gl; Hr = H - Hl;
            Gr2 = G2 - Gl2; Hr2 = H2 - Hl2;
            nl+=1;
            nr-=1;
            //------------------------------------
            if(lowValue == largestValue){// no unique feature values. cannot split on this feature. 
                break;
            }
            if(hightValue-lowValue<0.00000000001){// skip if values are approx equal. not valid split
                continue;
            }
            if(nl< min_samples_leaf || nr < min_samples_leaf){
                continue;
            }
            u_store[num_splits] = nl*prob_delta;
            num_splits +=1;
            score  =  ((Gl*Gl)/(Hl) + (Gr*Gr)/(Hr) - (G*G)/(H))/(2*n);
            
            if(observed_reduction<score){
                any_split = true;
                observed_reduction = score;
                split_value = middle;
                split_feature = col_index;  
                Gl_final = Gl;
                Hl_final = Hl;
            }
            if(std::isnan(observed_reduction) || std::isinf(observed_reduction)){
                std::cout << "Gl: " <<  Gl<< std::endl;
                std::cout << "Hl: " <<  Hl<< std::endl;
                std::cout << "nl: " <<  nl << std::endl;
                std::cout << "observed_reduction:  " <<  observed_reduction  <<std::endl;
                std::cout << "Gr:  " <<  Gr << std::endl;
                std::cout << "Hr:  " <<  Hr << std::endl;
                std::cout << "nr: " <<  nr << std::endl;
                std::cout << "G:  " <<  G << std::endl;
                std::cout << "H:  " <<  H << std::endl;
                std::cout << "n: " <<  n << std::endl;
                std::cout << "score:  " <<  score << std::endl;
                std::cout << "hi:  " <<  h_i << std::endl;
                throw exception("observed_reduction is inf" );

            } 
        }
        if(adaptive_complexity && num_splits>0){
            dVector u = u_store.head(num_splits);
            dArray max_cir = rmax_cir(u, cir_sim); // Input cir_sim!
            if(num_splits>1){
                // Asymptotically Gumbel
                    
                // Estimate Gumbel parameters
                dVector par_gumbel = par_gumbel_estimates(max_cir);
                // Estimate cdf of max cir for feature j
                for(int k=0; k< grid_size; k++){ 
                    gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
                }

            }else{
                

                // Asymptotically Gumbel
                    
                // Estimate Gumbel parameters
                dVector par_gumbel = par_gumbel_estimates(max_cir);
                // Estimate cdf of max cir for feature j
                for(int k=0; k< grid_size; k++){ 
                    gum_cdf_grid[k] = pgumbel<double>(grid[k], par_gumbel[0], par_gumbel[1], true, false);
                }
            }
            // Update empirical cdf for max max cir
            gum_cdf_mmcir_grid *= gum_cdf_grid; 
        }

    }
    
    if(any_split && adaptive_complexity){
        gum_cdf_mmcir_complement = dVector::Ones(grid_size) - gum_cdf_mmcir_grid.matrix();
        expected_max_S = simpson( gum_cdf_mmcir_complement, grid );
        double CRt = optimism * (n/total_obs)  *expected_max_S;
        double expected_reduction = learning_rate*(2.0-learning_rate)*observed_reduction*((n/total_obs) )  - learning_rate*CRt;

        // std::cout << "local_optimism: " <<  optimism<< std::endl;
        // std::cout << "CRt: " <<  CRt << std::endl;
        // std::cout << "n:  " <<  n  <<std::endl;
        // std::cout << "prob_node:  " <<  n/total_obs << std::endl;
        // std::cout << "expected_max_S:  " <<  expected_max_S << std::endl;
        // std::cout << "observed_reduction:  " <<  observed_reduction << std::endl;
        // std::cout << "expected_reduction:  " <<  expected_reduction <<std::endl;
        // std::cout << "G:  " << G << std::endl;
        // std::cout << "H:  " << H <<std::endl;
        // std::cout << "Gl: " <<  Gl_final<< std::endl;
        // std::cout << "Hl: " <<  Hl_final << std::endl;
        // std::cout << "seed: " <<  get_seed() << std::endl;
        // std::cout << "num_splits: " <<  num_splits << std::endl;
        // std::cout <<"\n" << std::endl;


        if(any_split && n/total_obs!=1.0 && expected_reduction<0.0){
            any_split = false;
        }
    }

    return tuple<bool,int,double,double,double,double,double>(any_split, split_feature, split_value,observed_reduction,y_var,w_var,expected_max_S);

}



#endif