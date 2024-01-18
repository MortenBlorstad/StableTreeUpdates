#include <vector>
#include <omp.h>
#include <chrono>  // for std::chrono_literals and std::this_thread::sleep_for
#include <thread>  // for std::this_thread::sleep_for
#include <iostream>

class ParallelSum {
    public:
        ParallelSum(const std::vector<double>& v);
        double sum();
        double slowsum();
        dVector learn(const dMatrix &X,const dVector &y , const dVector weights );
        void learnslow(const dMatrix &X, const dVector &y , const dVector weights) ;
        private:
            std::vector<double> vec;

};

    ParallelSum::ParallelSum(const std::vector<double>& v){
        vec = v;
    }
    double ParallelSum::sum() {
        double sum = 0.0;
        std::vector<double> vec2(vec.size());
        #pragma omp parallel for
        for (size_t i = 0; i < vec.size(); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // sleep for 500 milliseconds
            vec2[i] = vec[i];
        }
        return vec2[0];
    }

    double ParallelSum::slowsum() {
        double sum = 0.0;
        std::vector<double> vec2(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));  // sleep for 500 milliseconds
            vec2[i] = vec[i];
        }
        return vec2[0];
    }

    dVector ParallelSum::learn(const dMatrix &X, const dVector &y, const dVector weights ) {
        std::vector<Tree> vec2(100);
        #pragma omp parallel for
        for (size_t i = 0; i < 100; i++) {
            printf("tree %d\n", i);
            vec2[i] = Tree(0, 1000,  5.0, 5,  true,  1 ,1,  i);
        }
        #pragma omp parallel for
        for (int i = 0; i < 100; i++) {
            vec2[i].learn(X,y, weights);
        }
        dVector prediction(X.rows()); 
        prediction.setConstant(0);
        #pragma omp parallel for
        for (int i = 0; i < 100; ++i) {
            if(&vec2[i] ==NULL){
                throw exception("tree is null");
            }

            prediction += vec2[i].predict(X);
        }
        printf("update\n");
        #pragma omp parallel for
        for (int i = 0; i < 100; i++) {
            printf("u %d \n", i);
            vec2[i].update(X,y,weights);
        }

        prediction.setConstant(0);
        #pragma omp parallel for
        for (int i = 0; i < 100; ++i) {
            if(&vec2[i] ==NULL){
                throw exception("tree is null");
            }

            prediction += vec2[i].predict(X);
        }

        return prediction = prediction.array()/100;

    }
    //Tree(int _criterion,int max_depth, double min_split_sample,int min_samples_leaf, bool adaptive_complexity, int max_features,double learning_rate, unsigned int random_state); 

void ParallelSum::learnslow(const dMatrix &X,const dVector &y , const dVector weights ) {
        std::vector<Tree> vec2(100);
        for (size_t i = 0; i < 100; i++) {
            printf("tree %d\n", i);
            vec2[i]= Tree(0, 1000,  5.0, 5,  true,  1 ,1,  i);
        }
        printf("learn\n");
        printf("vec2 %d\n", vec2.size());
        for (int i = 0; i < vec2.size(); i++) {
            printf("l %d\n", i);
            vec2[i].learn(X,y,weights);
        }
        printf("update\n");
        for (int i = 0; i < vec2.size(); i++) {
            printf("u %d \n", i);
            vec2[i].update(X,y,weights);
        }

    }
