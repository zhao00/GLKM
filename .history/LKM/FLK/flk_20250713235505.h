#ifndef FLK_H_
#define FLK_H_
#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include <list>
#include <chrono>
#include <cmath>
#include <execution>
// #include "Keep_order.h"

using namespace std;

class flk{
public:
    // local
    vector<vector<int>> NN;
    vector<vector<double>> NND;
    double max_d;
    vector<int> knn_c;
    vector<int> knn_cou;
    vector<bool> c_indicator;

    bool debug;
    int num;
    vector<double> diYr;
    vector<double> sd;
    vector<int> y;
    vector<double> n;
    vector<vector<int>> Y;
    vector<unsigned int> n_iter;
    int c_true;
    bool local;
    vector<double> obj_knn;

    flk();
    flk(int c_true, bool debug);
    flk(vector<vector<int>> &NN, vector<vector<double>> &NND, double max_d, int c_true, bool debug);
    ~flk();

    void opt(unsigned int ITER, vector<vector<int>> &init_Y);
    unsigned int opt_once(unsigned int ITER);

    void init();
    void check_NN();
    int update_yi(int sam_i);
    void maintain_var(int sam_i, int c_old, int c_new);
    int compute_diff(int i);
    double sum_distance_within_cluster(unsigned int i);

};
#endif
