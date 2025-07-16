#ifndef _GKM_H
#define _GKM_H

#include<iostream>
#include<algorithm>
#include<vector>
#include <unordered_map>
#include<set>
#include "CppFuns.h"
// #include <sstream>
// #ifdef _WIN32
// #define WIN32_LEAN_AND_MEAN
// #include <windows.h>
// #include <psapi.h>
// #endif

// #include <windows.h>
// #include <psapi.h>

using namespace std;

class GKM{

private:
    int c_true;
    int N;
    std::vector<std::vector<int>> NN;
    std::vector<std::vector<double>> NND;
    double max_distance;
    size_t  knn;
    int debug = 0;
    

    //std::vector<int> labels;
    std::vector<std::vector<int>> clusters_to_samples;
    std::vector<double> tightness;
    std::vector<std::unordered_map<int, double>> nearest_clusters;
    //std::set<std::tuple<double, int, int>> tree;

    void initializeClusters();
    void buildTree();
    //void mergeClusters();
    void mergeClusters(int i, int j, double min_distance);
    void symmetry(std::vector<std::vector<int>> &NN, std::vector<std::vector<double>> &NND);
    double maximum_2Dvec(std::vector<std::vector<double>> &Vec);
    std::vector<int> replace_with_continuous(const std::vector<int>& nums);

public:
    std::vector<int> labels;
    std::set<std::tuple<double, int, int>> tree;
    int c_now;
    double loss = 0;
    bool connected = true; //k近邻图是否为全连接图，决定合并是否会正常结束，默认为True
    
    GKM();
    GKM(std::vector<std::vector<int>> &NN,std::vector<std::vector<double>> &NND,int c_true,int debug);
    ~GKM();

    void opt();


};



#endif