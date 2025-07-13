#include "GKM.h"

// #include <iostream>

// void printMemoryUsage(const std::string& tag = "") {
//     PROCESS_MEMORY_COUNTERS_EX pmc;
//     if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
//         SIZE_T memUsedKB = pmc.WorkingSetSize / 1024;
//         std::cout << "[MEMORY] " << tag << " - WorkingSetSize: " << memUsedKB << " KB" << std::endl;
//     }
// }


GKM::GKM(){}

GKM::GKM(std::vector<std::vector<int>> &NN,std::vector<std::vector<double>> &NND,int c_true, int debug=0)
    :NN(NN), NND(NND), c_true(c_true), debug(debug){
    
    if(debug==1){
        std::cout<<"start!"<<std::endl;
    }

    this->knn = NN[0].size();
    // if(debug==1){
    //     std::cout<<"knn finished!"<<std::endl;
    // }

    //找到NND的最大值gamma
    std::vector<double> last_column;
    for(auto& row:NND){
        last_column.push_back(row.back());
    }
    this->max_distance = *std::max_element(last_column.begin(),last_column.end());
    // std::cout<<"maximum 2dvec finished!"<<std::endl;
        
    cf::symmetry(this->NN, this->NND, 1, max_distance);       //还是不行

    // this->max_distance = maximum_2Dvec(NND);

    // if(debug==1){
    //     std::cout<<"max_distance finished!\t"<<max_distance<<::endl;
    // }
    // printMemoryUsage("Before symmetry");
    // this->symmetry(this->NN, this->NND);    //对称化时内存爆了
    // std::cout<<"symmetry finished!"<<std::endl;
    // printMemoryUsage("After symmetry");
    // for(int i=0;i<this->NN.size();++i){
    //     for(int j=0;j<this->NN[i].size();++j)
    //         std::cout<<this->NN[i][j]<<" ";
    //     std::cout<<std::endl;
    // }
    // std::cout<<this->max_distance<<std::endl;
    // if(debug==1){
    //     std::cout<<"symetry finished!"<<std::endl;
    // }
    initializeClusters();
    // printMemoryUsage("After initializeClusters");
    buildTree();
    // printMemoryUsage("After buildTree");
}


GKM::~GKM(){}


//void GKM::symmetry(std::vector<std::vector<int>> &NN, std::vector<std::vector<double>> &NND);
void GKM::symmetry(std::vector<std::vector<int>> &NN, std::vector<std::vector<double>> &NND){

    if(debug==1){
        std::cout<<"symmetry start!"<<std::endl;
        std::cout<<NN.size()<<std::endl;
    }
    int N = NN.size();
    int knn = NN[0].size();

    std::vector<std::vector<int>> RNN;
    std::vector<std::vector<double>> RNND;
    RNN.resize(N);
    RNND.resize(N);

    if(debug==1){
        std::cout<< "breakpoint0" <<std::endl;
    }
    int tmp_j = 0;
    double tmp_d = 0;
    for (int i = 0; i < N; i++){
        for (int k = 0; k < knn; k++){
            // std::cout << NN[i].size()<<" "<< k <<" "<<NN[i][k]<<std::endl;
            // std::cout << NND[i].size()<<" "<< k <<" "<<NND[i][k]<<std::endl;
            tmp_j = NN[i][k];    // 反向查找
            tmp_d = NND[i][k];
            RNN[tmp_j].push_back(i);    // RNN的第tmp_j列存储指向其的近邻
            RNND[tmp_j].push_back(tmp_d);
        }
        if(debug==1){
            std::cout<< i <<std::endl;
        }
    }
    if(debug==1){
        std::cout<< "breakpoint1" <<std::endl;
    }
    std::vector<bool> flag(N, false);
    for (int i = 0; i < N; i++){    //逐个检查样本
        for (auto j : NN[i]){       // 初始时全部标记为true，即对样本i，其指向的前k个近邻，都做了标记
            flag[j] = true;
        }

        for (int k = 0; k < RNN[i].size(); k++){    //遍历指向样本i的样本（个数不确定，等于RNN[i].size())
            tmp_j = RNN[i][k];      // tmp_j的前k个近邻有i
            if (flag[tmp_j] == false){      // tmp_j不是i的近邻，但是i是tmp_j的近邻

                NN[i].push_back(tmp_j);

                tmp_d = RNND[i][k];
                NND[i].push_back(tmp_d);
            }
        }

        for (int k = 0; k < knn; k++){  
            tmp_j = NN[i][k];
            flag[tmp_j] = false;    //将 flag 重置为 false，以准备下一次循环。注意flag的长度为N
        }
        if(debug==1){
            std::cout<< i<<std::endl;
        }
    }
    if(debug==1){
        std::cout<<"symmetry finish!"<<std::endl;
    }
}
double GKM::maximum_2Dvec(std::vector<std::vector<double>> &Vec){
    int N = Vec.size();
    std::vector<double> tmp(N, 0);

    for(int i = 0; i < N; i++){
        tmp[i] = *max_element(Vec[i].begin(), Vec[i].end());
    }

    double ret = *max_element(tmp.begin(), tmp.end());
    return ret;
}

void GKM::initializeClusters(){
    // std::cout<<"initializing clusters"<<std::endl;
    N = NN.size();
    labels.resize(N);
    clusters_to_samples.resize(N);
    tightness.resize(N,0);
    nearest_clusters.resize(N);
    c_now = N;

    for(int i=0; i<N; ++i){     //对于大数据集，这个步骤慢吗？
        // if(i%1000==0){
        //     std::cout<<"\t i="<<i<<std::endl;
        // }
        labels[i] = i;
        clusters_to_samples[i].push_back(i);
    }

}
std::vector<int> GKM::replace_with_continuous(const std::vector<int>& nums) {
    std::unordered_map<int, int> num_dict;  // Create an unordered map to store the mapping of values
    std::vector<int> new_nums;  // Create a new vector to store the mapped values
    
    // Iterate through each number in the input vector
    for (int num : nums) {
        // If the number is not already in the map, map it to the current map size
        if (num_dict.find(num) == num_dict.end()) {
            num_dict[num] = num_dict.size();
        }
        // Add the mapped value to the new vector
        new_nums.push_back(num_dict[num]);
    }
    
    return new_nums;
}

void GKM::buildTree(){
    // std::cout<<"building tree"<<std::endl;
    for(int i = 0; i < N; ++i) { // Loop over all samples
        // if(i%1000==0){
        //     std::cout<<"\t i="<<i<<std::endl;
        // }
        for(size_t index = 0; index < NN[i].size(); ++index) { // Loop over the neighbors of sample i
            int j = NN[i][index]; // j is the neighbor of i
            if(i==j){
                continue;
            }

            if(nearest_clusters[j].find(i) == nearest_clusters[j].end()) { // If i is not in the nearest clusters of j
                double distances_temp = NND[i][index] / 2; // Compute the temporary distance

                tree.insert({distances_temp, std::min(i, j), std::max(i, j)}); // Add to the tree
                nearest_clusters[i][j] = distances_temp; // Update nearest clusters for i
                nearest_clusters[j][i] = distances_temp; // Update nearest clusters for j
            }
        }
    }


}


void GKM::opt(){
    //int c_now = N;
    //labels.resize(10,0);
    while(c_now > c_true){
        if(tree.empty()){   //如果是恰好结束时tree为空呢
            connected = false;
            //labels[0] = -1;
            break;  
        }
        // if(c_now%1000==0){
        //     std::cout<<"c_now: "<<c_now<<std::endl;
        // }
        // if(c_now%20000==0){
        //     std::ostringstream oss;
        //     oss << "c_now = " << c_now;
        //     printMemoryUsage(oss.str());
        // }

        //for (auto it = tree.begin(); it != tree.end(); ++it) {
        //    std::cout << std::get<0>(*it) << " " << std::get<1>(*it) << " " << std::get<2>(*it) << std::endl;
        //}/////////////////////在每一次循环开始前输出tree/////////////////////////////////////////
        auto min_element = *tree.begin();
        tree.erase(tree.begin());

        double min_distance =std::get<0>(min_element);
        int i = std::min(std::get<1>(min_element),std::get<2>(min_element));
        int j = std::max(std::get<1>(min_element),std::get<2>(min_element));

        nearest_clusters[i].erase(j);
        nearest_clusters[j].erase(i);


        loss += min_distance;
        mergeClusters(i, j, min_distance);  // 合并i簇、j簇
        c_now -= 1;


        if(debug==2){
            std::cout<<c_now<<":"<<i<<" "<<j<<std::endl;
            for(int i=0;i<labels.size();i++){
                std::cout<<labels[i]<<" ";
            }
            std::cout<<std::endl;
        }
    }
    labels = replace_with_continuous(labels);
    loss *= 2;
}

void GKM::mergeClusters(int i, int j, double min_distance){
    // std::cout<<max_distance<<std::endl;
    //std::cout<<"-----"<<i<<"-"<<j<<"-----------"<<std::endl;
    int n_i_old = clusters_to_samples[i].size();
    int n_j_old = clusters_to_samples[j].size();
    double tightness_i = tightness[i];
    double tightness_j = tightness[j];
    
    tightness[i] = min_distance + tightness[i] + tightness[j];


    //labels[i] = 10;
    for(int sample_label: clusters_to_samples[j]){
        labels[sample_label] = i;
        //labels[sample_label] = 999;
        //std::cout<<j<<"\t"<<sample_label<<"\t"<<i<<std::endl;

    }

    clusters_to_samples[i].insert(clusters_to_samples[i].end(), clusters_to_samples[j].begin(), clusters_to_samples[j].end());
    clusters_to_samples[j].clear();

    std::set<std::tuple<double, int, int>> List_2k;
    for (auto& [p, distance] : nearest_clusters[i]) {
        //std::cout<<"*****"<<i<<" "<<p<<std::endl;
        tree.erase({distance, std::min(i,p), std::max(i,p)});
        int n_p = clusters_to_samples[p].size();
        int n_i = clusters_to_samples[i].size();

        if (nearest_clusters[j].find(p)!=nearest_clusters[j].end()){    // in
            double distance_i_p = (nearest_clusters[i][p] + tightness_i + tightness[p]) * (n_i_old + n_p) - tightness_i * n_i_old - tightness[p] * n_p;
            double distance_j_p = (nearest_clusters[j][p] + tightness_j + tightness[p]) * (n_j_old + n_p) - tightness_j * n_j_old - tightness[p] * n_p;

            double distance_temp = (tightness[i]*n_i+tightness[p]*n_p+distance_i_p+distance_j_p) / (n_i+n_p) - tightness[i]-tightness[p];

            List_2k.insert({distance_temp,std::min(i,p),std::max(i,p)});


        }else{
            double distance_i_p = (nearest_clusters[i][p] + tightness_i + tightness[p]) * (n_i_old + n_p) - tightness_i * n_i_old - tightness[p] * n_p;
            double distance_temp = (tightness[i]*n_i+tightness[p]*n_p+distance_i_p+(n_j_old*n_p)*max_distance) / (n_i+n_p) - tightness[i]-tightness[p];
        
            List_2k.insert({distance_temp,std::min(i,p),std::max(i,p)});

        }
    }

    std::vector<std::tuple<int,int>> remove_list;
    for(auto& [p,distance] : nearest_clusters[j]){
        //std::cout<<j<<" "<<p<<"*****"<<std::endl;
        tree.erase({distance, std::min(j,p), std::max(j,p)});
        remove_list.push_back({j,p});
        int n_p = clusters_to_samples[p].size();
        int n_i = clusters_to_samples[i].size();
        if (nearest_clusters[i].find(p) == nearest_clusters[i].end()){    

            double distance_j_p = (nearest_clusters[j][p] + tightness_j + tightness[p]) * (n_j_old + n_p) - tightness_j * n_j_old - tightness[p] * n_p;
            double distance_temp = (tightness[i]*n_i+tightness[p]*n_p+distance_j_p+(n_i_old*n_p)*max_distance) / (n_i+n_p) - tightness[i]-tightness[p];
            
            List_2k.insert({distance_temp,std::min(i,p),std::max(i,p)});

        }
    }

    nearest_clusters[i].clear();
    auto it = List_2k.begin();
    for (int count = 0; count < knn && it != List_2k.end(); ++count, ++it) {    // 前k个保留
        auto triplet = *it;
        tree.insert(triplet);
        nearest_clusters[std::get<1>(triplet)][std::get<2>(triplet)] = std::get<0>(triplet);
        nearest_clusters[std::get<2>(triplet)][std::get<1>(triplet)] = std::get<0>(triplet);
    }
    for (; it != List_2k.end(); ++it) {     //后k个截断
        auto triplet = *it;
        int p = std::get<1>(triplet) + std::get<2>(triplet) - i;
        
        auto find_it = nearest_clusters[p].find(i);     //nearesr_clusters[p][i]
        if (find_it != nearest_clusters[p].end()) {
            nearest_clusters[p].erase(i);
        }
    }

    for (auto& point : remove_list) {
        int first = std::get<0>(point);
        int second = std::get<1>(point);
        nearest_clusters[second].erase(first);  // Remove (i, p)
        nearest_clusters[first].erase(second); // Remove (p, i)
    }


}