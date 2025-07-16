#include "flk.h"
#include "time.h"

flk::flk(){}

flk::flk(int c_true, bool debug){
    this->debug = debug;
    this->c_true = c_true;
    this->n = vector<double>(c_true, 0);
    this->sd = vector<double>(c_true, 0);
    this->diYr = vector<double>(c_true, 0);
    this->obj_knn = vector<double>(c_true, 0);
}

flk::flk(vector<vector<int>> &NN, vector<vector<double>> &NND, double max_d, int c_true, bool debug): flk(c_true, debug){
    this->NN = NN;
    this->NND = NND;
    this->c_indicator = vector<bool>(c_true, false);
    this->knn_c = vector<int>(c_true, 0);
    this->knn_cou = vector<int>(c_true, 0);

    if (debug){
        // check_NN();
    }
    if (max_d < 0){
        this->max_d = cf::maximum_2Dvec(NND);
    }
    else{
        this->max_d = max_d;
    }
    cf::symmetry(this->NN, this->NND, 1, 0.0);
    if (debug){
        cout << "max_d = " << max_d << endl;
    }

    this->local = true;
}


void flk::check_NN(){
    for (unsigned int i = 0; i < NN.size(); i++){
        if (NN[i][0] != i){
            cout << "Error opening file" << endl;
            exit(EXIT_FAILURE);
        }
    }
}

flk::~flk(){}

void flk::opt(unsigned int ITER, vector<vector<int>> &init_Y){

    if(debug){
        cout<<"opt start!"<<endl;
    }

    Y = vector<vector<int>>(init_Y.size(), vector<int>(init_Y[0].size(), 0));
    n_iter = vector<unsigned int>(init_Y.size(), 0);    // 算法可以一次性传入多个初始标签，聚类之后一次性返回

    unsigned int iter;
    for (unsigned int i = 0; i < init_Y.size(); i++){
        y = init_Y[i];
        iter = opt_once(ITER);
        Y[i] = y;
        n_iter[i] = iter;
    }
}


unsigned int flk::opt_once(unsigned int ITER){
    int tmp_c;

    if (debug){
        cout<<"init start!"<<endl;
    }
    // init
    init();

    int c_old, c_new, nb;
    bool converge;
    double obj;

    time_t tt;
    tm * t_cur;

    if (debug){
        cout << "Iter begin" << endl;
    }

    unsigned int iter = 0;
    for (unsigned iter = 0; iter < ITER; iter++){
        if (debug){
            tt = time(NULL);
            t_cur = localtime(&tt);
            cout << "Iter = " << iter << ", time = " << t_cur->tm_min << ": " << t_cur->tm_sec << "s" << endl;
        }

        converge = true;
        for (unsigned int i = 0; i < y.size(); i++){
            c_old = y[i];
            c_new = update_yi(i);

            if (c_new != c_old){
                y[i] = c_new;
                maintain_var(i, c_old, c_new);
                converge = false;
            }
        }

        if (debug){
            obj = 0;
            for (unsigned int k = 0; k < c_true; k++){
                obj += sd[k]/n[k];
            }
            cout << "obj = " << obj << endl;
        }

        if (converge){
            break;
        }
    }
    if (debug){
        cout << "Iter = " << iter << endl;
    }
    return iter;
}

int flk::update_yi(int sam_i){
    int c_old, c_new;
    c_old = y[sam_i];
    if (n[c_old] == 1){
        return c_old;
    }

    c_new = compute_diff(sam_i);
    // c_new = distance(diff.begin(), min_element(diff.begin(), diff.end()));
    return c_new;
}

void flk::init(){

    // init n
    fill(n.begin(), n.end(), 0);
    for (auto yi : y){
        n[yi] ++;
    }

    int tmp_c;
    // compute sd = diag(Y'GY)
    fill(sd.begin(), sd.end(), 0);
    for (unsigned int i = 0; i < y.size(); i++){
        tmp_c = y[i];
        sd[tmp_c] += sum_distance_within_cluster(i);
    }
}

void flk::maintain_var(int sam_i, int c_old, int c_new){
    n[c_old] --;
    n[c_new] ++;

    sd[c_old] -= 2 * diYr[c_old] - NND[sam_i][0];
    sd[c_new] += 2 * diYr[c_new] + NND[sam_i][0];
}


int flk::compute_diff(int i){
    // knn_c
    int knn_c_len, tmp_c;
    knn_c_len = 0;
    for (int nb: NN[i]){
        tmp_c = y[nb];
        if (!c_indicator[tmp_c]){
            c_indicator[tmp_c] = true;
            knn_c[knn_c_len] = tmp_c;
            knn_c_len ++;
        }
    }

    // b (bj = 0, j in knn_c)
    for (int k = 0; k < knn_c_len; k++){
        tmp_c = knn_c[k];
        diYr[tmp_c] = 0;
        knn_cou[tmp_c] = 0;
    }

    // diYr_j = sum_{xl in Aj}  G_il 
    int tmp_nb;
    for (unsigned int k = 0; k < NN[i].size(); k++){
        tmp_nb = NN[i][k];
        tmp_c = y[tmp_nb];
        knn_cou[tmp_c] ++;
        diYr[tmp_c] += NND[i][k];
    }
    for (int k = 0; k < knn_c_len; k++){
        tmp_c = knn_c[k];
        diYr[tmp_c] += (n[tmp_c] - knn_cou[tmp_c]) * max_d;
    }


    // obj_knn
    int c_old = y[i];
    int j;
    double bj;
    for (int k = 0; k < knn_c_len; k++){
        j = knn_c[k];
        if (j == c_old){
            bj = 2 * diYr[j] - NND[i][0];
            obj_knn[j] = sd[j] / n[j] - (sd[j] - bj) / (n[j] - 1);

        }else{
            bj = 2 * diYr[j] + NND[i][0];
            obj_knn[j] = (sd[j] + bj) / (n[j] + 1) - sd[j] / n[j];
        }
    }

    int min_ind = knn_c[0];
    double min_val = obj_knn[min_ind];
    for (int k = 0; k < knn_c_len; k++){
        j = knn_c[k];
        if (obj_knn[j] < min_val){
            min_ind = j;
            min_val = obj_knn[j];
        }
    }

    int c_new = min_ind;

    for (int i = 0; i < knn_c_len; i++){
        tmp_c = knn_c[i];
        c_indicator[tmp_c] = false;
    }
    return c_new;
}

// the sum of distance between xi and Cj (j = y[i])
double flk::sum_distance_within_cluster(unsigned int i){
    int c = y[i];

    double ret = 0;
    int tmp_nb, tmp_c;
    int count = 0;

    for (unsigned int k = 0; k < NN[i].size(); k++){
        tmp_nb = NN[i][k];
        tmp_c = y[tmp_nb];
        if (tmp_c == c){
            count ++;
            ret += NND[i][k];
        }
    }

    ret += (n[c] - count) * max_d;
    return ret;
}