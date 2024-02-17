/*****************************************************************************
 * Dataset class
 * 
 * Copyright 2022-2024  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _dataset_hpp_
#define _dataset_hpp_

#include <limits> // std::numeric_limits
#include <vector> // std::vector


namespace UIC {
/*---------------------------------------------------------------------------*
 | Method to search nearest neighbors
 |    - KD = KD-tree (nanoflann)
 |    - BF = Brute Force 
 *---------------------------------------------------------------------------*/
enum KNN_TYPE { KD, BF };


/*---------------------------------------------------------------------------*
 | Dataset class
 *---------------------------------------------------------------------------*/
template <typename num_t>
struct DataSet
{
    const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
    
    const std::vector<std::vector<num_t>> &x, &y, &z;
    const std::vector<int> &group;
    const int E, tp, nn, exclusion_radius;
    const num_t p, epsilon;
    const UIC::KNN_TYPE knn_type; // NN search method
    
    int nx, ny, nz, dim_x, dim_y;
    int num_data, num_trn, num_val;
    std::vector<int> dtx;
    std::vector<int> idx_trn;
    std::vector<int> idx_val;
    std::vector<int> idx_all;
    std::vector<std::vector<num_t>> Y_trn;
    std::vector<std::vector<num_t>> Y_val;
    
public:
    
    DataSet ()
    {}
    
    DataSet (
        const std::vector<std::vector<num_t>> &x, // library
        const std::vector<std::vector<num_t>> &y, // target
        const std::vector<std::vector<num_t>> &z, // condition
        const std::vector<int > &group,
        const std::vector<bool> &trn,
        const std::vector<bool> &val,
        const int E,
        const int tau,
        const int tp,
        const int nn = 0,
        const num_t p = 2,
        const int exclusion_radius = -1,
        const num_t epsilon = -1,
        const UIC::KNN_TYPE knn_type = UIC::KNN_TYPE::KD,
        const bool uic_type = false)
        : x(x), y(y), z(z), group(group), E(E), tp(tp), nn(nn),
          exclusion_radius(exclusion_radius < 0 ? 0 : exclusion_radius),
          p(p), epsilon(epsilon), knn_type(knn_type)
    {
        num_data = group.size();
        //if (int(x.size()) != num_data) std::cerr << "length(group) must be nrow(x)." << std::endl;
        //if (int(y.size()) != num_data) std::cerr << "length(group) must be nrow(y)." << std::endl;
        //if (int(z.size()) != num_data) std::cerr << "length(group) must be nrow(z)." << std::endl;
        nx = x[0].size();
        ny = y[0].size();
        nz = z[0].size();
        dim_x = nz + E*nx;
        dim_y = ny;
        
        /* number of delayed times on X */
        int Tau = (tau < 1) ? 1 : tau;
        dtx.resize(E);
        if (uic_type)
            for (int i = 0; i < E; ++i) dtx[i] = (E - 1 - i)*Tau;
        else
            for (int i = 0; i < E; ++i) dtx[i] = i*Tau;
        
        /* check valid data indicies */
        for (int t = 0; t < num_data; ++t)
        {
            /* check time indicies */
            int ty = t + tp;
            if (ty < 0 || num_data <= ty) continue;
            int tx = t - (E - 1)*Tau;
            if (tx < 0 || num_data <= tx) continue;
            
            /* check group indicies */
            int Gy = group[ty];
            if (std::isnan(Gy)) continue;
            if (nz != 0 && group[t] != Gy) continue;
            bool sameG = true;
            for (auto dt : dtx) if (group[t-dt] != Gy) { sameG = false; continue; }
            if (!sameG) continue;
            
            /* check data completeness (no NAs) */
            if (!complete_case(X(t))) continue;
            if (!complete_case(Y(t))) continue;
            
            if (trn[t]) idx_trn.push_back(t);
            if (val[t]) idx_val.push_back(t);
            if (trn[t] || val[t]) idx_all.push_back(t);
        }
        num_trn = idx_trn.size();
        num_val = idx_val.size();
        
        /* make datasets for Y */
        for (auto t : idx_trn) Y_trn.push_back(Y(t));
        for (auto t : idx_val) Y_val.push_back(Y(t));
    }
    
    inline std::vector<num_t> X (int t, int E = -1) const
    {
        if (E < 0) E = this->E;
        int dim = nz + E*nx;
        std::vector<num_t> Xt(dim);
        for (int i = 0; i < nz; ++i) Xt[i] = z[t][i];
        for (int k = 0; k < E; ++k)
        {
            int t0 = t - dtx[k];
            int i0 = nz + k*nx;
            for (int i = 0; i < nx; ++i) Xt[i0+i] = x[t0][i];
        }
        return Xt;
    }
    
    inline std::vector<num_t> Y (int t) const
    {
        return y[t+tp];
    }
    
private:
    
    inline bool complete_case (std::vector<num_t> x) 
    { 
        for (auto xi : x) if (std::isnan(xi)) return false;
        return true; // check NAs
    }
};

} //namespace UIC

#endif
//* End */