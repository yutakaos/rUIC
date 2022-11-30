/*****************************************************************************
 * Find Nearest Neighbors
 * 
 * Copyright 2022  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _find_neighbors_hpp_
#define _find_neighbors_hpp_

//*** Header(s) ***/
#include <vector> // std::vectpr
#include "dataset.hpp"
#include "bfnn.hpp"
#include "nanoflann.hpp"

namespace UIC
{

enum KNN_TYPE { KD, BF };

template <typename num_t>
inline void find_neighbors (
    const size_t n_dim,
    std::vector<std::vector<size_t>> &index,    
    std::vector<std::vector<num_t >> &dist,
    const UIC::DataSet<num_t> &X,
    size_t nn,
    const num_t p = 2,
    const int exclusion_radius = -1,
    const num_t epsilon = -1,
    const bool to_weight = false,
    const KNN_TYPE knn_type = KNN_TYPE::KD)
{
    const size_t n_query = X.prd_data.size();
    std::vector<std::vector<size_t>>().swap(index);
    std::vector<std::vector<num_t >>().swap(dist);
    index.resize(n_query);
    dist .resize(n_query);
    
    if (nn == 0) nn = n_dim + 1;
    if (knn_type == KNN_TYPE::KD) // KD-tree nearest neighbor search
    {
        nanoflann::DataSet<num_t> cloud;
        cloud.set_data(X.lib_data, X.lib_time, exclusion_radius);
        cloud.set_norm(p);
        nanoflann::KDTreeInterface<num_t> kdnn(n_dim, cloud, epsilon);
        for (size_t k = 0; k < n_query; ++k)
        {
            kdnn.search(&index[k], &dist[k], X.prd_data[k], X.prd_time[k], nn);
        }
    }
    else // brute-force nearest neighbor search (BFNN)
    {
        BFNN<num_t> bfnn(p, exclusion_radius, epsilon);
        bfnn.set_data(n_dim, X.lib_data, X.lib_time);
        for (size_t k = 0; k < n_query; ++k)
        {
            bfnn.search(&index[k], &dist[k], X.prd_data[k], X.prd_time[k], nn);
        }
    }
    if (to_weight) // convert distances to weights for xmap
    {
        for (auto &ds : dist)
        {
            if (ds[0] == 0.0)
                for (auto &d : ds) d = (d == 0.0 ? 1.0 : 0.0); 
            else
            {
                num_t mind = ds[0];
                for (auto &d : ds) d = std::exp(-d / mind);
            }
            num_t sumw = 0;
            for (auto  w : ds) sumw += w;
            for (auto &w : ds) w /= sumw;
        }
    }
}

}

#endif
//* End */