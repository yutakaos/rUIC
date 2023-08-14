/*****************************************************************************
 * Find Nearest Neighbors
 * 
 * Copyright 2022-2023  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _find_neighbors_hpp_
#define _find_neighbors_hpp_

#include <vector> // std::vector

#include "./method/bfnn.hpp"
#include "./method/nanoflann.hpp"


namespace UIC {

/*---------------------------------------------------------------------------*
 | Method to search nearest neighbors
 |    - KD = KD-tree (nanoflann)
 |    - BF = Brute Force 
 *---------------------------------------------------------------------------*/
enum KNN_TYPE { KD, BF };


/*---------------------------------------------------------------------------*
 | Find neighbors 
 |---------------------------------------------------------------------------*
 |    dim   : data dimension to compute distances (<= data.cols)
 |    index : k-NN indices
 |    dist  : k-NN distances
 |    to_wieghts : whether simplex weights are calculated
 |---------------------------------------------------------------------------*
 |    DataX : dataset, which has the following member variables:
 |      - DataX.trn_data : training data
 |      - DataX.val_data : validation data
 |      - DataX.trn_time : time indices of training data
 |      - DataX.val_time : time indices of validation data
 *---------------------------------------------------------------------------*/
template <typename num_t, typename DataSet_t>
inline void find_neighbors (
    const size_t dim,
    std::vector<std::vector<size_t>> &index,    
    std::vector<std::vector<num_t >> &dist,
    const DataSet_t &DataX,
    size_t nn,
    const num_t p = 2,
    const int exclusion_radius = -1,
    const num_t epsilon = -1,
    const bool to_weight = false,
    const KNN_TYPE knn_type = KNN_TYPE::KD)
{
    const size_t n_query = DataX.val_data.size();
    std::vector<std::vector<size_t>>().swap(index);
    std::vector<std::vector<num_t >>().swap(dist);
    index.resize(n_query);
    dist .resize(n_query);
    
    if (nn == 0) nn = dim + 1;
    if (knn_type == KNN_TYPE::KD) // KD-tree nearest neighbor search
    {
        nanoflann::DataSet<num_t> cloud;
        cloud.set_data(DataX.trn_data, DataX.trn_time, exclusion_radius);
        cloud.set_norm(p);
        nanoflann::KDTreeInterface<num_t> kdnn(dim, cloud, epsilon);
        for (size_t k = 0; k < n_query; ++k)
        {
            kdnn.search(
                &index[k], &dist[k], DataX.val_data[k], DataX.val_time[k], nn);
        }
    }
    else // brute-force nearest neighbor search
    {
        bfnn<num_t> bfnn(p, exclusion_radius, epsilon);
        bfnn.set_data(dim, DataX.trn_data, DataX.trn_time);
        for (size_t k = 0; k < n_query; ++k)
        {
            bfnn.search(
                &index[k], &dist[k], DataX.val_data[k], DataX.val_time[k], nn);
        }
    }
    
    /* convert distances to weights for simplex projection */
    if (to_weight)
    {
        for (auto &ds : dist)
        {
            if (ds[0] == 0.0)
                for (auto &d : ds) d = (d == 0.0 ? 1.0 : 0.0); 
            else {
                num_t mind = ds[0];
                for (auto &d : ds) d = std::exp(-d / mind);
            }
            num_t sumw = 0;
            for (auto  w : ds) sumw += w;
            for (auto &w : ds) w /= sumw;
        }
    }
}

} //namespace UIC

#endif
//* End */