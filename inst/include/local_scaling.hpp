/*------------------------------------------------------------------------------------------#
 * Local scaling (no scale, nearest neighbors, time velocity)
 *------------------------------------------------------------------------------------------#
 */

#ifndef _uic_local_scaling_hpp_
#define _uic_local_scaling_hpp_

//* Header(s) */
#include <limits> // std::numeric_limits
#include <vector> // std::vector
#include "nanoflann/nanoflann.hpp"


namespace UIC
{
    enum SCALE {no_scale, neighbor, velocity};
    
    namespace dist
    {
        template <typename num_t>
        num_t compute (
            const std::vector<num_t> &x,
            const std::vector<num_t> &y,
            const int n_dim,
            const nanoflann::NormStruct<num_t> &norm)
        {
            num_t d = 0;
            for (int k = 0; k < n_dim; ++k)
            {
                d = norm.sum(d, norm.pow(x[k], y[k]));
            }
            return norm.root(d);
        }
        
        template <typename num_t>
        void scaling_neighbor (
            std::vector<num_t> *scale,
            const int n_dim,
            const nanoflann::KDTreeSingleIndexAdaptor<num_t> &kd_tree,
            const std::vector<int> &vidx,
            const std::vector<std::vector<num_t>> &x,
            const std::vector<std::pair<int, int>> &idx_time)
        {
            (*scale).resize(vidx.size());
            std::vector<size_t> index;
            std::vector<num_t>  dist;
            bool has_zero = false;
            for (size_t i = 0; i < vidx.size(); ++i)
            {
                size_t vi = vidx[i];
                size_t kn = kd_tree.knn_search(&x[vi][0], idx_time[vi], &index, &dist, n_dim + 1);
                num_t d = 0;
                for (auto x : dist) d += x;
                (*scale)[i] = d / num_t(kn);
                if (d == 0) has_zero = true;
            }
            if (has_zero) // then, 0 is replaced to the minimum value
            {
                num_t min = std::numeric_limits<num_t>::max();
                for (auto  x : *scale) if (x != 0 && x < min) min = x;
                for (auto &x : *scale) if (x == 0) x = min;
            }
        }
        
        template <typename num_t>
        void scaling_velocity (
            std::vector<num_t> *scale,
            const int n_dim,
            const nanoflann::NormStruct<num_t> &norm,
            const std::vector<int> &vidx,
            const std::vector<std::vector<num_t>> &x,
            const std::vector<std::pair<int, int>> &idx_time)
        {
            (*scale).resize(vidx.size(), 0);
            num_t n = 0;
            bool has_zero = false;
            for (size_t i = 0; i < vidx.size() - 1; ++i)
            {
                int vi = vidx[i];
                int vj = vidx[i + 1];
                if (idx_time[vi].second == idx_time[vj].second)
                {
                    num_t d = dist::compute(x[vi], x[vj], n_dim, norm);
                    (*scale)[i] = ((*scale)[i] + d) / (++n);
                    (*scale)[i + 1] = d;
                    n = 1;
                }
                else n = 0;
                if ((*scale)[i] == 0) has_zero = true; 
            }
            if (has_zero) // then, 0 is replaced to the minimum value
            {
                num_t min = std::numeric_limits<num_t>::max();
                for (auto  x : *scale) if (x != 0 && x < min) min = x;
                for (auto &x : *scale) if (x == 0) x = min;
            }
        }
    }
    
    template <typename num_t>
    void local_scaling (
        const int n_dim,
        const SCALE scale_type,
        std::vector<num_t> *scale_lib,
        std::vector<num_t> *scale_prd,
        const std::vector<std::vector<num_t>> &x_lib,
        const std::vector<std::vector<num_t>> &x_prd,
        const std::vector<std::pair<int, int>> &idx_time_lib,
        const std::vector<std::pair<int, int>> &idx_time_prd,
        const std::vector<int> &vidx_lib,
        const std::vector<int> &vidx_prd,
        const int exclusion_radius,
        const nanoflann::NormStruct<num_t> &norm)
    {
        std::vector<num_t>().swap(*scale_lib);
        std::vector<num_t>().swap(*scale_prd);
        if (scale_type == UIC::no_scale || n_dim == 0)
        {
            (*scale_lib).resize(vidx_lib.size(), 1);
            (*scale_prd).resize(vidx_prd.size(), 1);
        }
        else if (scale_type == UIC::neighbor)
        {
            nanoflann::KDTreeSingleIndexAdaptor<num_t> kd_tree;
            kd_tree.initialize(n_dim, x_lib, idx_time_lib, exclusion_radius);
            kd_tree.set_norm(norm);
            dist::scaling_neighbor(scale_lib, n_dim, kd_tree, vidx_lib, x_lib, idx_time_lib);
            dist::scaling_neighbor(scale_prd, n_dim, kd_tree, vidx_prd, x_prd, idx_time_prd);
        }
        else if (scale_type == UIC::velocity)
        {
            dist::scaling_velocity(scale_lib, n_dim, norm, vidx_lib, x_lib, idx_time_lib);
            dist::scaling_velocity(scale_prd, n_dim, norm, vidx_prd, x_prd, idx_time_prd);
        }
    }
}

#endif

// End