/*------------------------------------------------------------------------------------------#
 * Local scaling (no scale, nearest neighbors, time velocity)
 *------------------------------------------------------------------------------------------#
 */


#ifndef _uic_local_scaling_hpp_
#define _uic_local_scaling_hpp_

//* Header(s) */
#include <vector>
#include <functional>
#include <nanoflann_uic.hpp>


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
        const nanoflann::NORM norm_type,
        const num_t p = 0.5)
    {
        size_t n_vlib = vidx_lib.size();
        size_t n_vprd = vidx_prd.size();
        std::vector<num_t>().swap(*scale_lib);
        std::vector<num_t>().swap(*scale_prd);
        (*scale_lib).resize(n_vlib, 1);
        (*scale_prd).resize(n_vprd, 1);
        if (n_dim == 0) return;
        
        if (scale_type == neighbor)
        {
            nanoflann::KDTreeSingleIndexAdaptor<num_t> kd_tree;
            kd_tree.initialize(n_dim, x_lib, idx_time_lib, exclusion_radius);
            kd_tree.set_norm(norm_type, true, p);
            kd_tree.build_index();
            
            std::vector<size_t> index;
            std::vector<num_t>  dist;
            for (size_t i = 0; i < n_vlib; ++i)
            {
                size_t vi = vidx_lib[i];
                size_t kn = kd_tree.knn_search(&x_lib[vi][0], idx_time_lib[vi], &index, &dist, n_dim + 1);
                num_t d = 0;
                for (auto x : dist) d += x;
                (*scale_lib)[i] = d / num_t(kn);
            }
            for (size_t i = 0; i < n_vprd; ++i)
            {
                size_t vi = vidx_prd[i];
                size_t kn = kd_tree.knn_search(&x_prd[vi][0], idx_time_prd[vi], &index, &dist, n_dim + 1);
                num_t d = 0;
                for (auto x : dist) d += x;
                (*scale_prd)[i] = d / num_t(kn);
            }            
        }
        else if (scale_type == velocity)
        {
            nanoflann::NormStruct<num_t> norm;
            norm.set_norm(norm_type, true, p);
            
            for (size_t i = 0; i < n_vlib; ++i)
            {
                num_t sum = 0, n = 0;
                int vi = vidx_lib[i];
                int si = idx_time_lib[vi].second;
                if (i != 0 && idx_time_lib[vidx_lib[i - 1]].second == si)
                {
                    int vj = vidx_lib[i - 1];
                    sum += dist::compute(x_lib[vi], x_lib[vj], n_dim, norm);
                    ++n;
                }
                if (i != n_vlib - 1 && idx_time_lib[vidx_lib[i + 1]].second == si)
                {
                    int vj = vidx_lib[i + 1];
                    sum += dist::compute(x_lib[vi], x_lib[vj], n_dim, norm);
                    ++n;
                }
                (*scale_lib)[i] = sum / n;
            }
            for (size_t i = 0; i < n_vprd; ++i)
            {
                num_t sum = 0, n = 0;
                int vi = vidx_prd[i];
                int si = idx_time_prd[vi].second;
                if (i != 0 && idx_time_prd[vidx_prd[i - 1]].second == si)
                {
                    int vj = vidx_lib[i - 1];
                    sum += dist::compute(x_prd[vi], x_prd[vj], n_dim, norm);
                    ++n;
                }
                if (i != n_vprd - 1 && idx_time_prd[vidx_prd[i + 1]].second == si)
                {
                    int vj = vidx_lib[i + 1];
                    sum += dist::compute(x_prd[vi], x_prd[vj], n_dim, norm);
                    ++n;
                }
                (*scale_prd)[i] = sum / n;
            }
        }
    }
}

#endif

// End