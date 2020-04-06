/*------------------------------------------------------------------------------------------#
 * Local scaling (no scale, nearest neighbors, time velocity)
 *------------------------------------------------------------------------------------------#
 */


#ifndef _uic_local_scaling_hpp_
#define _uic_local_scaling_hpp_

//* Header(s) */
#include <limits> // std::numeric_limits
#include <vector> // std::vector
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
        
        template <typename num_t>
        void compute_scale_neis (
            std::vector<num_t> *scale,
            const int n_dim,
            const nanoflann::KDTreeSingleIndexAdaptor<num_t> &kd_tree,
            const std::vector<int> &vidx,
            const std::vector<std::vector<num_t>> &x,
            const std::vector<std::pair<int, int>> &idx_time)
        {
            size_t n_vidx = vidx.size();
            std::vector<size_t> index;
            std::vector<num_t>  dist;
            
            bool has_zero = false;
            for (size_t i = 0; i < n_vidx; ++i)
            {
                size_t vi = vidx[i];
                size_t kn = kd_tree.knn_search(&x[vi][0], idx_time[vi], &index, &dist, n_dim + 1);
                num_t d = 0;
                for (auto x : dist) d += x;
                (*scale)[i] = d / num_t(kn);
                if (d == 0) has_zero = true;
            }
            
            if (has_zero)  //* replace 0 to the minimum value*/
            {
                num_t min = std::numeric_limits<num_t>::max();
                for (auto  x : *scale) if (x != 0 && x < min) min = x;
                for (auto &x : *scale) if (x == 0) x = min;
            }
        };
        
        template <typename num_t>
        void compute_scale_vels (
            std::vector<num_t> *scale,
            const int n_dim,
            const nanoflann::NormStruct<num_t> &norm,
            const std::vector<int> &vidx,
            const std::vector<std::vector<num_t>> &x,
            const std::vector<std::pair<int, int>> &idx_time)
        {
            size_t n_vidx = vidx.size();
            
            bool has_zero = false;
            for (size_t i = 0; i < n_vidx; ++i)
            {
                num_t sum = 0, n = 0;
                int vi = vidx[i];
                int si = idx_time[vi].second;
                if (i != 0 && idx_time[vidx[i - 1]].second == si)
                {
                    int vj = vidx[i - 1];
                    sum += dist::compute(x[vi], x[vj], n_dim, norm);
                    ++n;
                }
                if (i != n_vidx - 1 && idx_time[vidx[i + 1]].second == si)
                {
                    int vj = vidx[i + 1];
                    sum += dist::compute(x[vi], x[vj], n_dim, norm);
                    ++n;
                }
                (*scale)[i] = sum / n;
                if (sum == 0) has_zero = true;
            }
            
            if (has_zero)  //* replace 0 to the minimum value*/
            {
                num_t min = std::numeric_limits<num_t>::max();
                for (auto  x : *scale) if (x != 0 && x < min) min = x;
                for (auto &x : *scale) if (x == 0) x = min;
            }
        };
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
        std::vector<num_t>().swap(*scale_lib);
        std::vector<num_t>().swap(*scale_prd);
        (*scale_lib).resize(vidx_lib.size(), 1);
        (*scale_prd).resize(vidx_prd.size(), 1);
        if (scale_type == no_scale || n_dim == 0) return;
        
        if (scale_type == neighbor)
        {
            nanoflann::KDTreeSingleIndexAdaptor<num_t> kd_tree;
            kd_tree.initialize(n_dim, x_lib, idx_time_lib, exclusion_radius);
            kd_tree.set_norm(norm_type, true, p);
            kd_tree.build_index();
            dist::compute_scale_neis(scale_lib, n_dim, kd_tree, vidx_lib, x_lib, idx_time_lib);
            dist::compute_scale_neis(scale_prd, n_dim, kd_tree, vidx_prd, x_prd, idx_time_prd);
        }
        else if (scale_type == velocity)
        {
            nanoflann::NormStruct<num_t> norm;
            norm.set_norm(norm_type, true, p);
            dist::compute_scale_vels(scale_lib, n_dim, norm, vidx_lib, x_lib, idx_time_lib);
            dist::compute_scale_vels(scale_prd, n_dim, norm, vidx_prd, x_prd, idx_time_prd);
        }
    }
}

#endif

// End