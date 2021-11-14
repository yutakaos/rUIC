/*------------------------------------------------------------------------------------------#
 * Class UIC_METHOD
 *------------------------------------------------------------------------------------------#
 */

#ifndef _uic_method_hpp_
#define _uic_method_hpp_

//* Header(s
#include <functional> // std::function
#include <limits>     // std::numeric_limits
#include <vector>     // std::vector
#include "alglib/statistics.hpp"
#include "uic_base.hpp"
#include "local_scaling.hpp"


namespace UIC
{
    template <typename num_t>
    class UIC_METHOD : protected UIC_BASE <num_t>
    {
    protected:
        
        ModelSet<num_t> model_full;
        ModelSet<num_t> model_ref;
        
    private:
        
        int n_vlib, n_vprd;
        int exclusion_radius;
        num_t epsilon;
        nanoflann::NormStruct<num_t> norm;
        UIC::SCALE scaling;
        
        std::vector<int> vidx_lib;
        std::vector<int> vidx_prd;
        std::function<num_t (num_t, num_t)> calc_se;
        ModelSet<num_t> *model_p;
        const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
        
    public:
        
        UIC_METHOD ()
        {
            set_norm_params();
            set_estimator();
        }
        
        void set_norm_params (
            nanoflann::NORM norm_type = nanoflann::L2,
            UIC::SCALE scale_type = UIC::neighbor,
            num_t p = 0.5, int exclusion_radius = 0, num_t epsilon = -1)
        {
            this->scaling = scale_type;
            this->exclusion_radius = exclusion_radius;
            this->epsilon = epsilon;
            norm.set_norm(norm_type, true, p);
        }
        
        void set_estimator (bool naive = false)
        {
            if (naive)
                calc_se = [](num_t err, num_t   ) { return err * err; };
            else
                calc_se = [](num_t err, num_t w2) { return err * err / (1.0 + w2); };
        }
        
        void set_valid_indices ()
        {
            n_vlib = UIC::set_valid_indices(&vidx_lib, this->x_lib);
            n_vprd = UIC::set_valid_indices(&vidx_prd, this->x_prd);
            
            //* clear results */
            this->result.rmse  = qnan;
            this->result.rmse0 = qnan;
            this->result.uic   = qnan;
            this->result.pval  = qnan;
        }
        
        void make_dist_lib (bool full_model = true, int reduced_E = 0)
        {
            //* set pointers */
            model_p = full_model ? &model_full : &model_ref;
            
            if (full_model || this->E < reduced_E) reduced_E = 0;
            int ndim = this->x_lib[0].size() - reduced_E * this->ndim_x;
            
            //* local scaling */
            std::vector<num_t> scale_lib, scale_prd;
            local_scaling(
                ndim, scaling, &scale_lib, &scale_prd,
                this->x_lib,  this->x_prd, this->time_lib, this->time_prd,
                vidx_lib, vidx_prd, exclusion_radius, norm
            );
            
            //* compute distance matrix */
            clear_and_resize((*model_p).dist, n_vprd);
            for (int i = 0; i < n_vprd; ++i)
            {
                (*model_p).dist[i].resize(n_vlib);
                num_t d;
                for (int j = 0; j < n_vlib; ++j)
                {
                    d = dist::compute(
                        this->x_prd[vidx_prd[i]], this->x_lib[vidx_lib[j]], ndim, norm);
                    (*model_p).dist[i][j] = d / std::sqrt(scale_prd[i] * scale_lib[j]);
                }
            }
            (*model_p).PRED = false;
            (*model_p).sse  = qnan;
            (*model_p).E  = this->E - reduced_E;
            (*model_p).nn = (this->nn < 0) ? (*model_p).E + 1 : this->nn;
        }
        
        void set_neighbors (bool full_model = true)
        {
            //* set pointers */
            model_p = full_model ? &model_full : &model_ref;
            
            //* check and count valid library data */
            std::vector<int> j_lib;
            for (int i = 0; i < n_vlib; ++i)
            {
                if (!std::isnan(this->y_lib[vidx_lib[i]])) j_lib.push_back(i);
            }
            this->result.n_lib = j_lib.size();
            
            //* find neighbors */
            clear_and_resize((*model_p).neis, n_vprd);
            for (int i = 0; i < n_vprd; ++i)
            {
                int vi = vidx_prd[i];
                if (std::isnan(this->y_prd[vi])) continue;
                
                std::vector<int> indices;
                std::pair<int, int> idxi = this->time_lib[vi];
                for (auto j : j_lib)
                {
                    int vj = vidx_lib[j];
                    std::pair<int, int> idxj = this->time_lib[vj];
                    if (std::abs(idxj.first - idxi.first) <= exclusion_radius)
                    {
                        if (idxj.second == idxi.second) continue;
                    }
                    indices.push_back(j);
                }
                const std::vector<num_t> &dist = (*model_p).dist[i];
                UIC::find_neighbors(
                    &((*model_p).neis[i]), (*model_p).nn, dist, indices, epsilon);
            }
        }
        
        void primitive_simplex_map (bool full_model = true)
        {
            //* set pointers */
            model_p = full_model ? &model_full : &model_ref;
            
            size_t nt_prd = this->time_prd.size();
            clear_and_resize((*model_p).pred);
            clear_and_resize((*model_p).nenn);
            clear_and_resize((*model_p).se);
            (*model_p).pred.resize(nt_prd, qnan);
            (*model_p).nenn.resize(nt_prd, qnan);
            
            //* simplex map */
            num_t sum_se = 0, n_pred = 0;
            for (int i = 0; i < n_vprd; ++i)
            {
                const std::vector<num_t> &dist = (*model_p).dist[i];
                const std::vector<int>   &neis = (*model_p).neis[i];
                int nn = neis.size();
                if (nn == 0) continue;
                
                //* weights */
                std::vector<num_t> weight(nn);
                if (dist[neis[0]] == 0)
                {
                    for (int k = 0; k < nn; ++k)
                    {
                        weight[k] = (dist[neis[k]] == 0);
                    }
                }
                else
                {
                    for (int k = 0; k < nn; ++k)
                    {
                        weight[k] = std::exp(-dist[neis[k]] / dist[neis[0]]);
                    }
                }
                
                //* predictions */
                num_t x = 0, w2 = 0, sum_w = 0, se;
                for (int k = 0; k < nn; ++k)
                {
                    x  += weight[k] * this->y_lib[vidx_prd[neis[k]]];
                    w2 += weight[k] * weight[k];
                    sum_w += weight[k];
                }
                x  /= sum_w;
                w2 /= (sum_w * sum_w);
                se = calc_se(this->y_prd[vidx_prd[i]] - x, w2);
                sum_se += se;
                ++n_pred;
                (*model_p).pred[vidx_prd[i]] = x;
                (*model_p).nenn[vidx_prd[i]] = 1.0 / w2;
                (*model_p).se.push_back(se);
            }
            (*model_p).PRED = true;
            (*model_p).sse  = sum_se;
            if (full_model) this->result.n_pred = int(n_pred);
        }
        
        void compute_uic ()
        {
            if (!model_full.PRED || !model_ref.PRED) return;
            UIC::ResultSet<num_t> &RES = this->result;
            int n_pred = RES.n_pred;
            RES.E = model_full.E;
            RES.E0 = model_ref.E;
            RES.nn = model_full.nn;
            RES.nn0 = model_ref.nn;
            RES.rmse = std::sqrt(model_full.sse / num_t(n_pred));
            RES.rmse0 = std::sqrt(model_ref.sse / num_t(n_pred));
            RES.uic = log(RES.rmse0) - log(RES.rmse);
            
            //* compute p-value */
            alglib::real_1d_array x;
            x.setlength(n_pred);
            for (int i = 0; i < n_pred; ++i) x(i) = model_full.se[i] - model_ref.se[i];
            double delta, both, left, right;
            delta = (model_full.sse - model_ref.sse) * (1.0 - 1.0 / num_t(n_pred));
            alglib::wilcoxonsignedranktest(x, n_pred, delta, both, left, right);
            RES.pval = right;
        }
    };
}

#endif

// End