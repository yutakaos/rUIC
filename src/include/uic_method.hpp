/*------------------------------------------------------------------------------------------#
 * Class UIC_METHOD
 *------------------------------------------------------------------------------------------#
 */


#ifndef _uic_method_hpp_
#define _uic_method_hpp_

//* Header(s
#include <functional> // std::function
#include <limits>     // std::numeric_limits
#include <random>     // std::mt19937_64, std::uniform_real_distribution
#include <vector>     // std::vector
#include <uic_base.hpp>
#include <local_scaling.hpp>


namespace UIC
{
    template <typename num_t>
    class UIC_METHOD : protected UIC_BASE <num_t>
    {
    protected:
        
        std::vector<num_t> pred_full;
        std::vector<num_t> nenn_full;
        
    private:
        
        bool RMSE_F, RMSE_R;
        int n_vlib, n_vprd;
        const num_t qnan = std::numeric_limits<num_t>::quiet_NaN();
        
        nanoflann::NormStruct<num_t> norm;
        nanoflann::NORM norm_type;
        SCALE scale_type;
        int exclusion_radius;
        num_t p ,epsilon;
        
        std::vector<int> vidx_lib;
        std::vector<int> vidx_prd;
        std::vector<std::vector<num_t>> dist_full;
        std::vector<std::vector<num_t>> dist_reduced;
        std::vector<std::vector<int>> neis_full;
        std::vector<std::vector<int>> neis_reduced;
        std::vector<num_t> sqerr_full;
        std::vector<num_t> sqerr_reduced;
        std::vector<num_t> pred_reduced;
        std::vector<num_t> nenn_reduced;
        
        std::vector<std::vector<num_t>> *dist_p;
        std::vector<std::vector<int>> *neis_p;
        std::vector<num_t> *pred_p;
        std::vector<num_t> *nenn_p;
        std::vector<num_t> *sqerr_p;
        
        std::mt19937_64 mt;
        std::uniform_real_distribution<num_t> runif;
        std::function<num_t (num_t, num_t)> get_sqerr;
        
    public:
        
        UIC_METHOD ()
        {
            set_norm_params();
            set_rmse_estimator();
        }
        
        void set_norm_params (
            nanoflann::NORM norm_type = nanoflann::L2, SCALE scale_type = UIC::neighbor,
            num_t p = 0.5, int exclusion_radius = 0, num_t epsilon = -1)
        {
            this->norm_type = norm_type;
            this->scale_type = scale_type;
            this->p = p;
            this->exclusion_radius = exclusion_radius;
            this->epsilon = epsilon;
            norm.set_norm(norm_type, true, p);
        }
        
        void set_rmse_estimator (bool naive = false)
        {
            if (naive) get_sqerr = [](num_t err, num_t   ) { return err * err; };
            else       get_sqerr = [](num_t err, num_t w2) { return err * err / (1.0 + w2); };
        }
        
        void set_valid_indices ()
        {
            n_vlib = UIC::set_valid_indices(&vidx_lib, this->x_lib);
            n_vprd = UIC::set_valid_indices(&vidx_prd, this->x_prd);
        }
        
        void make_dist_lib (bool full_model = true)
        {
            //* compute local scaling parameters */
            int nd = this->x_lib[0].size() - (full_model ? 0 : this->n_xd);
            std::vector<num_t> scale_lib, scale_prd;
            local_scaling(
                nd, scale_type, &scale_lib, &scale_prd, this->x_lib,  this->x_prd,
                this->time_lib, this->time_prd, vidx_lib, vidx_prd, exclusion_radius, norm_type, p
            );
            
            //* set pointers */
            dist_p = full_model ? &dist_full : &dist_reduced;
            
            //* compute distance matrix */
            std::vector<std::vector<num_t>>().swap(*dist_p);
            (*dist_p).resize(n_vprd);
            for (int i = 0; i < n_vprd; ++i)
            {
                (*dist_p)[i].resize(n_vlib);
                num_t d;
                for (int j = 0; j < n_vlib; ++j)
                {
                    d = dist::compute(this->x_prd[vidx_prd[i]], this->x_lib[vidx_lib[j]], nd, norm);
                    (*dist_p)[i][j] = d / std::sqrt(scale_prd[i] * scale_lib[j]);
                }
            }
            RMSE_F = RMSE_R = false;
            this->result.rmseF = qnan;
            this->result.rmseR = qnan;
            this->result.uic   = qnan;
            this->result.pval  = qnan;
        }
        
        void set_neighbors (bool full_model = true)
        {
            //* check and count valid library data */
            std::vector<int> j_lib;
            for (int i = 0; i < n_vlib; ++i)
            {
                if (!std::isnan(this->y_lib[vidx_lib[i]])) j_lib.push_back(i);
            }
            this->result.n_lib = j_lib.size();
            
            //* set pointers */
            dist_p = full_model ? &dist_full : &dist_reduced;
            neis_p = full_model ? &neis_full : &neis_reduced;
            
            //* find neighbors */
            std::vector<std::vector<int>>().swap(*neis_p);
            (*neis_p).resize(n_vprd);
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
                UIC::find_neighbors(&(*neis_p)[i], this->nn, (*dist_p)[i], indices, epsilon);
            }
        }
        
        void primitive_simplex_map (bool full_model = true)
        {
            //* set pointers */
            dist_p = full_model ? &dist_full : &dist_reduced;
            neis_p = full_model ? &neis_full : &neis_reduced;
            pred_p = full_model ? &pred_full : &pred_reduced;
            nenn_p = full_model ? &nenn_full : &nenn_reduced;
            sqerr_p = full_model ? &sqerr_full : &sqerr_reduced;
            num_t *rmse_p = full_model ? &this->result.rmseF : &this->result.rmseR;
            
            //* predictions by simplex map */
            size_t nt_prd = this->time_prd.size();
            std::vector<num_t> weight;
            std::vector<num_t>().swap(*pred_p);
            std::vector<num_t>().swap(*nenn_p);
            std::vector<num_t>().swap(*sqerr_p);
            (*pred_p).resize(nt_prd, qnan);
            (*nenn_p).resize(nt_prd, qnan);
            num_t sqerr, sum_sqerr = 0, n_pred = 0;
            for (int i = 0; i < n_vprd; ++i)
            {
                int nn = (*neis_p)[i].size();
                if (nn == 0) continue;
                
                UIC::compute_weights(&weight, (*dist_p)[i], (*neis_p)[i]);
                num_t x = 0, w2 = 0;
                for (int k = 0; k < nn; ++k)
                {
                    x  += weight[k] * this->y_lib[vidx_prd[(*neis_p)[i][k]]];
                    w2 += weight[k] * weight[k];
                }
                sqerr = get_sqerr(this->y_prd[vidx_prd[i]] - x, w2);
                sum_sqerr += sqerr;
                ++n_pred;
                (*pred_p)[vidx_prd[i]] = x;
                (*nenn_p)[vidx_prd[i]] = 1.0 / w2;
                (*sqerr_p).push_back(sqerr);
            }
            (*rmse_p) = std::sqrt(sum_sqerr / n_pred);
            
            if (full_model)
            {
                RMSE_F = true;
                this->result.n_pred = int(n_pred);
            }
            else RMSE_R = true;
        }
        
        void compute_uic ()
        {
            if (!RMSE_F || !RMSE_R) return;
            this->result.uic = log(this->result.rmseR) - log(this->result.rmseF);
        }
        
        void bootstrap_pval (const std::vector<int> &seed)
        {
            if (!RMSE_F || !RMSE_R) return;
            int n_boot = seed.size();
            if (n_boot == 0) return;
            
            num_t n_prd = sqerr_full.size();
            num_t counter = 0;
            for (int r = 0; r < n_boot; ++r)
            {
                mt.seed(seed[r]);
                num_t sseF = 0, sseR = 0;
                for (num_t i = 0; i < n_prd; ++i)
                {
                    int k = int(runif(mt) * n_prd);
                    sseF += sqerr_full[k];
                    sseR += sqerr_reduced[k];
                }
                if (sseR <= sseF) ++counter;
            }
            this->result.pval = counter / num_t(n_boot);
        }
    };
}

#endif

// End