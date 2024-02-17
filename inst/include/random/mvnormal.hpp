/*****************************************************************************
 * Multivariate normal distribution (Eigen)
 * 
 * Copyright 2023-2024  Yutaka Osada. All rights reserved.
 * 
 *****************************************************************************/

#ifndef _random_mvnormal_hpp_
#define _random_mvnormal_hpp_

#include <random> // std::normal_distribution
//#include <Eigen/Cholesky>
// Eigen::Matrix, Eigen::Dynamic, Eigen::LLT


namespace UIC {
namespace Rand_Eigen {

template <typename num_t = double>
class mvnormal
{
    using EigenVector = Eigen::Matrix<num_t, Eigen::Dynamic, 1>;
    using EigenMatrix = Eigen::Matrix<num_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    /* parameters */
    size_t dim;     // dimension
    EigenVector mu; // mean
    EigenMatrix A;  // L matrix from the Cholesky decomposition of variance
    
    /* standard normal RNG */
    std::normal_distribution<num_t> rand;
    
public:
    
    inline mvnormal (const EigenVector &mean, const EigenMatrix &var)
    {
        dim = mean.size();
        mu  = mean;
        A   = Eigen::LLT<EigenMatrix>(var).matrixL();
    }
    
    template <class Engine_t>
    inline EigenVector operator() (Engine_t &engine)
    {
        EigenVector z(dim);
        for (size_t k = 0; k < dim; ++k) z(k) = rand(engine);
        return mu + A * z;
    }
};

} //namespace Rand_Eigen
} //namespace UIC

#endif
//* End */