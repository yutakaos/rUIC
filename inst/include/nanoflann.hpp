/***********************************************************************
 * Software License Agreement (BSD License)
 * 
 * Copyright 2022  Yutaka Osada (ytosada@gmail.com). All rights reserved.
 *     - Modification for UIC.
 * 
 * based on nanoflann (version:0x142)
 * 
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2022  Jose Luis Blanco (joseluisblancoc@gmail.com).
 *   All rights reserved.
 * 
 * THE BSD LICENSE
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

/** \mainpage nanoflann C++ API documentation
 *  nanoflann is a C++ header-only library for building KD-Trees, mostly
 *  optimized for 2D or 3D point clouds.
 *
 *  nanoflann does not require compiling or installing, just an
 *  #include <nanoflann.hpp> in your code.
 */

#ifndef _nanoflann_hpp_
#define _nanoflann_hpp_

//*** Header(s) ***/
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

/* Library version: 0xMmP (M=Major,m=minor,P=patch) */
#define NANOFLANN_VERSION 0x142

namespace nanoflann
{
    namespace memory
    {
        /**
         * Pooled storage allocator
         *
         * The following routines allow for the efficient allocation of storage
         * in small chunks from a specified pool. Rather than allowing each
         * structure to be freed individually, an entire pool of storage is
         * freed at once. This method has two advantages over just using
         * malloc() and free(). First, it is far more efficient for allocating
         * small objects, as there is no overhead for remembering all the
         * information needed to free each object or consolidating fragmented
         * memory. Second, the decision about how long to keep an object is made
         * at the time of allocation, and there is no need to track down all the
         * objects to free them.
         */
        
        /* size of machine word in bytes */
        const size_t WORDSIZE  = 16;  // Must be power of 2
        /* minimum number of bytes requested at a time from	the system */
        const size_t BLOCKSIZE = 8192; // Must be multiple of WORDSIZE
        
        /* We maintain memory alignment to word boundaries by requiring that
         * all allocations be in multiples of the machine wordsize. */
        class PooledAllocator
        {
            uint32_t remaining; // number of bytes left in current block
            void* base; // pointer to base of current block
            void* loc;  // current location in block to next allocate memory
            
            void internal_init()
            {
                remaining = 0;
                base = nullptr;
                usedMemory = 0;
                wastedMemory = 0;
            }
            
        public:
            uint32_t usedMemory;
            uint32_t wastedMemory;
            
             PooledAllocator() { internal_init(); }
            ~PooledAllocator() { free_all(); }
            
            /* frees all allocated memory chunks */
            void free_all()
            {
                while (base != nullptr)
                {
                    void* prev = *(static_cast<void**>(base));
                    ::free(base);
                    base = prev;
                }
                internal_init();
            }
            
            /* allocates (using this pool) a generic type */
            template <typename T> T* allocate(const size_t count = 1)
            {
                T* mem = static_cast<T*>(this->malloc(sizeof(T) * count));
                return mem;
            }
            
        private:
            /* returns a pointer to a piece of new memory */
            void* malloc(const size_t req_size)
            {
                /* round size up to a multiple of wordsize */
                const uint32_t size = (req_size + (WORDSIZE - 1)) & ~(WORDSIZE - 1);
                
                /* check whether a new block must be allocated */
                if (size > remaining)
                {
                    wastedMemory += remaining;
                    
                    /* allocate new storage. */
                    uint32_t blocksize = size + sizeof(void*) + (WORDSIZE - 1);
                    if (blocksize > BLOCKSIZE) blocksize = BLOCKSIZE;
                    
                    // use the standard C malloc to allocate memory
                    void* m = ::malloc(blocksize);
                    if (!m)
                    {
                        fprintf(stderr, "Failed to allocate memory.\n");
                        throw std::bad_alloc();
                    }
                    
                    /* fill first word of new block with pointer to previous block */
                    static_cast<void**>(m)[0] = base;
                    base = m;
                    
                    uint32_t shift = 0;
                    remaining = blocksize - sizeof(void*) - shift;
                    loc = (static_cast<char*>(m) + sizeof(void*) + shift);
                }
                void* rloc = loc;
                loc = static_cast<char*>(loc) + size;
                remaining  -= size;
                usedMemory += size;
                return rloc;
            }
        };
    }
    
    /*** Norm adaptor class ***/
    template <typename num_t>
    class NormAdaptor
    {
        num_t p;
        
    public:
        std::function<num_t(num_t, num_t)> sum;
        std::function<num_t(num_t, num_t)> pow;
        std::function<num_t(num_t)> root;
        std::function<num_t(num_t, num_t, num_t)> update_min;
        
        NormAdaptor () : p(2.0)
        {}
        
        inline void set_norm (num_t p = 2)
        {
            this->p = p;
            if      (p == 2.0) set_L2 ();
            else if (p == 1.0) set_L1 ();
            else if (p <= 0.0) set_Max();
            else set_Lp();
        }
        
    private:
        void set_L2 ()
        {
            sum  = [](num_t a, num_t b) { return a + b; };
            pow  = [](num_t a, num_t b) { return (a - b) * (a - b); };
            root = [](num_t a) { return std::sqrt(a); };
            update_min = [](num_t min_d, num_t cut_d, num_t d)
            { return min_d + cut_d - d; };
        }
        
        void set_L1 ()
        {
            sum  = [](num_t a, num_t b) { return a + b; };
            pow  = [](num_t a, num_t b) { return std::abs(a - b); };
            root = [](num_t a) { return a; };
            update_min = [](num_t min_d, num_t cut_d, num_t d)
            { return min_d + cut_d - d; };
        }
        
        void set_Max ()
        {
            sum  = [](num_t a, num_t b) { return a > b ? a : b; };
            pow  = [](num_t a, num_t b) { return std::abs(a - b); };
            root = [](num_t a) { return a; };
            update_min = [](num_t min_d, num_t cut_d, num_t)
            { return min_d < cut_d ? min_d : cut_d; };
        }
        
        void set_Lp ()
        {
            sum  = [    ](num_t a, num_t b) { return a + b; };
            pow  = [this](num_t a, num_t b)
            { return std::pow(std::abs(a - b), p); };
            root = [this](num_t a) { return std::pow(a, 1.0 / p); };
            update_min = [](num_t min_d, num_t cut_d, num_t d)
            { return min_d + cut_d - d; };
        }
    };
    
    /*** Data class (point cloud) ***/
    template <typename num_t>
    struct DataSet
    {
        using data_t = std::vector<num_t>;
        using time_t = std::pair<int,int>;
        
        NormAdaptor<num_t> norm;
        std::vector<data_t> data;
        std::vector<time_t> time;
        int exclusion_radius;
        
        inline void set_data (
            const std::vector<data_t> &data,
            const std::vector<time_t> &time,
            int exclusion_radius = -1)
        {
            std::vector<data_t>().swap(this->data);
            std::vector<time_t>().swap(this->time);
            this->data = data;
            this->time = time;
            this->exclusion_radius = exclusion_radius;
        }
        
        inline void set_norm (num_t p) { norm.set_norm(p); }
        
        /* for KD-TREE */
        inline bool time_exclusion (const size_t idx, const time_t &qtime) const
        {
            /* check whether library and query have the same group? */
            if (time[idx].second != qtime.second) return false;
            /* check whether library and query are near timestamp? */
            int radius = std::abs(time[idx].first - qtime.first);
            if (radius > exclusion_radius) return false;
            return true;
        }
        
        inline num_t eval_norm (
            const data_t &query, size_t idx, size_t size,
            num_t worst_dist = -1) const
        {
            num_t d = num_t();
            size_t dim = 0;
            while (dim + 3 < size)
            {
                d = norm.sum(d, norm.pow(query[dim], data[idx][dim]));
                ++dim;
                d = norm.sum(d, norm.pow(query[dim], data[idx][dim]));
                ++dim;
                d = norm.sum(d, norm.pow(query[dim], data[idx][dim]));
                ++dim;
                d = norm.sum(d, norm.pow(query[dim], data[idx][dim]));
                ++dim;
                if (worst_dist > 0 && d > worst_dist) return d;
            }
            while (dim < size)
            {
                d = norm.sum(d, norm.pow(query[dim], data[idx][dim]));
                ++dim;
            }
            return d;
        }
        
        inline size_t get_size () const { return data.size(); }
        inline size_t get_dim  () const { return data[0].size(); }
        inline num_t  get_pt (const size_t idx, const size_t dim) const
            { return data[idx][dim]; }
    };
    
    /*** Result class ***/
    template <typename num_t>
    class ResultSet
    {
        const num_t NUM_MAX = std::numeric_limits<num_t>::max();
        
        std::vector<size_t> *indices;
        std::vector<num_t>  *dists;
        size_t nn, nn_at_build;
        size_t count;
        bool tied;
        
    public:
        explicit inline ResultSet (const size_t nn, const bool tied = false)
        {
            count = 0;
            this->nn = nn_at_build = nn;
            this->tied = tied;
        }
        
        inline void set_out_container (
                std::vector<size_t> *indices,
                std::vector<num_t>  *dists)
        {
            count = 0;
            (*indices).resize(nn_at_build);
            (*dists)  .resize(nn_at_build);
            this->indices = indices;
            this->dists   = dists;
            if (nn_at_build != 0) (*dists)[nn_at_build - 1] = NUM_MAX;
        }
        
        /* for KD-TREE */
        inline bool add_point (num_t dist, size_t index)
        {
            if (count == nn)
            {
                if (tied && dist == (*dists)[nn_at_build - 1])
                {
                    ++nn;
                    (*indices).resize(nn);
                    (*dists)  .resize(nn);
                }
            }
            size_t i;
            for (i = count; i > 0; --i)
            {
                if (dist < (*dists)[i - 1])
                {
                    if (i < nn)
                    {
                        (*dists)  [i] = (*dists)  [i - 1];
                        (*indices)[i] = (*indices)[i - 1];
                    }
                }
                else break;
            }
            if (i < nn)
            {
                (*dists)  [i] = dist;
                (*indices)[i] = index;
            }
            if (count < nn) ++count;
            return true;
        }
        
        inline num_t worst_dist () const { return (*dists)[nn_at_build - 1]; }
        
        inline size_t size () const { return count; }
        inline bool   full () const { return count >= nn_at_build; }
    };
    
    /*** KD-TREE base class ***/
    template <class Derived, typename num_t>
    class KDTreeBase
    {
        const num_t EPS = static_cast<num_t>(0.00001);
        
    public:
        /* type definitions */
        struct Node
        {
            /* child nodes (both=nullptr mean its a leaf node) */
            Node *child1, *child2;
            union {
                /* leaf node: point indices */
                struct leaf    {size_t left, right; } lr;
                /* non-leaf node: dimension and values used for subdivision*/
                struct nonleaf {int divfeat; num_t divlow, divhigh; } sub;
            } node_type;
        };
        struct Interval { num_t low, high; };
        using BoundingBox = typename std::vector<Interval>;
        
        /* variables */
        Node *root_node;
        int dim; // data dimensionality
        size_t m_leaf_max_size;
        size_t m_size;                // number of data at searching
        size_t m_size_at_index_build; // number of data at index build
        std::vector<size_t> vAcc; // indices to data vectors
        
        memory::PooledAllocator pool; // pooled memory allocator
        BoundingBox root_bbox; // KD-tree used to find neighbours
        
        /* functions */
        num_t compute_initial_dists (
            const Derived &obj, const std::vector<num_t> &query,
            std::vector<num_t> &dists) const
        {
            //assert(query);
            auto norm = obj.data.norm;
            num_t distsq = num_t();
            for (int i = 0; i < obj.dim; ++i)
            {
                if (query[i] < obj.root_bbox[i].low)
                {
                    dists[i] = norm.pow(query[i], obj.root_bbox[i].low);
                    distsq   = norm.sum(distsq, dists[i]);
                }
                if (query[i] > obj.root_bbox[i].high)
                {
                    dists[i] = norm.pow(query[i], obj.root_bbox[i].high);
                    distsq   = norm.sum(distsq, dists[i]);
                }
            }
            return distsq;
        }
        
        /**
         * Create a tree node that subdivides the list of vectors from vind[first]
         * to vind[last]. The routine is called recursively on each sublist.
         */
        Node* divideTree(
            Derived& obj, const size_t L/*left*/, const size_t R/*right*/,
            BoundingBox& bbox)
        {
            Node* node = obj.pool.template allocate<Node>(); // allocate memory
            
            /* if too few exemplars remain, then make this a leaf node */
            if (obj.dim == 0 || R - L <= obj.m_leaf_max_size)
            {
                node->child1 = node->child2 = nullptr; // mark as leaf node
                node->node_type.lr.left  = L;
                node->node_type.lr.right = R;
                
                for (int i = 0; i < obj.dim; ++i) // compute bounding-box
                {
                    num_t val = obj.data.get_pt(obj.vAcc[L], i);
                    bbox[i].low = bbox[i].high = val;
                    for (size_t k = L + 1; k < R; ++k)
                    {
                        val = obj.data.get_pt(obj.vAcc[k], i);
                        if (bbox[i].low  > val) bbox[i].low  = val;
                        if (bbox[i].high < val) bbox[i].high = val;
                    }
                }
            }
            else
            {
                size_t idx; int cutfeat; num_t cutval;
                middle_split(obj, L, R - L, idx, cutfeat, cutval, bbox);
                node->node_type.sub.divfeat = cutfeat;
                
                BoundingBox L_bbox(bbox);
                BoundingBox R_bbox(bbox);
                L_bbox[cutfeat].high = cutval;
                R_bbox[cutfeat].low  = cutval;
                node->child1 = divideTree(obj, L, L + idx, L_bbox);
                node->child2 = divideTree(obj, L + idx, R, R_bbox);
                node->node_type.sub.divlow  = L_bbox[cutfeat].high;
                node->node_type.sub.divhigh = R_bbox[cutfeat].low;
                for (int i = 0; i < obj.dim; ++i)
                {
                    bbox[i].low  = std::min(L_bbox[i].low , R_bbox[i].low);
                    bbox[i].high = std::max(L_bbox[i].high, R_bbox[i].high);
                }
            }
            return node;
        }
        
    private:
        /* helper function to compute min and max elements */
        inline void compute_minmax(
            const Derived &obj, size_t ind, size_t count, int element,
            num_t &min_elem, num_t &max_elem)
        {
            min_elem = obj.data.get_pt(vAcc[ind], element);
            max_elem = min_elem;
            for (size_t i = 1; i < count; ++i)
            {
                num_t val = obj.data.get_pt(vAcc[ind + i], element);
                if (val < min_elem) min_elem = val;
                if (val > max_elem) max_elem = val;
            }
        }
        
        /* helper function to get value in [min, max] */
        inline num_t get_val_in_range (num_t val, num_t min, num_t max)
        {
            if (val < min) return min;
            if (max < val) return max;
            return val;
        }
        
        void middle_split(
            Derived &obj, size_t ind, size_t count, size_t &index,
            int &cutfeat, num_t &cutval, const BoundingBox &bbox)
        {
            num_t max_span = bbox[0].high - bbox[0].low;
            for (int i = 1; i < obj.dim; ++i)
            {
                num_t span = bbox[i].high - bbox[i].low;
                if (span > max_span) max_span = span;
            }
            
            num_t min_elem, max_elem;
            num_t max_spread = -1;
            cutfeat = 0;
            for (int i = 0; i < obj.dim; ++i)
            {
                num_t span = bbox[i].high - bbox[i].low;
                if (span > (1 - EPS) * max_span)
                {
                    compute_minmax(obj, ind, count, i, min_elem, max_elem);
                    num_t spread = max_elem - min_elem;
                    if (spread > max_spread)
                    {
                        cutfeat = i;
                        max_spread = spread;
                    }
                }
            }
            
            /* split in the middle */
            num_t split_val = (bbox[cutfeat].low + bbox[cutfeat].high) / 2;
            compute_minmax(obj, ind, count, cutfeat, min_elem, max_elem);
            cutval = get_val_in_range(split_val, min_elem, max_elem);
            
            size_t lim1, lim2;
            plane_split(obj, ind, count, cutfeat, cutval, lim1, lim2);
            index = get_val_in_range(count/2, lim1, lim2);
        }
        
        /**
         *  Subdivide the list of points by a plane perpendicular on axe
         * corresponding to the 'cutfeat' dimension at 'cutval' position.
         *
         *  On return:
         *    data[ind[   0, ..., lim1-1]][cutfeat] <  cutval
         *    data[ind[lim1, ..., lim2-1]][cutfeat] == cutval
         *    data[ind[lim2, ..., count ]][cutfeat] >  cutval
         */
        void plane_split(
            Derived& obj, size_t ind, const size_t count, int cutfeat,
            num_t& cutval, size_t& lim1, size_t& lim2)
        {
            std::function<num_t(size_t)> get_pt = [&obj, &cutfeat]
                (size_t i) { return obj.data.get_pt(i, cutfeat); };
            
            /* move vector indices for left subtree to front of list. */
            size_t L = 0;         // left
            size_t R = count - 1; // right
            for (;;)
            {
                while (     L <= R && get_pt(vAcc[ind+L]) <  cutval) ++L;
                while (R && L <= R && get_pt(vAcc[ind+R]) >= cutval) --R;
                if (L > R || !R) break;
                std::swap(vAcc[ind+L], vAcc[ind+R]);
                ++L; --R;
            }
            /* split in the middle to maintain a balanced tree */
            lim1 = L;
            R = count - 1;
            for (;;)
            {
                while (     L <= R && get_pt(vAcc[ind+L]) <= cutval) ++L;
                while (R && L <= R && get_pt(vAcc[ind+R]) >  cutval) --R;
                if (L > R || !R) break;
                std::swap(vAcc[ind+L], vAcc[ind+R]);
                ++L; --R;
            }
            lim2 = L;
        }
    };
    
    /*** KD-TREE interface class (based on KDTreeSingleIndexAdaptor) ***/
    template <typename num_t>
    class KDTreeInterface : public KDTreeBase<KDTreeInterface<num_t>, num_t>
    {
        using data_t = std::vector<num_t>;
        using time_t = std::pair<int,int>;
        
        num_t epsError;
        num_t exclusion_dist;
        
        using Derived = nanoflann::KDTreeInterface<num_t>;
        using Base = typename nanoflann::KDTreeBase<Derived, num_t>;
        using Node = typename Base::Node;
        using BoundingBox = typename Base::BoundingBox;
        
    public:
        const nanoflann::DataSet<num_t> &data;
        
        explicit KDTreeInterface(const KDTreeInterface<num_t>&) = delete;
        explicit KDTreeInterface(
            const int n_dim, const DataSet<num_t> &inputData,
            const num_t epsilon = -1, const num_t eps = 0.0)
            : data(inputData)
        {
            Base::pool.free_all();
            exclusion_dist = epsilon < 0 ? -1 : data.norm.pow(epsilon, 0);
            
            //* search parameters */
            Base::m_leaf_max_size = 10; // default
            epsError = 1.0 + eps;
            
            //* initialization */
            const size_t num_data = data.get_size();
            Base::dim = n_dim < 0 ? data.get_dim() : n_dim;
            Base::root_node = nullptr;
            Base::m_size = Base::m_size_at_index_build = num_data;
            Base::vAcc.resize(Base::m_size);
            for (size_t i = 0; i < Base::m_size; i++) Base::vAcc[i] = i;
            if (Base::m_size == 0) return;
            
            //* compute BoudingBox */
            BoundingBox &bbox = Base::root_bbox;
            bbox.resize(Base::dim);
            for (int i = 0; i < Base::dim; ++i)
            {
                bbox[i].low = bbox[i].high = data.get_pt(0, i);
                for (size_t k = 1; k < num_data; ++k)
                {
                    num_t val = data.get_pt(k, i);
                    if (val < bbox[i].low ) bbox[i].low  = val;
                    if (val > bbox[i].high) bbox[i].high = val;
                }
            }
            
            //* devide tree */
            Base::root_node = this->divideTree(*this, 0, Base::m_size, bbox);
        }
        
        size_t search (
            std::vector<size_t> *index, // for neighbor indices
            std::vector<num_t>  *dist,  // for neighbor distances
            const data_t &query,
            const time_t &qtime,
            const size_t num_nearest,
            const bool tied = true) const
        {
            /* set result container */
            size_t nn = Base::m_size < num_nearest ? Base::m_size : num_nearest;
            nanoflann::ResultSet<num_t> result(nn, tied);
            result.set_out_container(index, dist);
            
            /* find neighbors */
            this->find_neighbors(result, query, qtime);
            for (auto &x : *dist) x = data.norm.root(x);
            return result.size();
        }
        
    private:
        /**
         * Find set of nearest neighbors to vec[0:dim-1]. Their indices are
         * stored inside the result object.
         */
        template <typename RESULTSET>
        bool find_neighbors (
            RESULTSET& result, const data_t &query, const time_t &qtime) const
        {
            //assert(query);
            if (Base::m_size == 0) return false;
            //if (this->size(*this) == 0) return false;
            //if (!Base::root_node) throw std::runtime_error("[nanoflann]");
            std::vector<num_t> dists(Base::dim, 0);
            num_t dist = this->compute_initial_dists(*this, query, dists);
            search_level(result, query, qtime, Base::root_node, dist, dists);
            return result.full();
        }
        
        /**
         * Performs an exact search in the tree starting from a node.
         */
        template <class RESULTSET>
        bool search_level(
            RESULTSET& result, const data_t &query, const time_t &qtime,
            const Node *node, num_t mindist, std::vector<num_t>& dists) const
        {
            /* leaf nodes */
            if (node->child1 == nullptr && node->child2 == nullptr)
            {
                num_t worst_dist = result.worst_dist();
                auto lr = node->node_type.lr;
                for (size_t i = lr.left; i < lr.right; ++i)
                {
                    const size_t index = Base::vAcc[i];
                    if (data.time_exclusion(index, qtime)) continue;
                    num_t dist = data.eval_norm(query, index, Base::dim);
                    if (dist <= exclusion_dist) continue;
                    if (dist <= worst_dist)
                    {
                        if (!result.add_point(dist, index)) return false;
                        /* false = done searching! */
                    }
                }
                return true;
            }
            
            /* non-leaf nodes */
            /* which child branch should be taken first? */
            int idx = node->node_type.sub.divfeat;
            num_t val = query[idx];
            num_t diff1 = val - node->node_type.sub.divlow;
            num_t diff2 = val - node->node_type.sub.divhigh;
            
            Node *child1, *child2; // best and non-best childs
            num_t cut_dist;
            if (diff1 + diff2 < 0)
            {
                child1 = node->child1;
                child2 = node->child2;
                cut_dist = data.norm.pow(val, node->node_type.sub.divhigh);
            }
            else
            {
                child1 = node->child2;
                child2 = node->child1;
                cut_dist = data.norm.pow(val, node->node_type.sub.divlow);
            }
            
            /* call recursively to search next level down */
            bool next = search_level(result, query, qtime, child1, mindist, dists);
            if (!next) return false;
            
            num_t dist = dists[idx];
            mindist = data.norm.update_min(mindist, cut_dist, dist);
            dists[idx] = cut_dist;
            if (mindist * epsError <= result.worst_dist())
            {
                next = search_level(result, query, qtime, child2, mindist, dists);
                if (!next) return false;
            }
            dists[idx] = dist;
            return true;
        }
    };
}

#endif
//* End */