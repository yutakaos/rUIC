/***********************************************************************
 * Software License Agreement (BSD License)
 * 
 * Copyright 2020  Yutaka Osada. All rights reserved.
 *     - Modification for rUIC.
 * 
 * based on nanoflann (version:0x132)
 * 
 * 
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 * Copyright 2011-2016  Jose Luis Blanco (joseluisblancoc@gmail.com).
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
 *  nanoflann_Lp does not require compiling or installing, just an
 *  #include <nanoflann_Lp.hpp> in your code.
 */

#ifndef _ruic_NANOFLANN_LP_HPP_
#define _ruic_NANOFLANN_LP_HPP_

//* Header(s) */
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

/** Library version: 0xMmP (M=Major,m=minor,P=patch) */
#define NANOFLANN_VERSION 0x132


namespace nanoflann
{
    namespace utils
    {
        template <typename num_t>
        union node_union
        {
            // [leaf] left, right: indices of points in leaf node
            // [nonleaf] divfeat: dimension used for subdivision
            // [nonleaf] divlow, divhigh: the values used for subdivision
            struct leaf { size_t left, right; } lr;
            struct nonleaf { int divfeat; num_t divlow, divhigh; } sub;
        };
        
        template <typename num_t>
        struct Node
        {
            node_union <num_t> node_type;
            Node *child1, *child2;
            // Child nodes (both = NULL mean it's a leaf node)
        };
        
        template <typename num_t>
        struct Interval
        {
            num_t low, high;
        };
        
        template <typename num_t>
        num_t get_val_within_range (num_t val, num_t min, num_t max)
        {
            if (val < min) return min;
            if (max < val) return max;
            return val;
        }
    }
    
    namespace memory
    {
        //* Class: Pooled storage allocator */
        class PooledAllocator
        {
            size_t remaining; // Number of bytes left in current block of storage
            void *base;       // Pointer to base of current block of storage
            void *loc;        // Current location in block to next allocate memory
            const size_t WORDSIZE  = 16;
            const size_t BLOCKSIZE = 8192;
            
        public:
            
            size_t usedMemory;
            size_t wastedMemory;
            
             PooledAllocator () { internal_init(); }
            ~PooledAllocator () { free_all(); }
            
            //* Frees all allocated memory chunks */
            void free_all ()
            {
                while (base != NULL)
                {
                    void *prev = *(static_cast<void **>(base)); // Get pointer to prev block.
                    ::free(base);
                    base = prev;
                }
                internal_init();
            }
            
            //** Allocates (using this pool) a generic type T */
            template <typename T> T *allocate (const size_t count = 1)
            {
                T *mem = static_cast<T *>(this->malloc(sizeof(T) * count));
                return mem;
            }
            
        private:
            
            void internal_init ()
            {
                remaining = 0;
                base = NULL;
                usedMemory = 0;
                wastedMemory = 0;
            }
            
            //* Returns a pointer to a piece of new memory */
            void *malloc (const size_t req_size)
            {
                const size_t size = (req_size + (WORDSIZE - 1)) & ~(WORDSIZE - 1);
                
                if (size > remaining)
                {
                    wastedMemory += remaining;
                    
                    //* Allocate new storage. */
                    size_t blocksize = size + sizeof(void *) + (WORDSIZE - 1);
                    if (blocksize > BLOCKSIZE) blocksize = BLOCKSIZE;
                    
                    //* use the standard C malloc to allocate memory */
                    void *m = ::malloc(blocksize);
                    if (!m)
                    {
                        fprintf(stderr, "Failed to allocate memory.\n");
                        return NULL;
                    }
                    
                    //* Fill first word of new block with pointer to previous block */
                    static_cast<void **>(m)[0] = base;
                    base = m;
                    
                    size_t shift = 0;
                    remaining = blocksize - sizeof(void *) - shift;
                    loc = (static_cast<char *>(m) + sizeof(void *) + shift);
                }
                void *rloc = loc;
                loc = static_cast<char *>(loc) + size;
                remaining  -= size;
                usedMemory += size;
                return rloc;
            }
        };
    }
    
    //* Norm methods */
    enum NORM { L2, L1, Max, Lp };
    
    template <typename num_t>
    struct NormStruct
    {
    private:
        
        num_t p;
        
    public:
        
        std::function<num_t(num_t, num_t)> sum;
        std::function<num_t(num_t, num_t)> pow;
        std::function<num_t(num_t)> root;
        std::function<num_t(num_t, num_t, num_t)> update_min_dist;
        
        void set_norm (NORM norm, bool do_root = true, num_t p = 0.5)
        {
            if (norm == L2)  set_L2();
            if (norm == L1)  set_L1();
            if (norm == Max) set_Max();
            if (norm == Lp)  set_Lp(p);
            if (!do_root) root = [](num_t a) { return a; };
        }
        
    private:
        
        void set_L2 ()
        {
            sum  = [](num_t a, num_t b) { return a + b; };
            pow  = [](num_t a, num_t b) { return (a - b) * (a - b); };
            root = [](num_t a) { return std::sqrt(a); };
            update_min_dist = [](num_t min_d, num_t cut_d, num_t d) { return min_d + cut_d - d; };
        }
        
        void set_L1 ()
        {
            sum  = [](num_t a, num_t b) { return a + b; };
            pow  = [](num_t a, num_t b) { return std::abs(a - b); };
            root = [](num_t a) { return a; };
            update_min_dist = [](num_t min_d, num_t cut_d, num_t d) { return min_d + cut_d - d; };
        }
        
        void set_Max ()
        {
            sum  = [](num_t a, num_t b) { return a > b ? a : b; };
            pow  = [](num_t a, num_t b) { return std::abs(a - b); };
            root = [](num_t a) { return a; };
            update_min_dist = [](num_t min_d, num_t cut_d, num_t) { return min_d < cut_d ? min_d : cut_d; };
        }
        
        void set_Lp (num_t p_ip)
        {
            p = p_ip;
            sum  = [    ](num_t a, num_t b) { return a + b; };
            pow  = [this](num_t a, num_t b) { return std::pow(std::abs(a - b), p); };
            root = [this](num_t a) { return std::pow(a, 1.0 / p); };
            update_min_dist = [](num_t min_d, num_t cut_d, num_t d) { return min_d + cut_d - d; };
        }
    };
    
    //* Data adaptor */
    template <class num_t>
    struct DataAdaptor
    {
    private:
        
        std::vector<std::vector<num_t>> dataset;
        std::vector<size_t> indices_raw;
        std::vector<std::pair<int, int>> time_indices;
        int exclusion_radius;
        
    public:
        
        DataAdaptor () : exclusion_radius(-1)
        {}
        
        inline void set_dataset (
            const std::vector<std::vector<num_t>>  &dataset_ip,
            const std::vector<std::pair<int, int>> &time_indices_ip,
            const int exclusion_radius_ip)
        {
            std::vector<std::vector<num_t>>().swap(dataset);
            std::vector<size_t>().swap(indices_raw);
            std::vector<std::pair<int, int>>().swap(time_indices);
            
            dataset = dataset_ip;
            time_indices = time_indices_ip;
            exclusion_radius = exclusion_radius_ip;
            
            //* remove points with qnan */
            size_t nt = dataset.size();
            indices_raw.resize(nt);
            for (size_t t = 0; t < nt; ++t) indices_raw[t] = t;
            for (int t = nt - 1; t >= 0; --t)
            {
                bool has_nan = false;
                for (auto p : dataset[t])
                {
                    if (std::isnan(p))
                    {
                        has_nan = true;
                        break;
                    }
                }
                if (has_nan)
                {
                    dataset.erase(dataset.begin() + t);
                    indices_raw.erase(indices_raw.begin() + t);
                }
            }
        }
        
        inline size_t get_raw_index (size_t index) const
        {
            return indices_raw[index];
        }
        
        inline bool check_invalid_query (size_t index, const std::pair<int, int> &time) const
        {
            size_t k = indices_raw[index];
            if (time_indices[k].second != time.second) return true;
            if (std::abs(time_indices[k].first - time.first) > exclusion_radius) return false;
            return true;
        }
        
        inline size_t kdtree_get_point_count () const
        {
            return dataset.size();
        }
        
        inline num_t kdtree_get_pt (const size_t idx, int dim) const
        {
            return dataset[idx][dim];
        }
        
        template <typename norm_t>
        inline num_t eval_norm (
            norm_t &norm, const num_t *a, const size_t idx, size_t size,
            num_t worst_dist = -1) const
        {
            num_t result = num_t();
            const num_t *last = a + size;
            const num_t *lastgroup = last - 3;
            
            size_t dim = 0;
            //* Process 4 items with each loop for efficiency. */
            while (a < lastgroup)
            {
                const num_t diff0 = norm.pow(a[0], dataset[idx][dim++]);
                const num_t diff1 = norm.pow(a[1], dataset[idx][dim++]);
                const num_t diff2 = norm.pow(a[2], dataset[idx][dim++]);
                const num_t diff3 = norm.pow(a[3], dataset[idx][dim++]);
                result = norm.sum(result, norm.sum(diff0, diff1));
                result = norm.sum(result, norm.sum(diff2, diff3));
                a += 4;
                if (0 < worst_dist && worst_dist < result) return result;
            }
            /* Process last 0-3 components. */
            while (a < last)
            {
                result = norm.sum(result, norm.pow(*a++, dataset[idx][dim++]));
            }
            return result;
        }
    };
    
    //* Result set */
    template <typename num_t>
    class ResultSet
    {
        typedef std::pair<size_t, num_t> pair_t;
        
    private:
        
        std::vector<pair_t> pairs;
        size_t nn, nnp, count;
        bool tied;
        
    public:
        
        inline ResultSet () : nn(0), count(0), tied(false)
        {}
        
        inline void set_init (const size_t nn, const bool tied = false)
        {
            std::vector<pair_t>().swap(pairs);
            count = 0;
            this->nn = nnp = nn;
            this->tied = tied;
            pairs.resize(nn);
            pairs[nn - 1].second = std::numeric_limits<num_t>::max();
        }
        
        template <typename norm_t>
        inline void output (
            norm_t &norm, std::vector<size_t> *op_indices, std::vector<num_t> *op_dists,
            const bool sorted = false)
        {
            if (sorted) sort();
            (*op_indices).clear();
            (*op_dists)  .clear();
            for (auto p : pairs)
            {
                (*op_indices).push_back(p.first);
                (*op_dists).push_back(norm.root(p.second));
            }
        }
        
        inline size_t size () { return pairs.size(); }
        
        inline bool add_point (num_t dist, size_t index)
        {
            if (dist > pairs[nn - 1].second) return true;
            
            pairs.resize(nnp + 1);
            size_t i;
            for (i = count; i > 0; --i)
            {
                if (dist < pairs[i - 1].second) pairs[i] = pairs[i - 1];
                else break;
            }
            pairs[i] = std::make_pair(index, dist);
            
            if (tied)
            {
                if (count > nn - 1)
                {
                    if (pairs[nn - 1].second == pairs[nnp].second)
                    {   // new pair has the tied distance
                        ++nnp;
                    }
                    else if (pairs[nn - 1].second != pairs[nnp - 1].second)
                    {   // old pair doesn't has the tied distance
                        nnp = count = nn;
                    }
                }
            }
            pairs.resize(nnp);
            if (count < nnp) ++count;
            return true;
        }
        
        inline bool full ()
        {
            if (count == nn + 1) return true;
            pairs.resize(count);
            return false;
        }
        
        inline num_t worst_dist () const { return pairs[nn - 1].second; }
        
    private:
        
        inline void sort ()
        {
            std::sort(pairs.begin(), pairs.end(),
                [] (pair_t p, pair_t q) { return p.second < q.second; }
            );
        }
    };
    
    //* KD-tree base-class */
    template <class Derived, typename num_t>
    class KDTreeBase
    {
    protected:
        
        typedef typename utils::Node<num_t> Node;
        typedef typename std::vector<utils::Interval<num_t>> BoundingBox;
        
    private:
        
        const num_t EPS = static_cast<num_t>(0.00001);
        
    public:
        
        Node *root_node;
        std::vector<size_t> indices;
        
        size_t root_leaf_max_size;
        size_t root_size;                // Number of current points in the dataset
        size_t root_size_at_index_build; // Number of points in the dataset when the index was built
        int root_dim;                    // Dimensionality of each data point
        
        BoundingBox root_bbox;  // The KD-tree used to find neighbours
        memory::PooledAllocator pool;  // Pooled memory allocator for efficiency
        
    public:
        
        //* Frees the previously-built index. Automatically called within build_index(). */
        void free_index (Derived &obj)
        {
            obj.pool.free_all();
            obj.root_node = NULL;
            obj.root_size_at_index_build = 0;
        }
        
        //* Returns number of points in dataset  */
        size_t size (const Derived &obj) const { return obj.root_size; }
        
        //* Helper accessor to the dataset points: */
        inline num_t dataset_get (const Derived &obj, size_t idx, int dim) const
        {
            return obj.dataset.kdtree_get_pt(idx, dim);
        }
        
        template <typename norm_t>
        num_t compute_initial_dists (
            norm_t &norm, const Derived &obj, const num_t *vec, std::vector<num_t> &dists) const
        {
            assert(vec);
            num_t distsq = num_t();
            for (int i = 0; i < obj.root_dim; ++i)
            {
                if (vec[i] < obj.root_bbox[i].low)
                {
                    dists[i] = norm.pow(vec[i], obj.root_bbox[i].low);
                    distsq = norm.sum(distsq, dists[i]);
                }
                if (vec[i] > obj.root_bbox[i].high)
                {
                    dists[i] = norm.pow(vec[i], obj.root_bbox[i].high);
                    distsq = norm.sum(distsq, dists[i]);
                }
            }
            return distsq;
        }
        
        /**
         * Create a tree node that subdivides the list of vecs from indices[first]
         * to indices[last].  The routine is called recursively on each sublist.
         */
        Node* divide_tree (
            Derived &obj, const size_t L /* left */, const size_t R /* right */,
            BoundingBox &bbox)
        {
            Node *node = obj.pool.template allocate<Node>(); // allocate memory
            
            //* If too few exemplars remain, then make this a leaf node. */
            if (R - L <= static_cast<size_t>(obj.root_leaf_max_size))
            {
                node->child1 = node->child2 = NULL;  // Mark as leaf node.
                node->node_type.lr.left  = L;
                node->node_type.lr.right = R;
                
                // compute bounding-box of leaf points
                for (int i = 0; i < obj.root_dim; ++i)
                {
                    bbox[i].low  = dataset_get(obj, obj.indices[L], i);
                    bbox[i].high = dataset_get(obj, obj.indices[L], i);
                    for (size_t k = L + 1; k < R; ++k)
                    {
                        num_t val = dataset_get(obj, obj.indices[k], i);
                        if (bbox[i].low  > val) bbox[i].low  = val;
                        if (bbox[i].high < val) bbox[i].high = val;
                    }
                }
            }
            else
            {
                size_t idx;
                int cut_dim;
                num_t cut_val;
                
                middle_split(obj, &obj.indices[0] + L, R - L, idx, cut_dim, cut_val, bbox);
                node->node_type.sub.divfeat = cut_dim;
                
                BoundingBox L_bbox(bbox);
                BoundingBox R_bbox(bbox);
                L_bbox[cut_dim].high = cut_val;
                R_bbox[cut_dim].low  = cut_val;
                node->child1 = divide_tree(obj, L, L + idx, L_bbox);
                node->child2 = divide_tree(obj, L + idx, R, R_bbox);
                node->node_type.sub.divlow  = L_bbox[cut_dim].high;
                node->node_type.sub.divhigh = R_bbox[cut_dim].low;
                
                for (int i = 0; i < obj.root_dim; ++i)
                {
                    bbox[i].low  = std::min(L_bbox[i].low , R_bbox[i].low);
                    bbox[i].high = std::max(L_bbox[i].high, R_bbox[i].high);
                }
            }
            return node;
        }
        
    private:
        
        void compute_minmax (
            const Derived &obj, size_t *ind, size_t count, int element,
            num_t &min_elem, num_t &max_elem)
        {
            min_elem = dataset_get(obj, ind[0], element);
            max_elem = dataset_get(obj, ind[0], element);
            for (size_t i = 1; i < count; ++i)
            {
                num_t val = dataset_get(obj, ind[i], element);
                if (val < min_elem) min_elem = val;
                if (val > max_elem) max_elem = val;
            }
        }
        
        void middle_split (
            Derived &obj, size_t *ind, size_t count, size_t &index, int &cut_dim, num_t &cut_val,
            const BoundingBox &bbox)
        {
            num_t max_span = bbox[0].high - bbox[0].low;
            for (int i = 1; i < obj.root_dim; ++i)
            {
                num_t span = bbox[i].high - bbox[i].low;
                if (span > max_span) max_span = span;
            }
            
            num_t min_elem, max_elem, max_spread = -1;
            cut_dim = 0;
            for (int i = 0; i < obj.root_dim; ++i)
            {
                num_t span = bbox[i].high - bbox[i].low;
                if (span > (1 - EPS) * max_span)
                {
                    compute_minmax(obj, ind, count, i, min_elem, max_elem);
                    num_t spread = max_elem - min_elem;
                    if (spread > max_spread)
                    {
                        cut_dim = i;
                        max_spread = spread;
                    }
                }
            }
            compute_minmax(obj, ind, count, cut_dim, min_elem, max_elem);
            
            // split in the middle
            num_t split_val = (bbox[cut_dim].low + bbox[cut_dim].high) / 2;
            cut_val = utils::get_val_within_range(split_val, min_elem, max_elem);
            
            size_t lim1, lim2;
            plane_split(obj, ind, count, cut_dim, cut_val, lim1, lim2);
            index = utils::get_val_within_range(count / 2, lim1, lim2);
        }
        
        /**
         *  Subdivide the list of points by a plane perpendicular on axe corresponding
         *  to the 'cut_dim' dimension at 'cut_val' position.
         *
         *  On return:
         *    dataset[ind[   0, ..., lim1-1]][cut_dim] <  cut_val
         *    dataset[ind[lim1, ..., lim2-1]][cut_dim] == cut_val
         *    dataset[ind[lim2, ..., count ]][cut_dim] >  cut_val
         */
        void plane_split (
            Derived &obj, size_t *ind, const size_t count, int cut_dim, num_t &cut_val,
            size_t &lim1, size_t &lim2)
        {
            //* Move vector indices for left subtree to front of list. */
            size_t L = 0;  // left
            size_t R = count - 1;  // right
            for (;;)
            {
                while (L <= R && dataset_get(obj, ind[L], cut_dim) <  cut_val)      ++L;
                while (L <= R && dataset_get(obj, ind[R], cut_dim) >= cut_val && R) --R;
                if (L > R || !R) break;  // "!R" was added to support unsigned Index types
                std::swap(ind[L++], ind[R--]);
            }
            lim1 = L;
            
            R = count - 1;
            for (;;)
            {
                while (L <= R && dataset_get(obj, ind[L], cut_dim) <= cut_val)      ++L;
                while (L <= R && dataset_get(obj, ind[R], cut_dim) >  cut_val && R) --R;
                if (L > R || !R) break;  // "!R" was added to support unsigned Index types
                std::swap(ind[L++], ind[R--]);
            }
            lim2 = L;
        }
    };
    
    //* KD-tree static index */
    template <typename num_t>
    class KDTreeSingleIndexAdaptor
        : public KDTreeBase <KDTreeSingleIndexAdaptor<num_t>, num_t>
    {
        typedef typename nanoflann::KDTreeSingleIndexAdaptor<num_t> Derived;
        typedef typename nanoflann::KDTreeBase<Derived, num_t> Base;
        typedef typename Base::Node Node;
        typedef typename Base::BoundingBox BoundingBox;
        
    public:
        
        NormStruct <num_t> norm;
        DataAdaptor<num_t> dataset;
        
    public:
        
        KDTreeSingleIndexAdaptor ()
        {}
        
        void initialize (
            const int n_dim,
            const std::vector<std::vector<num_t>>  &points,
            const std::vector<std::pair<int, int>> &time_indices,
            const int exclusion_radius = 0)
        {
            dataset.set_dataset(points, time_indices, exclusion_radius);
            Base::root_node = NULL;
            Base::root_dim = n_dim;
            Base::root_leaf_max_size = 10;
            set_norm();
            build_index();
        }
        
        void set_norm (NORM norm_type = L2, bool do_root = true, num_t p = 0.5)
        {
            norm.set_norm(norm_type, do_root, p);
        }
        
        size_t knn_search (
            const num_t *query, const std::pair<int, int> &time,
            std::vector<size_t> *op_indices, std::vector<num_t> *op_dists,
            const size_t num_closest, const bool tied = true) const
        {
            size_t nn = Base::root_size < num_closest ? Base::root_size : num_closest;
            nanoflann::ResultSet<num_t> result;
            result.set_init(nn, tied);
            this->find_neighbors(result, query, time);
            result.output(norm, op_indices, op_dists);
            for (auto &x : *op_indices) x = dataset.get_raw_index(x);
            return result.size();
        }
        
    private:
        
        void build_index ()
        {
            this->free_index(*this);
            Base::root_size = dataset.kdtree_get_point_count();
            Base::root_size_at_index_build = Base::root_size;
            
            Base::indices.resize(Base::root_size);
            for (size_t i = 0; i < Base::root_size; i++) Base::indices[i] = i;
            if (Base::root_size == 0) return;
            
            compute_BoundingBox(Base::root_bbox);
            Base::root_node = this->divide_tree(*this, 0, Base::root_size, Base::root_bbox);
        }
        
        void compute_BoundingBox (BoundingBox &bbox)
        {
            bbox.resize(Base::root_dim);
            for (int i = 0; i < Base::root_dim; ++i)
            {
                bbox[i].low = bbox[i].high = this->dataset_get(*this, 0, i);
                for (size_t k = 1; k < Base::root_size; ++k)
                {
                    num_t val = this->dataset_get(*this, k, i);
                    if (val < bbox[i].low ) bbox[i].low  = val;
                    if (val > bbox[i].high) bbox[i].high = val;
                }
            }
        }
        
        /**
         * Find set of nearest neighbors to 'query[0:dim-1]'.
         * Their indices are stored inside the 'result' object.
         * Return 'true' if the requested neighbors could be found.
         */
        template <typename RESULTSET>
        bool find_neighbors (
            RESULTSET &result, const num_t *query, const std::pair<int, int> &time,
            const float eps = 0) const
        {
            assert(query);
            if (this->size(*this) == 0) return false;
            if (!Base::root_node)
            {
                throw std::runtime_error(
                    "[nanoflann] findNeighbors() called before building the index."
                );
            }
            std::vector<num_t> dists(Base::root_dim, num_t(0));
            num_t distsq = this->compute_initial_dists(norm, *this, query, dists);
            search_level(result, query, time, Base::root_node, distsq, dists, 1.0 + eps);
            return result.full();
        }
        
        /**
         * Performs an exact search in the tree starting from a node.
         * Return 'true' if the search should be continued, false if the results are sufficient.
         */
        template <class RESULTSET>
        bool search_level (
            RESULTSET &result_set, const num_t *query,
            const std::pair<int, int> &time, const Node *node, num_t mindistsq,
            std::vector<num_t> &dists, const float epsError) const
        {
            if (node->child1 == NULL && node->child2 == NULL)  //* Leaf node */
            {
                for (size_t i = node->node_type.lr.left; i < node->node_type.lr.right; ++i)
                {
                    size_t index = Base::indices[i];
                    if (dataset.check_invalid_query(index, time)) continue;
                    num_t dist = dataset.eval_norm(norm, query, index, Base::root_dim);
                    if (!result_set.add_point(dist, index))
                    {
                        return false; // Done searching!
                    }
                }
                return true;
            }
            
            /* Which child branch should be taken first? */
            int cut_dim = node->node_type.sub.divfeat;
            num_t val = query[cut_dim];
            num_t diff1 = val - node->node_type.sub.divlow;
            num_t diff2 = val - node->node_type.sub.divhigh;
            
            Node *bestChild, *otherChild;
            num_t cut_dist;
            if (diff1 + diff2 < 0)
            {
                bestChild  = node->child1;
                otherChild = node->child2;
                cut_dist = norm.pow(val, node->node_type.sub.divhigh);
            }
            else
            {
                bestChild  = node->child2;
                otherChild = node->child1;
                cut_dist = norm.pow(val, node->node_type.sub.divlow);
            }
            
            /* Call recursively to search next level down. */
            if (!search_level(result_set, query, time, bestChild, mindistsq, dists, epsError))
            {
                return false;  // Done searching
            }
            
            num_t dist = dists[cut_dim];
            mindistsq = norm.update_min_dist(mindistsq, cut_dist, dist);
            dists[cut_dim] = cut_dist;
            if (mindistsq * epsError <= result_set.worst_dist())
            {
                if (!search_level(result_set, query, time, otherChild, mindistsq, dists, epsError))
                {
                    return false;  // Done searching
                }
            }
            dists[cut_dim] = dist;
            return true;
        }
    };
}
    
#endif