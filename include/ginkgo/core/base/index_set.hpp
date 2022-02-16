/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_
#define GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_


#include <algorithm>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


/**
 * An index set class represents an ordered set of intervals. The index set
 * contains subsets which store the starting and end points of a range,
 * [a,b), storing the first index and one past the last index. As the
 * index set only stores the end-points of ranges, it can be quite efficient in
 * terms of storage.
 *
 * This class is particularly useful in storing continuous ranges. For example,
 * consider the index set (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 18, 19, 20, 21,
 * 42). Instead of storing the entire array of indices, one can store intervals
 * ([1,9), [10,13), [18,22), [42,43)), thereby only using half the storage.
 *
 * We store three arrays, one (subsets_begin) with the starting indices of the
 * subsets in the index set, another (subsets_end) storing one
 * index beyond the end indices of the subsets and the last
 * (superset_cumulative_indices) storing the cumulative number of indices in the
 * subsequent subsets with an initial zero which speeds up the
 * querying. Additionally, the arrays conataining the range boundaries
 * (subsets_begin, subsets_end) are stored in a sorted fashion.
 *
 * Therefore the storage would look as follows
 *
 * > index_set = (1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 18, 19, 20, 21, 42)
 * > subsets_begin = {1, 10, 18, 42}
 * > subsets_end = {9, 13, 22, 43}
 * > superset_cumulative_indices = {0, 8, 11, 15, 16}
 *
 * @tparam index_type  type of the indices being stored in the index set.
 *
 * @ingroup IndexSet
 */
template <typename IndexType = int32>
class IndexSet : public EnablePolymorphicObject<IndexSet<IndexType>> {
    friend class EnablePolymorphicObject<IndexSet>;

public:
    /**
     * The type of elements stored in the index set.
     */
    using index_type = IndexType;

    /**
     * Creates an empty IndexSet tied to the specified Executor.
     *
     * @param exec  the Executor where the IndexSet data is allocated
     */
    IndexSet(std::shared_ptr<const Executor> exec)
        : EnablePolymorphicObject<IndexSet>(std::move(exec))
    {}

    /**
     * Creates an index set on the specified executor from the initializer list.
     *
     * @param exec  the Executor where the index set data will be allocated
     * @param init_list  the indices that the index set should hold in an
     *                   initializer_list.
     * @param is_sorted  a parameter that specifies if the indices array is
     *                   sorted or not. `true` if sorted.
     */
    IndexSet(std::shared_ptr<const gko::Executor> executor,
             std::initializer_list<IndexType> init_list,
             const bool is_sorted = false)
        : EnablePolymorphicObject<IndexSet>(std::move(executor)),
          index_space_size_(
              *(std::max_element(std::begin(init_list), std::end(init_list))) +
              1)
    {
        this->populate_subsets(
            Array<IndexType>(this->get_executor(), init_list), is_sorted);
    }

    /**
     * Creates an index set on the specified executor and the given size
     *
     * @param exec  the Executor where the index set data will be allocated
     * @param size  the maximum index the index set it allowed to hold. This
     *              is the size of the index space.
     * @param indices  the indices that the index set should hold.
     * @param is_sorted  a parameter that specifies if the indices array is
     *                   sorted or not. `true` if sorted.
     */
    IndexSet(std::shared_ptr<const gko::Executor> executor,
             const index_type size, const gko::Array<index_type>& indices,
             const bool is_sorted = false)
        : EnablePolymorphicObject<IndexSet>(std::move(executor)),
          index_space_size_(size)
    {
        GKO_ASSERT(index_space_size_ >= indices.get_num_elems());
        this->populate_subsets(indices, is_sorted);
    }

    /**
     * Creates a copy of another IndexSet on a different executor.
     *
     * @param exec  the executor where the new IndexSet will be created
     * @param other  the IndexSet to copy from
     */
    IndexSet(std::shared_ptr<const Executor> exec, const IndexSet& other)
        : IndexSet(exec)
    {
        *this = other;
    }

    /**
     * Returns the size of the index set space.
     *
     * @return  the size of the index set space.
     */
    index_type get_size() const { return this->index_space_size_; }

    /**
     * Returns if the index set is contiguous
     *
     * @return  if the index set is contiguous.
     */
    bool is_contiguous() const { return (this->get_num_subsets() <= 1); }

    /**
     * Return the actual number of indices stored in the index set
     *
     * @return  number of indices stored in the index set
     */
    index_type get_num_elems() const { return this->num_stored_indices_; };

    /**
     * Return the global index given a local index.
     *
     * Consider the set idx_set = (0, 1, 2, 4, 6, 7, 8, 9). This function
     * returns the element at the global index k stored in the index set. For
     * example, `idx_set.get_global_index(0) == 0` `idx_set.get_global_index(3)
     * == 4` and `idx_set.get_global_index(7) == 9`
     *
     * @note This function returns a scalar value and needs a scalar value.
     *       For repeated queries, it is more efficient to use the Array
     *       functions that take and return arrays which allow for more
     *       throughput.
     *
     * @param local_index  the local index.
     * @return  the global index from the index set.
     *
     * @warning This single entry query can have significant kernel lauch
     *          overheads and should be avoided if possible.
     */
    index_type get_global_index(index_type local_index) const;

    /**
     * Return the local index given a global index.
     *
     * Consider the set idx_set = (0, 1, 2, 4, 6, 7, 8, 9). This function
     * returns the local index in the index set of the provided index set. For
     * example, `idx_set.get_local_index(0) == 0` `idx_set.get_local_index(4)
     * == 3` and `idx_set.get_local_index(6) == 4`.
     *
     * @note This function returns a scalar value and needs a scalar value.
     *       For repeated queries, it is more efficient to use the Array
     *       functions that take and return arrays which allow for more
     *       throughput.
     *
     * @param global_index  the global index.
     *
     * @return  the local index of the element in the index set.
     *
     * @warning This single entry query can have significant kernel lauch
     *          overheads and should be avoided if possible.
     */
    index_type get_local_index(index_type global_index) const;

    /**
     * Return which set the global index belongs to.
     *
     * Consider the set idx_set = (0, 1, 2, 4, 6, 7, 8, 9). This function
     * returns the subset id in the index set of the input global index. For
     * example, `idx_set.get_subset_id(0) == 0` `idx_set.get_subset_id(4)
     * == 1` and `idx_set.get_subset_id(6) == 2`.
     *
     * @note This function returns a scalar value and needs a scalar value.
     *       For repeated queries, it is more efficient to use the Array
     *       functions that take and return arrays which allow for more
     *       throughput.
     *
     * @param global_index  the global index.
     *
     * @return  the local index of the element in the index set.
     *
     * @warning This single entry query can have significant kernel lauch
     *          overheads and should be avoided if possible.
     */
    index_type get_subset_id(index_type global_index) const;

    /**
     * This is an array version of the scalar function above.
     *
     * @param local_indices  the local index array.
     * @param is_sorted  a parameter that specifies if the query array is sorted
     *                   or not. `true` if sorted .
     *
     * @return  the global index array from the index set.
     *
     * @note Whenever possible, passing a sorted array is preferred as the
     *       queries can be significantly faster.
     * @note Passing local indices from [0, size) is equivalent to using the
     *       @to_global_indices function.
     */
    Array<index_type> map_local_to_global(
        const Array<index_type>& local_indices,
        const bool is_sorted = false) const;

    /**
     * This is an array version of the scalar function above.
     *
     * @param global_indices  the global index array.
     * @param is_sorted  a parameter that specifies if the query array is sorted
     *                   or not. `true` if sorted.
     *
     * @return  the local index array from the index set.
     *
     * @note Whenever possible, passing a sorted array is preferred as the
     *       queries can be significantly faster.
     */
    Array<index_type> map_global_to_local(
        const Array<index_type>& global_indices,
        const bool is_sorted = false) const;

    /**
     * This function allows the user obtain a decompresed global_indices Array
     * from the indices stored in the index set
     *
     * @return  the decompressed set of indices.
     */
    Array<index_type> to_global_indices() const;

    /**
     * Checks if the individual global indeices exist in the index set.
     *
     * @param global_indices  the indices to check.
     * @param is_sorted  a parameter that specifies if the query array is sorted
     *                   or not. `true` if sorted.
     *
     * @return  the Array that contains element wise whether the corresponding
     *          global index in the index set or not.
     */
    Array<bool> contains(const Array<index_type>& global_indices,
                         const bool is_sorted = false) const;

    /**
     * Checks if the global index exists in the index set.
     *
     * @param global_index  the index to check.
     *
     * @return  whether the element exists in the index set.
     *
     * @warning This single entry query can have significant kernel lauch
     *          overheads and should be avoided if possible.
     */
    bool contains(const index_type global_index) const;

    /**
     * Returns the number of subsets stored in the index set.
     *
     * @return  the number of stored subsets.
     */
    index_type get_num_subsets() const
    {
        return this->subsets_begin_.get_num_elems();
    }

    /**
     * Returns a pointer to the beginning indices of the subsets.
     *
     * @return  a pointer to the beginning indices of the subsets.
     */
    const index_type* get_subsets_begin() const
    {
        return this->subsets_begin_.get_const_data();
    }

    /**
     * Returns a pointer to the end indices of the subsets.
     *
     * @return  a pointer to the end indices of the subsets.
     */
    const index_type* get_subsets_end() const
    {
        return this->subsets_end_.get_const_data();
    }

    /**
     * Returns a pointer to the cumulative indices of the superset of
     * the subsets.
     *
     * @return  a pointer to the cumulative indices of the superset of the
     *          subsets.
     */
    const index_type* get_superset_indices() const
    {
        return this->superset_cumulative_indices_.get_const_data();
    }

private:
    void populate_subsets(const gko::Array<index_type>& indices,
                          const bool is_sorted);

    index_type index_space_size_;
    index_type num_stored_indices_;
    gko::Array<index_type> subsets_begin_;
    gko::Array<index_type> subsets_end_;
    gko::Array<index_type> superset_cumulative_indices_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_
