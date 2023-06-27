
// BSD 3-Clause License
//
// Copyright (c) 2023 Jack Myers
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "../distance_functions.cuh"
#include "../point_map.cuh"
#include "nd_cubic_segmentation.cuh"
#include <thrust/random.h>
#include <tuple>

namespace naga::segmentation {

template<class PointMapType>
using default_distance_squared
    = distance_functions::loopless::euclidean_squared<
        point_map_traits<PointMapType>::point_traits::dimensions>;

// we put the following alias in a namespace to avoid name collisions
namespace knn {

template<class PointMapType>
using nd_cubic_segmentation = ::naga::segmentation::nd_cubic_segmentation<
    std::decay_t<
        typename point_map_traits<PointMapType>::point_traits::value_type>,
    point_map_traits<PointMapType>::point_traits::dimensions>;

}

template<class T, class U>
__host__ __device__ void insertion_sort(
    T* array,
    U* matched_array,
    uint size,
    bool is_sorted_except_last = false
) {
    if (!is_sorted_except_last) {
        for (uint i = 1; i < size; ++i) {
            T key         = array[i];
            U key_matched = matched_array[i];
            int j         = i - 1;
            while (j >= 0 && array[j] > key) {
                array[j + 1]         = array[j];
                matched_array[j + 1] = matched_array[j];
                j--;
            }
            array[j + 1]         = key;
            matched_array[j + 1] = key_matched;
        }
    } else {
        T key         = array[size - 1];
        U key_matched = matched_array[size - 1];
        for (uint i = size - 1; i > 0; --i) {
            if (array[i - 1] > key) {
                array[i]             = array[i - 1];
                matched_array[i]     = matched_array[i - 1];
                array[i - 1]         = key;
                matched_array[i - 1] = key_matched;
            } else {
                break;
            }
        }
    }
}

template<
    class PointMapType,
    class DistanceSquaredOp = default_distance_squared<PointMapType>>
__host__ void batched_nearest_neighbors(
    uint k,
    sclx::array<
        std::decay_t<
            typename point_map_traits<PointMapType>::point_traits::value_type>,
        2>& distances_squared,
    sclx::array<sclx::index_t, 2>& indices,
    const PointMapType& query_points,
    const knn::nd_cubic_segmentation<PointMapType>& segmented_ref_points,
    DistanceSquaredOp&& distance_squared_op = DistanceSquaredOp()
) {
    using distance_traits = distance_functions::distance_function_traits<
        std::decay_t<DistanceSquaredOp>>;
    static_assert(
        distance_traits::is_squared,
        "Distance function must return squared distance."
    );

    using point_type = typename point_map_traits<PointMapType>::point_type;
    using value_type = std::remove_const_t<
        typename point_map_traits<PointMapType>::point_traits::value_type>;
    constexpr uint dimensions
        = point_map_traits<PointMapType>::point_traits::dimensions;

    uint shared_mem_per_thread
        = sizeof(value_type) * (k + dimensions) + sizeof(sclx::index_t) * k;
    uint max_shared_mem_per_block = /*48KB*/ 49152;
    uint max_threads_per_block
        = max_shared_mem_per_block / shared_mem_per_thread;
    if (max_threads_per_block == 0) {
        sclx::throw_exception<std::invalid_argument>(
            "Requested number of neighbors is too large for shared memory.",
            "naga::"
        );
    }

    for (int i = 0; i < dimensions; i++) {
        if (segmented_ref_points.shape()[i] > std::numeric_limits<int>::max()) {
            sclx::throw_exception<std::overflow_error>(
                "Partitioner has too many partitions in dimension "
                    + std::to_string(i) + ", will cause overflow.",
                "naga::"
            );
        }
    }

    if (query_points.size() != distances_squared.shape()[1]) {
        sclx::throw_exception<std::invalid_argument>(
            "Query points and distances array have different sizes.",
            "naga::"
        );
    }

    if (query_points.size() != indices.shape()[1]) {
        sclx::throw_exception<std::invalid_argument>(
            "Query points and indices array have different sizes.",
            "naga::"
        );
    }

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        auto results = sclx::make_array_tuple(distances_squared, indices);
        sclx::local_array<value_type, 2> distances_squared_shared(
            handler,
            {k, max_threads_per_block}
        );
        sclx::local_array<sclx::index_t, 2> indices_shared(
            handler,
            {k, max_threads_per_block}
        );
        sclx::local_array<value_type, 2> query_point_shared(
            handler,
            {dimensions, max_threads_per_block}
        );

        handler.launch(
            sclx::md_range_t<1>{query_points.size()},
            results,
            [=] __device__(
                const sclx::md_index_t<1>& idx,
                const auto& info
            ) mutable {
                value_type* distances_squared_tmp
                    = &distances_squared_shared(0, info.local_thread_id()[0]);
                sclx::index_t* indices_tmp
                    = &indices_shared(0, info.local_thread_id()[0]);
                value_type* query_point
                    = &query_point_shared(0, info.local_thread_id()[0]);
                const auto& point_tmp = query_points[idx];
                memcpy(
                    query_point,
                    &point_tmp[0],
                    dimensions * sizeof(value_type)
                );

                int max_search_radius = static_cast<int>(
                    (segmented_ref_points.shape()[0] + 1) / 2
                );
                for (int i = 1; i < dimensions; i++) {
                    max_search_radius = max(
                        max_search_radius,
                        static_cast<int>(
                            (segmented_ref_points.shape()[i] + 1) / 2
                        )
                    );
                }

                sclx::md_index_t<dimensions> part_idx
                    = segmented_ref_points.get_partition_index(query_point);
                int search_radius     = 0;
                uint n_found          = 0;
                bool new_points_found = true;
                int search_index_list[dimensions]{};

                constexpr size_t prime_numbers[]{
                    2305843009213693951,
                    82589933,
                    77232917,
                    2147483647,
                    305175781,
                    2521008887,
                    63018038201,
                    489133282872437279,
                    15285151248481,
                    228204732751,
                    65610001};
                thrust::default_random_engine rng(idx[0]);
                thrust::uniform_int_distribution<int> dist(
                    0,
                    sizeof(prime_numbers) / sizeof(size_t) - 1
                );
                while ((new_points_found || n_found < k)
                       && search_radius <= max_search_radius) {

                    new_points_found = false;

                    int search_length = 2 * search_radius + 1;

                    size_t max_linear_search_idx
                        = 2 * dimensions
                        * math::loopless::pow<dimensions - 1>(search_length);
                    max_linear_search_idx
                        = search_length == 1 ? 1 : max_linear_search_idx;
                    size_t prime_number = prime_numbers[dist(rng)];
                    size_t state        = 0;

                    for (size_t linear_search_idx = 0;
                         linear_search_idx < max_linear_search_idx;
                         ++linear_search_idx) {

                        bool is_valid_linear_index = true;

                        state = (state + prime_number) % max_linear_search_idx;
                        size_t linear_search_idx_tmp = state;

                        int locked_dim = (linear_search_idx_tmp / 2
                                          / math::loopless::pow<dimensions - 1>(
                                              search_length
                                          ))
                                       % dimensions;

                        search_index_list[locked_dim]
                            = (linear_search_idx
                               / math::loopless::pow<dimensions - 1>(
                                   search_length
                               )) % 2
                                   == 0
                                ? 0
                                : search_length - 1;

                        linear_search_idx_tmp
                            -= (linear_search_idx_tmp
                                / math::loopless::pow<dimensions - 1>(
                                    search_length
                                ))
                             * math::loopless::pow<dimensions - 1>(search_length
                             );

                        for (int i = 0; i < dimensions - 1; i++) {
                            search_index_list[(locked_dim + i + 1) % dimensions]
                                = linear_search_idx_tmp % search_length;
                            linear_search_idx_tmp /= search_length;
                        }
                        for (int i = 0; i < locked_dim; ++i) {
                            is_valid_linear_index
                                = is_valid_linear_index
                               && !(search_index_list[i] == 0
                                    || search_index_list[i] == search_length - 1
                               );
                        }

                        sclx::md_index_t<dimensions> part_search_idx = part_idx;

                        if (idx[0] == query_points.size() / 2) {
                            printf(
                                "search_index_list: %d %d %d\n",
                                search_index_list[0],
                                search_index_list[1],
                                search_index_list[2]
                            );
                        }

                        for (int i = 0; i < dimensions; i++) {
                            search_index_list[i] = search_index_list[i]
                                                 - search_radius + part_idx[i];
                            if (search_index_list[i] < 0
                                || search_index_list[i] >= static_cast<int>(
                                       segmented_ref_points.shape()[i]
                                   )) {
                                is_valid_linear_index = false;
                                break;
                            }
                            part_search_idx[i] = search_index_list[i];
                        }
                        if (!is_valid_linear_index) {
                            continue;
                        }

                        const auto& part
                            = segmented_ref_points.get_partition(part_search_idx
                            );

                        uint p_idx = 0;
                        for (const auto& reference_point : part) {
                            auto distance_squared = distance_squared_op(
                                query_point,
                                reference_point,
                                dimensions
                            );
                            if (n_found < k) {
                                distances_squared_tmp[n_found]
                                    = distance_squared;
                                indices_tmp[n_found] = part.indices()[p_idx];
                                n_found++;
                                if (n_found == k) {
                                    insertion_sort(
                                        distances_squared_tmp,
                                        indices_tmp,
                                        k
                                    );
                                }
                                new_points_found = true;
                            } else {
                                if (distance_squared
                                    < distances_squared_tmp[k - 1]) {
                                    distances_squared_tmp[k - 1]
                                        = distance_squared;
                                    indices_tmp[k - 1] = part.indices()[p_idx];
                                    insertion_sort(
                                        distances_squared_tmp,
                                        indices_tmp,
                                        k,
                                        true
                                    );
                                }
                            }
                            p_idx++;
                        }
                    }

                    ++search_radius;
                }

                if (n_found < k) {
                    printf(
                        "Warning: Not enough points found for query point %u\n",
                        static_cast<uint>(idx[0])
                    );
                }

                memcpy(
                    &distances_squared(0, idx[0]),
                    distances_squared_tmp,
                    k * sizeof(value_type)
                );
                memcpy(
                    &indices(0, idx[0]),
                    indices_tmp,
                    k * sizeof(sclx::index_t)
                );
            },
            sclx::md_range_t<1>{max_threads_per_block}
        );
    }).get();
}

template<
    class PointMapType,
    class DistanceSquaredOp = distance_functions::loopless::euclidean_squared<
        point_map_traits<PointMapType>::point_traits::dimensions>>
std::tuple<
    sclx::array<
        std::remove_const_t<
            typename point_map_traits<PointMapType>::point_traits::value_type>,
        2>,
    sclx::array<sclx::index_t, 2>>
batched_nearest_neighbors(
    uint k,
    const PointMapType& query_points,
    const knn::nd_cubic_segmentation<PointMapType>& segmented_ref_points,
    DistanceSquaredOp&& distance_squared_op = {}
) {

    using value_type = std::remove_const_t<
        typename point_map_traits<PointMapType>::point_traits::value_type>;

    sclx::array<value_type, 2> distances_squared{k, query_points.size()};
    sclx::array<sclx::index_t, 2> indices{k, query_points.size()};

    batched_nearest_neighbors(
        k,
        distances_squared,
        indices,
        query_points,
        segmented_ref_points,
        distance_squared_op
    );

    return std::make_tuple(distances_squared, indices);
}

}  // namespace naga::segmentation
