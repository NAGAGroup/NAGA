
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
#include "rectangular_partitioner.cuh"
#include <tuple>

namespace naga {

template<class T>
__device__ int insert_if_less_than_max(T value, T* arr, int n) {
    T max     = arr[0];
    int index = 0;
    for (int i = 1; i < n; i++) {
        if (max < arr[i]) {
            max   = arr[i];
            index = i;
        }
    }
    if (value < max) {
        arr[index] = value;
        return index;
    } else {
        return -1;
    }
}

template<
    class PointMapType,
    class T,
    class DistanceSquaredOp = distance_functions::loopless::euclidean_squared<
        point_map_traits<PointMapType>::point_traits::dimensions>>
std::tuple<
    sclx::array<
        std::remove_const_t<
            typename point_map_traits<PointMapType>::point_traits::value_type>,
        2>,
    sclx::array<sclx::index_t, 2>>
compute_euclidean_nearest_neighbors_2d(
    uint k,
    const PointMapType& query_points,
    const rectangular_partitioner<
        T,
        point_map_traits<PointMapType>::point_traits::dimensions>&
        segmented_ref_points,
    DistanceSquaredOp&& distance_squared_op = DistanceSquaredOp()
) {
    constexpr uint max_k = 256;
    if (k > max_k) {
        sclx::throw_exception<std::invalid_argument>(
            "Requested number of neighbors exceeds maximum of "
                + std::to_string(max_k),
            "naga::"
        );
    }

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

    for (int i = 0; i < dimensions; i++) {
        if (segmented_ref_points.shape()[i] > std::numeric_limits<int>::max()) {
            sclx::throw_exception<std::overflow_error>(
                "Partitioner has too many partitions in dimension "
                    + std::to_string(i) + ", will cause overflow.",
                "naga::"
            );
        }
    }

    sclx::array<value_type, 2> distances_squared{k, query_points.size()};
    sclx::array<sclx::index_t, 2> indices{k, query_points.size()};

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        auto results = sclx::make_array_tuple(distances_squared, indices);

        handler.launch(
            sclx::md_range_t<1>{query_points.size()},
            results,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                value_type distances_squared_tmp[max_k]{};
                value_type query_point[dimensions];
                const auto& point_tmp = query_points[idx];
                memcpy(
                    query_point,
                    &point_tmp[0],
                    dimensions * sizeof(value_type)
                );

                int max_search_radius
                    = static_cast<int>(segmented_ref_points.shape()[0] / 2);
                max_search_radius = max(
                    max_search_radius,
                    static_cast<int>(segmented_ref_points.shape()[1] / 2)
                );

                sclx::md_index_t<dimensions> part_idx
                    = segmented_ref_points.get_partition_index(query_point);
                int search_radius     = 0;
                uint n_found          = 0;
                bool new_points_found = true;
                while (new_points_found || n_found < k) {
                    new_points_found  = false;
                    int search_length = 2 * search_radius + 1;
                    for (int search_idx = 0;
                         search_idx < search_length * search_length;
                         ++search_idx) {
                        int i = search_idx % search_length;
                        int j = search_idx / search_length;
                        if (!(j == 0 || i == 0 || j == search_length - 1
                              || i == search_length - 1))
                            continue;

                        sclx::md_index_t<dimensions> part_search_idx = part_idx;

                        i = (i - search_radius) + static_cast<int>(part_idx[0]);
                        j = (j - search_radius) + static_cast<int>(part_idx[1]);
                        if (i < 0 || j < 0
                            || i >= segmented_ref_points.shape()[0]
                            || j >= segmented_ref_points.shape()[1])
                            continue;

                        part_search_idx[0] = i;
                        part_search_idx[1] = j;

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
                                distances_squared_tmp[n_found] = distance_squared;
                                indices(n_found, idx[0])
                                    = part.indices()[p_idx];
                                n_found++;
                            } else {
                                int index = insert_if_less_than_max(
                                    distance_squared,
                                    distances_squared_tmp,
                                    k
                                );
                                if (index != -1) {
                                    indices(index, idx[0])
                                        = part.indices()[p_idx];
                                }
                            }
                            p_idx++;
                        }
                    }
                }

                memcpy(
                    distances_squared_tmp,
                    &distances_squared(0, idx[0]),
                    k * sizeof(value_type)
                );
            }
        );
    });

    return std::make_tuple(distances_squared, indices);
}

}  // namespace naga