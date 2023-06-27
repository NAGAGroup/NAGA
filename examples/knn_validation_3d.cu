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
#include <chrono>
#include <naga/detail/nanoflann_utils.h>
#include <naga/segmentation/nearest_neighbors.cuh>
#include <thrust/random.h>

int main() {
    size_t grid_size   = 20;
    float grid_spacing = 0.1;
    uint k             = 64;

    size_t point_count = grid_size * grid_size * grid_size;
    sclx::array<float, 2> grid{3, point_count};

    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{point_count},
            grid,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                grid(0, idx[0])
                    = static_cast<float>(idx[0] % grid_size) * grid_spacing;
                grid(1, idx[0])
                    = static_cast<float>((idx[0] / grid_size) % grid_size)
                    * grid_spacing;
                grid(2, idx[0])
                    = static_cast<float>(idx[0] / grid_size / grid_size)
                    * grid_spacing;
            }
        );
    }).get();

    size_t num_query_points = 30;
    sclx::array<float, 2> query_points{3, num_query_points};


    sclx::execute_kernel([&](sclx::kernel_handler& handle) {
        handle.launch(
            sclx::md_range_t<1>{num_query_points},
            query_points,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                thrust::default_random_engine rng;
                thrust::uniform_real_distribution<float> dist(0, grid_spacing * static_cast<float>(grid_size - 1));
                rng.discard(idx[0]);
                query_points(0, idx[0]) = dist(rng);
                query_points(1, idx[0]) = dist(rng);
                query_points(2, idx[0]) = dist(rng);
            }
        );
    }).get();


    using point_map_t = naga::default_point_map<float, 3>;
    using segmentation_t
        = naga::segmentation::knn::nd_cubic_segmentation<point_map_t>;
    point_map_t query_point_map{query_points};

    segmentation_t partitioner(grid, k);
    auto [distances_squared, indices]
        = naga::segmentation::batched_nearest_neighbors(
            k,
            query_point_map,
            partitioner
        );

    using nanoflann_cloud_t = naga::detail::PointCloud<float>;
    nanoflann_cloud_t cloud{};
    for (size_t i = 0; i < point_count; ++i) {
        auto point = &grid(0, i);
        cloud.pts.push_back(
            nanoflann_cloud_t::Point{point[0], point[1], point[2]}
        );
    }

    using kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<float, nanoflann_cloud_t>,
        nanoflann_cloud_t,
        3 /* dim */
        >;
    kd_tree_t tree{3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams{10}};

    for (size_t i = 0; i < num_query_points; ++i) {
        auto point = query_point_map[i];
        std::vector<size_t> ret_index(k);
        std::vector<float> out_dist_sqr(k);
        nanoflann::KNNResultSet<float> resultSet(k);
        resultSet.init(&ret_index[0], &out_dist_sqr[0]);
        tree.findNeighbors(resultSet, &point[0], nanoflann::SearchParams{});

        for (size_t j = 0; j < k; ++j) {
            auto index_ptr
                = std::find(&indices(0, i), &indices(0, i) + k, ret_index[j]);
            if (index_ptr == &indices(0, i) + k) {
                std::cout << "Error: index " << ret_index[j]
                          << " not found in indices for point " << i
                          << std::endl;
                std::cout << "Expected indices: ";
                for (size_t l = 0; l < k; ++l) {
                    std::cout << ret_index[l] << " ";
                }
                std::cout << std::endl;
                std::cout << "Actual indices:   ";
                for (size_t l = 0; l < k; ++l) {
                    std::cout << indices(l, i) << " ";
                }
                return 1;
            }
        }
    }

    return 0;
}
