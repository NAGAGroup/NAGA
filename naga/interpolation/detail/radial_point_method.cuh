
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

namespace naga::interpolation::detail::radial_point_method {

template<class T, class PointMapType, class ShapeFunctionType>
static sclx::array<T, 2> compute_weights(
    sclx::array<T, 2>& source_points,
    sclx::array<size_t, 2>& interpolating_indices,
    sclx::array<T, 2>& source2query_dist_squared,
    const PointMapType& query_points,
    const T& approx_particle_spacing,
    uint group_size                         = 1,
    const ShapeFunctionType& shape_function = ShapeFunctionType{},
    const std::vector<int>& devices         = {}
) {
    using point_type = typename point_map_traits<PointMapType>::point_type;
    using value_type = std::decay_t<
        typename point_map_traits<PointMapType>::point_traits::value_type>;
    constexpr static uint dimensions
        = point_map_traits<PointMapType>::point_traits::dimensions;
    static_assert(
        std::is_same_v<T, std::decay_t<value_type>>,
        "source_points and query_points must have the same value_type"
    );

    size_t num_source_points = source_points.shape()[1];
    size_t num_query_points  = query_points.size();
    uint support_size = static_cast<uint>(interpolating_indices.shape()[0]);

    if (num_query_points % group_size != 0) {
        sclx::throw_exception<std::invalid_argument>(
            "query_points.size() must be a multiple of group_size"
        );
    }
    if (source_points.shape()[0] != dimensions) {
        sclx::throw_exception<std::invalid_argument>(
            "source_points.shape()[0] must be equal to dimensions"
        );
    }

    // in order to match the paper, I will be using the same notation
    //
    // B0 - is the matrix of shape functions, where each row is the set of
    // shape functions for one of the support points
    // psi_vector - vector of known values at each support point
    // a - vector of interpolating weights
    //
    // recall that we assume B0*a = psi_vector, therefore we can solve for
    // a by inverting B0 and multiplying by psi_vector
    //
    // Our method also augments the formualation with a monomial basis for
    // better conditioning.
    //
    // This means the interpolation ends up looking like:
    // psi(x) = B(x)^T * a + P(x)^T * b
    //
    // Therefore, we actually need to solver for b as well
    //
    // P0 - matrix of monomials where each row is the set of monomials for
    // one of the support points
    //
    // We also enforce the constraint that P0^T * a = 0
    //
    // So we get
    // G0 := [[B0, P0]; [P0^T, 0]] * [[a]; [b]] = [[psi_vector]; [0]]
    //
    // Interpolating to x is then
    // psi(x) = G(x)^T * G0^-1 * [[psi_vector]; [0]]
    //        = M * [[psi_vector]; [0]]
    // We can see that we only need M[:, 1:num_support] since the rest of
    // the columns are multiplied by 0
    //
    // Therefore we can compute our weights as w_i = \sum_j=1^num_support
    // (G(x)^T)_j * (G0^-1)_j,i

    size_t G0_size = (support_size + dimensions + 1)
                   * (support_size + dimensions + 1) * sizeof(T);
    size_t G0_inv_size    = G0_size;
    size_t G_x_size       = (support_size + dimensions + 1) * sizeof(T);
    size_t inv_info_size  = sizeof(int);
    size_t inv_pivot_size = sizeof(int) * (support_size + dimensions + 1);
    size_t mem_per_query_point
        = G0_size + G0_inv_size + G_x_size + inv_info_size + inv_pivot_size;
    size_t minimum_required_mem = 1000 * mem_per_query_point;

    // algorithm outline
    // for each device split
    //      1. determine if enough memory on device using some threshold
    //      2. if not, use host memory
    //      3. depending on the memory pool, set the batch size for
    //      computing the weights
    //      4. for each batch solve for the weights using W = G(x)^T *
    //      (G0^-1)[:, 1:num_support]

    sclx::array<T, 2> weights{
        interpolating_indices.shape()[0],
        query_points.size()};
    if (!devices.empty()) {
        weights.set_primary_devices(devices);
    }
    auto device_split_info = sclx::get_device_split_info(weights);

    std::vector<std::future<void>> futures;

    for (auto& split_info : device_split_info) {
        int device_id      = std::get<0>(split_info);
        size_t slice_start = std::get<1>(split_info);
        size_t slice_range = std::get<2>(split_info);

        auto device_lambda = [=]() {
            auto weights_slice
                = weights.get_range({slice_start}, {slice_start + slice_range});
            auto weights_prefetch_fut
                = weights_slice.prefetch_async(std::vector<int>{device_id});
            auto indices_slice = interpolating_indices.get_range(
                {slice_start / group_size},
                {(slice_start + slice_range) / group_size}
            );
            auto indices_prefetch_fut
                = indices_slice.prefetch_async(std::vector<int>{device_id});
            auto distances_slice = source2query_dist_squared.get_range(
                {slice_start / group_size},
                {(slice_start + slice_range) / group_size}
            );
            auto distances_prefetch_fut
                = distances_slice.prefetch_async(std::vector<int>{device_id});

            weights_prefetch_fut.get();
            indices_prefetch_fut.get();
            distances_prefetch_fut.get();

            sclx::cuda::memory_status_info device_memory_status
                = sclx::cuda::query_memory_status(device_id);

            // we don't want to overload the device, so we have a buffer
            // of 5% of the total memory
            size_t device_modified_total = 95 * device_memory_status.total / 100;
            size_t device_modified_free
                = (device_modified_total
                   <= (device_memory_status.total - device_memory_status.free))
                    ? 0
                    : device_modified_total
                          - (device_memory_status.total
                             - device_memory_status.free);

            size_t batch_size;
            if (device_modified_free > minimum_required_mem) {
                batch_size = device_modified_free / mem_per_query_point;
            } else {
                sclx::throw_exception<std::runtime_error>(
                    "Not enough memory to compute interpolation weights",
                    "naga::interpolation::"
                );
            }
            batch_size = std::min(batch_size, slice_range);

            size_t num_batches = (slice_range + batch_size - 1) / batch_size;
            sclx::array<value_type, 3> G0_storage(
                {support_size + dimensions + 1,
                 support_size + dimensions + 1,
                 batch_size},
                false
            );
            G0_storage.set_primary_devices(std::vector<int>{device_id});

            sclx::array<value_type, 3> G0_inv_storage(
                {support_size + dimensions + 1,
                 support_size + dimensions + 1,
                 batch_size},
                false
            );
            G0_inv_storage.set_primary_devices(std::vector<int>{device_id});

            sclx::array<value_type, 3> G_x_storage(
                {support_size + dimensions + 1, 1, batch_size},
                false
            );
            G_x_storage.set_primary_devices(std::vector<int>{device_id});

            for (size_t b = 0; b < num_batches; ++b) {
                auto weights_batch = weights_slice.get_range(
                    {b * batch_size},
                    {std::min((b + 1) * batch_size, slice_range)}
                );
                auto indices_batch = indices_slice.get_range(
                    {b * batch_size / group_size},
                    {std::min(
                        (b + 1) * batch_size / group_size,
                        slice_range / group_size
                    )}
                );
                auto distances_batch = distances_slice.get_range(
                    {b * batch_size / group_size},
                    {std::min(
                        (b + 1) * batch_size / group_size,
                        slice_range / group_size
                    )}
                );
                auto G0 = G0_storage.get_range({0}, {indices_batch.shape()[1]});
                auto G0_inv
                    = G0_inv_storage.get_range({0}, {indices_batch.shape()[1]});
                auto G_x
                    = G_x_storage.get_range({0}, {indices_batch.shape()[1]});

                // compute G0, only if b % group_size == 0
                // since it is the same for all points in the group
                std::future<void> G0_future;
                const auto& assign_G0 =
                    [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                        for (uint linear_idx = 0;
                             linear_idx < (support_size + dimensions + 1)
                                              * (support_size + dimensions + 1);
                             ++linear_idx) {
                            uint i
                                = linear_idx % (support_size + dimensions + 1);
                            uint j
                                = linear_idx / (support_size + dimensions + 1);
                            size_t xi_idx = indices_batch(
                                math::min(support_size - 1, i),
                                idx[0]
                            );
                            size_t xj_idx = indices_batch(
                                math::min(support_size - 1, j),
                                idx[0]
                            );
                            const value_type* xi = &source_points(0, xi_idx);
                            const value_type* xj = &source_points(0, xj_idx);
                            if (i >= support_size && j >= support_size) {
                                G0(i, j, idx[0]) = 0;
                            } else if (i >= support_size) {
                                G0(i, j, idx[0]) = (i - support_size) == 0
                                                     ? 1
                                                     : xj[i - support_size - 1];
                            } else if (j >= support_size) {
                                G0(i, j, idx[0]) = (j - support_size) == 0
                                                     ? 1
                                                     : xi[j - support_size - 1];
                            } else {
                                value_type r_squared
                                    = distance_functions::loopless::
                                        euclidean_squared<dimensions>{}(xi, xj);
                                G0(i, j, idx[0]) = shape_function(
                                    r_squared,
                                    approx_particle_spacing
                                );
                            }
                        }
                    };
                if (b % group_size == 0) {
                    auto G0_future_ = sclx::execute_kernel(
                        [&](const sclx::kernel_handler& handler) {
                            handler.launch(
                                sclx::md_range_t<1>{G0.shape()[2]},
                                G0,
                                assign_G0
                            );
                        }
                    );

                    G0_future = std::move(G0_future_);
                } else {
                    G0_future
                        = std::move(std::async(std::launch::deferred, []() {}));
                }

                const auto& assign_G_x =
                    [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                        const point_type& x = query_points
                            [idx[0] + slice_start + b * batch_size];
                        for (uint i = 0; i < support_size + dimensions + 1;
                             ++i) {
                            if (i >= support_size) {
                                G_x(i, 0, idx[0]) = (i - support_size) == 0
                                                      ? 1
                                                      : x[i - support_size - 1];
                            } else {
                                G_x(i, 0, idx[0]) = shape_function(
                                    distances_batch(i, idx[0]),
                                    approx_particle_spacing
                                );
                            }
                        }
                    };
                auto G_x_future = sclx::execute_kernel(
                    [&](const sclx::kernel_handler& handler) {
                        handler.launch(
                            sclx::md_range_t<1>{G_x.shape()[2]},
                            G_x,
                            assign_G_x
                        );
                    }
                );

                G0_future.get();
                if (b % group_size == 0) {
                    linalg::batched_matrix_inverse(G0, G0_inv, false);
                }
                G_x_future.get();

                sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
                    handler.launch(
                        sclx::md_range_t<2>{weights_batch.shape()},
                        weights_batch,
                        [=] __device__(const sclx::md_index_t<2>& idx, const auto&) {
                            value_type sum = 0;
                            for (uint i = 0; i < support_size + dimensions + 1;
                                 ++i) {
                                sum += G0_inv(i, idx[0], idx[1])
                                     * G_x(i, 0, idx[1]);
                            }
                            weights_batch(idx[0], idx[1]) = sum;
                        }
                    );
                }).get();
            }
        };

        futures.emplace_back(std::async(std::launch::async, device_lambda));
    }

    for (auto& future : futures) {
        future.get();
    }

    return weights;
}

}  // namespace naga::interpolation::detail::radial_point_method
