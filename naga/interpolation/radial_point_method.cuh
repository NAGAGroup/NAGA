
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
#include "../linalg/batched_matrix_inverse.cuh"
#include "../point_map.cuh"
#include <scalix/array.cuh>
#include <scalix/execute_kernel.cuh>

namespace naga::interpolation {

template<class PointType>
struct mq_shape_function {
    using value_type = typename point_traits<PointType>::value_type;
    constexpr static uint dimensions = point_traits<PointType>::dimensions;

    __host__ __device__ mq_shape_function(
        const value_type& q       = .5f,
        const value_type& alpha_c = .1f
    )
        : q_(q),
          alpha_c_(alpha_c) {}

    __host__ __device__ value_type operator()(
        const PointType& a,
        const PointType& b,
        const value_type& approx_particle_spacing
    ) const {
        value_type c = approx_particle_spacing * alpha_c_;
        value_type r_squared
            = distance_functions::loopless::euclidean_squared<dimensions>{}(
                a,
                b,
                {}
            );
        return math::pow(r_squared + c * c, q_);
    }

    value_type q_;
    value_type alpha_c_;
};

template<class PointMapType>
using default_shape_function
    = mq_shape_function<typename point_map_traits<PointMapType>::point_type>;

template<
    class PointMapType,
    class ShapeFunctionType = default_shape_function<PointMapType>>
class radial_point_method {
  public:
    using point_type = typename point_map_traits<PointMapType>::point_type;
    using value_type = std::decay_t<
        typename point_map_traits<PointMapType>::point_traits::value_type>;
    constexpr static uint dimensions
        = point_map_traits<PointMapType>::point_traits::dimensions;

    radial_point_method(
        sclx::array<value_type, 2>& source_points,
        sclx::array<size_t, 2>& source_indices,
        const PointMapType& query_points,
        const value_type& approx_particle_spacing,
        uint group_size                         = 1,
        const ShapeFunctionType& shape_function = ShapeFunctionType{},
        std::vector<int> devices                = {}
    )
        : group_size_(group_size),
          source_points_size_(source_points.shape()[1]),
          weights_{source_indices.shape()[0], query_points.size()},
          indices_(source_indices),
          approx_particle_spacing_(approx_particle_spacing) {
        compute_weights(
            source_points,
            source_indices,
            query_points,
            approx_particle_spacing,
            group_size,
            shape_function,
            devices
        );
    }

    void compute_weights(
        sclx::array<value_type, 2> source_points,
        sclx::array<size_t, 2> source_indices,
        const PointMapType& query_points,
        const value_type& approx_particle_spacing,
        uint group_size                         = 1,
        const ShapeFunctionType& shape_function = ShapeFunctionType{},
        std::vector<int> devices                = {}
    ) {
        size_t num_source_points = source_points.shape()[1];
        size_t num_query_points  = query_points.size();
        uint support_size        = static_cast<uint>(source_indices.shape()[0]);

        if (num_query_points % group_size_ != 0) {
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

        size_t G0_size
            = (support_size + dimensions + 1) * (support_size + dimensions + 1);
        size_t G0_inv_size          = G0_size;
        size_t G_x_size             = (support_size + dimensions + 1);
        size_t mem_per_query_point  = G0_size + G0_inv_size + G_x_size;
        size_t minimum_required_mem = 1000 * mem_per_query_point;

        // algorithm outline
        // for each device split
        //      1. determine if enough memory on device using some threshold
        //      2. if not, use host memory
        //      3. depending on the memory pool, set the batch size for
        //      computing the weights
        //      4. for each batch solve for the weights using W = G(x)^T *
        //      (G0^-1)[:, 1:num_support]

        if (!devices.empty()) {
            weights_.set_primary_devices(devices);
        }
        auto device_split_info = sclx::get_device_split_info(weights_);

        std::vector<std::future<void>> futures;

        for (auto& split_info : device_split_info) {
            int device_id      = std::get<0>(split_info);
            size_t slice_start = std::get<1>(split_info);
            size_t slice_range = std::get<2>(split_info);

            auto device_lambda = [=]() {
                auto weights_slice = weights_.get_range(
                    {slice_start},
                    {slice_start + slice_range}
                );
                auto indices_slice = indices_.get_range(
                    {slice_start / group_size_},
                    {(slice_start + slice_range) / group_size_}
                );

                sclx::cuda::memory_status_info device_memory_status
                    = sclx::cuda::query_memory_status(device_id);
                sclx::cuda::memory_status_info host_memory_status
                    = sclx::cuda::host::query_memory_status();

                size_t batch_size;
                if (device_memory_status.free > minimum_required_mem) {
                    batch_size
                        = device_memory_status.free / mem_per_query_point;
                } else if (host_memory_status.free > minimum_required_mem) {
                    batch_size = host_memory_status.free / mem_per_query_point;
                } else {
                    sclx::throw_exception<std::runtime_error>(
                        "Not enough memory to compute interpolation weights",
                        "naga::interpolation::"
                    );
                }
                // here we reduce the batch size by 40% to avoid overloading
                // the memory
                batch_size
                    = (batch_size * 6 / 10 == 0) ? 1 : batch_size * 6 / 10;
                batch_size = std::min(batch_size, slice_range);

                size_t num_batches
                    = (slice_range + batch_size - 1) / batch_size;
                sclx::array<value_type, 3> G0(
                    {support_size + dimensions + 1,
                     support_size + dimensions + 1,
                     batch_size},
                    false
                );
                G0.set_primary_devices(std::vector<int>{device_id});

                sclx::array<value_type, 3> G0_inv(
                    {support_size + dimensions + 1,
                     support_size + dimensions + 1,
                     batch_size},
                    false
                );
                G0_inv.set_primary_devices(std::vector<int>{device_id});

                sclx::array<value_type, 3> G_x(
                    {support_size + dimensions + 1, 1, batch_size},
                    false
                );
                G_x.set_primary_devices(std::vector<int>{device_id});

                for (size_t b = 0; b < num_batches; ++b) {
                    auto weights_batch = weights_slice.get_range(
                        {b * batch_size},
                        {std::min((b + 1) * batch_size, slice_range)}
                    );
                    auto indices_batch = indices_slice.get_range(
                        {b * batch_size / group_size_},
                        {std::min(
                            (b + 1) * batch_size / group_size_,
                            slice_range / group_size_
                        )}
                    );

                    // compute G0, only if b % group_size == 0
                    std::future<void> G0_future;
                    const auto& G0_dev_lambda =
                        [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                            for (uint linear_idx = 0;
                                 linear_idx
                                 < (support_size + dimensions + 1)
                                       * (support_size + dimensions + 1);
                                 ++linear_idx) {
                                uint i = linear_idx
                                       % (support_size + dimensions + 1);
                                uint j = linear_idx
                                       / (support_size + dimensions + 1);
                                size_t xi_idx = indices_batch(
                                    math::min(support_size - 1, i),
                                    idx[0]
                                );
                                size_t xj_idx = indices_batch(
                                    math::min(support_size - 1, j),
                                    idx[0]
                                );
                                const value_type* xi
                                    = &source_points(0, xi_idx);
                                const value_type* xj
                                    = &source_points(0, xj_idx);
                                if (i >= support_size && j >= support_size) {
                                    G0(i, j, idx[0]) = 0;
                                } else if (i >= support_size) {
                                    G0(i, j, idx[0])
                                        = (i - support_size) == 0
                                            ? 1
                                            : xj[i - support_size - 1];
                                } else if (j >= support_size) {
                                    G0(i, j, idx[0])
                                        = (j - support_size) == 0
                                            ? 1
                                            : xi[j - support_size - 1];
                                } else {
                                    G0(i, j, idx[0]) = shape_function(
                                        xi,
                                        xj,
                                        approx_particle_spacing
                                    );
                                }
                            }
                        };
                    if (b % group_size_ == 0) {
                        auto G0_future_ = sclx::execute_kernel(
                            [=](const sclx::kernel_handler& handler) {
                                handler.launch(
                                    sclx::md_range_t<1>{batch_size},
                                    G0,
                                    G0_dev_lambda
                                );
                            }
                        );

                        G0_future = std::move(G0_future_);
                    } else {
                        G0_future = std::move(
                            std::async(std::launch::deferred, []() {})
                        );
                    }

                    const auto& G_x_lambda =
                        [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                            const point_type& x = query_points[idx[0]];
                            for (uint i = 0; i < support_size + dimensions + 1;
                                 ++i) {
                                size_t xi_idx = indices_batch(
                                    math::min(support_size - 1, i),
                                    idx[0]
                                );
                                const value_type* xi
                                    = &source_points(0, xi_idx);
                                if (i >= support_size) {
                                    G_x(i, 0, idx[0])
                                        = (i - support_size) == 0
                                            ? 1
                                            : x[i - support_size - 1];
                                } else {
                                    G_x(i, 0, idx[0]) = shape_function(
                                        xi,
                                        x,
                                        approx_particle_spacing
                                    );
                                }
                            }
                        };
                    auto G_x_future = sclx::execute_kernel(
                        [&](const sclx::kernel_handler& handler) {
                            handler.launch(
                                sclx::md_range_t<1>{batch_size},
                                G_x,
                                G_x_lambda
                            );
                        }
                    );

                    G0_future.get();
                    if (b % group_size == 0) {
                        linalg::batched_matrix_inverse(G0, G0_inv, false);
                    }
                    G_x_future.get();

                    sclx::execute_kernel([&](const sclx::kernel_handler& handler
                                         ) {
                        handler.launch(
                            sclx::md_range_t<2>{support_size, batch_size},
                            weights_batch,
                            [=] __device__(const sclx::md_index_t<2>& idx, const auto&) {
                                value_type sum = 0;
                                for (uint i = 0;
                                     i < support_size + dimensions + 1;
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
    }

    void interpolate(
        sclx::array<value_type, 2> field,
        sclx::array<value_type, 2> destination
    ) {
        if (field.shape()[1] != source_points_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "field array must have the same number of rows as the number "
                "of source points used to construct the interpolator"
            );
        }

        if (destination.shape()[1] != weights_.shape()[1]) {
            sclx::throw_exception<std::invalid_argument>(
                "destination array must have the same number of columns as the "
                "number of points used to construct the interpolator"
            );
        }

        auto weights    = weights_;
        auto indices    = indices_;
        auto group_size = group_size_;

        sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<2>(destination.shape()),
                destination,
                [=] __device__(const sclx::md_index_t<2>& idx, const auto&) {
                    value_type sum = 0;
                    for (uint i = 0; i < weights.shape()[0]; ++i) {
                        sum += weights(i, idx[1])
                             * field(idx[0], indices(i, idx[1]));
                    }
                    destination(idx[0], idx[1]) = sum;
                }
            );
        });
    }

    void interpolate(
        sclx::array<value_type, 1> field,
        sclx::array<value_type, 1> destination
    ) {
        if (field.shape()[0] != source_points_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "field array must have the same number of rows as the number "
                "of source points used to construct the interpolator"
            );
        }

        if (destination.shape()[0] != weights_.shape()[1]) {
            sclx::throw_exception<std::invalid_argument>(
                "destination array must have the same number of columns as the "
                "number of points used to construct the interpolator"
            );
        }

        auto weights    = weights_;
        auto indices    = indices_;
        auto group_size = group_size_;

        sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<1>{destination.shape()[0]},
                destination,
                [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                    value_type sum = 0;
                    for (uint i = 0; i < weights.shape()[0]; ++i) {
                        sum += weights(i, idx[0])
                             * field(indices(i, idx[0] / group_size));
                    }
                    destination(idx[0]) = sum;
                }
            );
        }).get();
    }

  private:
    // group size allows for groups of interpolated points to use the same
    // source points. This is useful, for example, when interpolating to
    // quadrature points in a meshless method as the quadrature points can
    // usually use the nearest neighbors of their associated, integrated point
    // as their source points.
    uint group_size_;
    size_t source_points_size_;
    sclx::array<value_type, 2> weights_;
    sclx::array<size_t, 2> indices_;
    value_type approx_particle_spacing_;
};

}  // namespace naga::interpolation
