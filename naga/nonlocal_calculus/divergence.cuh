
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

#include "detail/divergence.cuh"
#include <cuda/barrier>
#include <scalix/algorithm/elementwise_reduce.cuh>
#include <scalix/assign_array.cuh>

namespace naga::nonlocal_calculus {

REGISTER_SCALIX_KERNEL_TAG(apply_divergence);

template<class T, uint Dimensions>
class divergence_operator {
  public:
    divergence_operator() = default;

    friend class operator_builder<T, Dimensions>;

    using default_field_map = default_point_map<T, Dimensions>;

    static divergence_operator create(
        const sclx::array<T, 2>& domain,
        const sclx::array<T, 2>& query_points,
        size_t override_query_size = 0
    ) {
        // the following code was designed assuming that all
        // devices have the same amount of memory
        //
        // it could work with different amounts of memory, but it's untested
        override_query_size = override_query_size == 0 ? query_points.shape()[1] : override_query_size;
        if (override_query_size < query_points.shape()[1]) {
            sclx::throw_exception<std::invalid_argument>(
                "override_query_size must be at least as large as the number "
                "of query points.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }

        divergence_operator result;

        auto total_weights = sclx::zeros<T, 3>({
            Dimensions,
            detail::num_interp_support,
            override_query_size});
        result.weights_ = total_weights.get_range({0}, {query_points.shape()[1]});
        auto total_support_indices = sclx::zeros<size_t, 2>({
            detail::num_interp_support,
            override_query_size
        });
        result.support_indices_ = total_support_indices.get_range({0}, {query_points.shape()[1]});

        size_t total_available_mem = 0;
        int device_count           = sclx::cuda::traits::device_count();
        for (int d = 0; d < device_count; ++d) {
            sclx::cuda::memory_status_info info
                = sclx::cuda::query_memory_status(d);
            size_t allocated_mem = info.total - info.free;
            size_t reduced_total_mem
                = info.total * 95 / 100;  // reduced to be safe
            if (reduced_total_mem < allocated_mem) {
                sclx::throw_exception<std::runtime_error>(
                    "Not enough memory to build divergence operator on device "
                        + std::to_string(d),
                    "naga::nonlocal_calculus::divergence_operator::"
                );
            }
            total_available_mem += reduced_total_mem - allocated_mem;
        }

        size_t builder_scratchpad_size
            = operator_builder<T, Dimensions>::get_scratchpad_size(
                query_points.shape()[1]
            );

        // indices size excluded from self_scratchpad_size because it's
        // included in the builder_scratchpad_size
        size_t max_domain_copy_size = query_points.elements() * sizeof(T);
        size_t self_scratchpad_size = sizeof(T) * Dimensions
                                        * detail::num_interp_support
                                        * query_points.shape()[1]
                                    + max_domain_copy_size;

        size_t total_scratchpad_size
            = builder_scratchpad_size + self_scratchpad_size;
        size_t batch_scratch_pad_size
            = total_scratchpad_size / query_points.shape()[1];
        auto minimum_required_mem = 16384 * batch_scratch_pad_size;

        size_t batch_size;
        if (total_available_mem > minimum_required_mem) {
            batch_size = std::min(
                query_points.shape()[1],
                minimum_required_mem / batch_scratch_pad_size
            );
        } else {
            sclx::throw_exception<std::runtime_error>(
                "Not enough memory to compute divergence operator.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }

        sclx::array<T, 2> batch_domain_total{
            query_points.shape()[0],
            batch_size
        };
        for (size_t i = 0; i < query_points.shape()[1]; i += batch_size) {
            size_t batch_end
                = std::min(i + batch_size, query_points.shape()[1]);
            auto batch_query
                = batch_domain_total.get_range({0}, {batch_end - i});
            auto query_slice = query_points.get_range({i}, {batch_end});
            sclx::assign_array(query_slice, batch_query);

            operator_builder<T, Dimensions> builder(domain, batch_query);
            divergence_operator op_batch
                = builder.template create<detail::divergence_operator_type>();

            auto weights_slice = result.weights_.get_range({i}, {batch_end});
            sclx::assign_array(op_batch.weights_, weights_slice);

            auto support_indices_slice
                = result.support_indices_.get_range({i}, {batch_end});
            sclx::assign_array(
                op_batch.support_indices_,
                support_indices_slice
            );
        }

        result.weights_ = total_weights;
        result.support_indices_ = total_support_indices;

        return result;
    }

    static divergence_operator create(const sclx::array<T, 2>& domain) {
        return create(domain, domain);
    }

    void set_subspace_to(
        const divergence_operator& subspace_op,
        size_t start,
        size_t end
    ) {
        auto dst_weights = weights_.get_range({start}, {end});
        auto dst_indices = support_indices_.get_range({start}, {end});
        sclx::assign_array(subspace_op.weights_, dst_weights);
        sclx::assign_array(subspace_op.support_indices_, dst_indices);
    }

    template<class FieldMap = default_field_map>
    void apply(
        const FieldMap& field,
        const sclx::array<T, 1>& result,
        const T& centering_offset = T(0)
    ) const {
        using field_type                = typename FieldMap::point_type;
        constexpr uint field_dimensions = field_type::dimensions;
        static_assert(
            field_dimensions == Dimensions,
            "Field map has incorrect dimensions."
        );
        if (result.elements() != weights_.shape()[2]) {
            sclx::throw_exception<std::invalid_argument>(
                "Result array has incorrect shape.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }
        if (result.elements() != field.size()) {
            sclx::throw_exception<std::invalid_argument>(
                "Result array has incorrect size.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }
        if (!cusparse_enabled_) {
            apply_default(field, result, centering_offset);
        } else {
            apply_cusparse(field, result, centering_offset);
        }
    }

    void apply_v2(
        const T (&const_velocity)[Dimensions],
        const sclx::array<T, 1>& scalar_field,
        const sclx::array<T, 1>& result,
        const T& centering_offset = T(0)
    ) const {
        if (result.elements() != weights_.shape()[2]) {
            sclx::throw_exception<std::invalid_argument>(
                "Result array has incorrect shape.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }
        if (result.elements() != scalar_field.elements()) {
            sclx::throw_exception<std::invalid_argument>(
                "Result array has incorrect size.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }
        if (!cusparse_enabled_) {
            sclx::throw_exception<std::invalid_argument>(
                "This method is only supported when the cusparse algorithm is "
                "enabled.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        } else {
            apply_cusparse_v2(
                const_velocity,
                scalar_field,
                result,
                centering_offset
            );
        }
    }

    void apply_cusparse_v2(
        const T (&const_velocity)[Dimensions],
        sclx::array<T, 1> scalar_field,
        sclx::array<T, 1> result,
        const T& centering_offset = T(0)
    ) const {
        sclx::assign_array(scalar_field, result).get();
        sclx::algorithm::transform(
            result,
            result,
            centering_offset,
            [=] __device__(const T& val, const T& centering_offset) {
                return val - centering_offset;
            }
        ).get();
        std::vector<std::future<void>> div_futures;
        for (int d = 0; d < Dimensions; ++d) {
            auto fut = cusparse_desc_->mat_mult(
                const_velocity[d],
                0.,
                cusparse_desc_->div_mats[d],
                scalar_field,
                cusparse_desc_->scratchpad[d]
            );
            div_futures.push_back(std::move(fut));
        }

        std::vector<std::future<void>> scale_futures;
        for (int d = 0; d < Dimensions; ++d) {
            div_futures[d].get();
            auto fut = sclx::algorithm::transform(
                cusparse_desc_->scratchpad[d],
                cusparse_desc_->scratchpad[d],
                const_velocity[d],
                [=] __device__(const T& val, const T& const_velocity) {
                    return val * const_velocity;
                }
            );
            scale_futures.push_back(std::move(fut));
        }

        for (auto& fut : scale_futures) {
            fut.get();
        }

        if constexpr (Dimensions == 2) {
            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>{},
                result,
                cusparse_desc_->scratchpad[0],
                cusparse_desc_->scratchpad[1]
            )
                .get();
        } else {
            return sclx::algorithm::elementwise_reduce(
                       sclx::algorithm::plus<>{},
                       result,
                       cusparse_desc_->scratchpad[0],
                       cusparse_desc_->scratchpad[1],
                       cusparse_desc_->scratchpad[2]
            )
                .get();
        }
    }

    [[nodiscard]] sclx::array<const T, 3> weights() const { return weights_; }

    [[nodiscard]] sclx::array<const sclx::index_t, 2> support_indices() const {
        return support_indices_;
    }

    [[nodiscard]] sclx::array<sclx::index_t, 2> support_indices() {
        return support_indices_;
    }

    template<class FieldMap = default_field_map>
    void apply_cusparse(
        const FieldMap& field,
        const sclx::array<T, 1>& result,
        const T& centering_offset = T(0)
    ) const {
        using field_type = typename FieldMap::point_type;
        static_assert(
            field_type::dimensions == Dimensions,
            "Field map has incorrect dimensions."
        );
        std::vector<std::future<void>> futures;
        for (int d = 0; d < Dimensions; ++d) {
            auto fut = sclx::execute_kernel([&,
                                             d](sclx::kernel_handler& handler) {
                auto& scratchpad = cusparse_desc_->scratchpad;
                handler.launch(
                    sclx::md_range_t<1>{cusparse_desc_->scratchpad[0].shape()},
                    sclx::array_list<T, 1, Dimensions>{
                        cusparse_desc_->scratchpad
                    },
                    [=] __device__(const sclx::md_index_t<1>& index, const auto&) {
                        scratchpad[d][index]
                            = field[index[0]][d] - centering_offset;
                    }
                );
            });
            futures.push_back(std::move(fut));
        }

        futures[0].get();
        cusparse_desc_
            ->mat_mult(
                1.,
                0.,
                cusparse_desc_->div_mats[0],
                cusparse_desc_->scratchpad[0],
                result
            )
            .get();
        for (uint d = 1; d < Dimensions; ++d) {
            futures[d].get();
            cusparse_desc_
                ->mat_mult(
                    1.,
                    1.,
                    cusparse_desc_->div_mats[d],
                    cusparse_desc_->scratchpad[d],
                    result
                )
                .get();
        }
    }

    template<class FieldMap = default_field_map>
    void apply_default(
        const FieldMap& field,
        const sclx::array<T, 1>& result,
        const T& centering_offset = T(0)
    ) const {
        using field_type = typename FieldMap::point_type;
        static_assert(
            field_type::dimensions == Dimensions,
            "Field map has incorrect dimensions."
        );

        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            auto max_total_threads_per_dev = std::numeric_limits<size_t>::min();
            int device_count               = sclx::cuda::traits::device_count();
            int current_device = sclx::cuda::traits::current_device();
            for (int d = 0; d < device_count; ++d) {
                sclx::cuda::set_device(d);
                cudaDeviceProp props{};
                cudaGetDeviceProperties(&props, d);
                size_t total_threads_per_dev = props.maxThreadsPerMultiProcessor
                                             * props.multiProcessorCount;
                max_total_threads_per_dev = std::max(
                    max_total_threads_per_dev,
                    total_threads_per_dev
                );
            }
            sclx::cuda::set_device(current_device);

            constexpr uint num_nodes_per_block = 4;
            constexpr uint num_grid_strides    = 4;

            sclx::shape_t<2> block_shape{
                detail::num_interp_support,
                num_nodes_per_block
            };
            size_t problem_size = result.elements() * detail::num_interp_support
                                / num_grid_strides;
            size_t total_threads
                = std::min(max_total_threads_per_dev, problem_size);
            size_t grid_size = total_threads / (block_shape.elements());

            sclx::local_array<T, 2> divergence_tmp{handler, block_shape};

            auto& support_indices = support_indices_;
            auto& weights         = weights_;

            handler.launch<apply_divergence>(
                sclx::md_range_t<2>{
                    support_indices_.shape()[0],
                    result.elements()
                },
                result,
                [=] __device__(
                    const sclx::md_index_t<2>& index,
                    const sclx::kernel_info<2, 2>& info
                ) mutable {
                    field_type support_point_field_value
                        = field[support_indices[index]];
                    auto local_thread_id = info.local_thread_id();

                    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
                    __shared__ char barrier_storage
                        [sizeof(barrier_t) * num_nodes_per_block];
                    auto* barriers
                        = reinterpret_cast<barrier_t*>(barrier_storage);

                    if (info.stride_count() == 0) {
                        if (local_thread_id[0] == 0) {
                            init(
                                &barriers[local_thread_id[1]],
                                detail::num_interp_support
                            );
                        }
                        handler.syncthreads();
                    }

                    {
                        T* local_weights = &weights(0, index[0], index[1]);
                        T& local_div     = divergence_tmp[local_thread_id];
                        {
                            T div = 0;
                            for (uint d = 0; d < Dimensions; ++d) {
                                div += local_weights[d]
                                     * (support_point_field_value[d]
                                        - centering_offset);
                            }
                            local_div = div;
                        }
                    }

                    T* shared_result = &divergence_tmp(0, local_thread_id[1]);
                    for (uint s = 1; s < detail::num_interp_support; s *= 2) {
                        uint idx
                            = 2 * s * static_cast<uint>(local_thread_id[0]);
                        barriers[local_thread_id[1]].arrive_and_wait();

                        if (idx + s < detail::num_interp_support) {
                            shared_result[idx] += shared_result[idx + s];
                        }
                    }
                    barriers[local_thread_id[1]].arrive_and_wait();

                    if (local_thread_id[0] == 0) {
                        result[index[1]] = shared_result[0];
                    }
                },
                block_shape,
                grid_size
            );
        }).get();
    }

    void apply(
        const sclx::array<T, 2>& field,
        const sclx::array<T, 1>& result,
        const T& centering_offset = T(0)
    ) const {
        apply(default_field_map{field}, result, centering_offset);
    }

    template<class Archive>
    void save(Archive& ar) const {
        sclx::serialize_array(ar, weights_);
        sclx::serialize_array(ar, support_indices_);
    }

    template<class Archive>
    void load(Archive& ar) {
        sclx::deserialize_array(ar, weights_);
        sclx::deserialize_array(ar, support_indices_);
    }

    void enable_cusparse_algorithm() {
        if (sclx::cuda::traits::device_count() > 1) {
            std::cerr
                << "Warning: cusparse algorithm only supports single GPU, \n"
                   "but multiple GPUs are available. For any arrays \n"
                   "distributed across multiple devices, the algorithm will \n"
                   "perform less efficiently than the default algorithm which "
                   "\n"
                   "distributes the computation across all available "
                   "devices.\n";
        }

        if (cusparse_desc_ == nullptr) {
            sclx::array<T, 2> ndim_weights(support_indices_.shape());
            cusparse_desc_ = std::make_shared<cusparse_algo_desc>();
            for (int d = 0; d < Dimensions; ++d) {
                sclx::execute_kernel([&](sclx::kernel_handler& handler) {
                    auto& weights = weights_;
                    handler.launch(
                        sclx::md_range_t<2>{ndim_weights.shape()},
                        ndim_weights,
                        [=] __device__(const sclx::md_index_t<2>& index, const auto&) {
                            ndim_weights[index]
                                = weights(d, index[0], index[1]);
                        }
                    );
                }).get();
                cusparse_desc_->div_mats[d]
                    = matrix_type::create_from_index_stencil(
                        support_indices_,
                        ndim_weights
                    );
            }
            for (auto& sarr : cusparse_desc_->scratchpad) {
                sarr = sclx::array<T, 1>(
                    sclx::shape_t<1>{support_indices_.shape()[1]}
                );
            }
            cusparse_enabled_ = true;
        }
    }

    void disable_cusparse_algorithm() {
        cusparse_desc_    = nullptr;
        cusparse_enabled_ = false;
    }

    bool cusparse_algorithm_enabled() const { return cusparse_enabled_; }

    template<class TO, uint DimensionsO>
    friend class advection_operator;

    divergence_operator slice(size_t new_size) const {
        divergence_operator result;
        result                  = *this;
        result.weights_         = weights_.get_range({0}, {new_size});
        result.support_indices_ = support_indices_.get_range({0}, {new_size});
        result.cusparse_desc_.reset();
        result.cusparse_enabled_ = false;
        return result;
    }

  private:
    static divergence_operator create(
        const sclx::array<T, 2>& domain,
        const sclx::array<sclx::index_t, 2>& support_indices,
        const sclx::array<T, 2>& quad_interp_weights,
        const sclx::array<T, 1>& interaction_radii
    ) {
        divergence_operator op;
        op.weights_ = sclx::array<T, 3>{
            Dimensions,
            support_indices.shape()[0],
            domain.shape()[1]
        };
        op.support_indices_ = support_indices;

        detail::compute_divergence_weights<T, Dimensions>(
            op.weights_,
            domain,
            interaction_radii,
            quad_interp_weights,
            support_indices
        );

        return op;
    }

    sclx::array<T, 3> weights_;
    sclx::array<sclx::index_t, 2> support_indices_;
    using matrix_type
        = naga::linalg::matrix<T, naga::linalg::storage_type::sparse_csr>;
    using vector_type
        = naga::linalg::vector<T, naga::linalg::storage_type::dense>;
    struct cusparse_algo_desc {
        matrix_type div_mats[Dimensions];
        sclx::array<T, 1> scratchpad[Dimensions];
        naga::linalg::matrix_mult<matrix_type, vector_type, vector_type>
            mat_mult;
    };
    std::shared_ptr<cusparse_algo_desc> cusparse_desc_{nullptr};
    bool cusparse_enabled_{false};
};

}  // namespace naga::nonlocal_calculus

#include "detail/divergence.inl"