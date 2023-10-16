
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
#include <scalix/assign_array.cuh>

namespace naga::nonlocal_calculus {

REGISTER_SCALIX_KERNEL_TAG(apply_divergence);

template<class T, uint Dimensions>
class divergence_operator {
  public:
    divergence_operator() = default;

    friend class operator_builder<T, Dimensions>;

    using default_field_map = default_point_map<T, Dimensions>;

    static divergence_operator create(const sclx::array<T, 2>& domain) {
        // the following code was designed assuming that all
        // devices have the same amount of memory
        //
        // it could work with different amounts of memory, but it's untested

        divergence_operator result;

        result.weights_ = sclx::array<T, 3>{
            Dimensions,
            detail::num_interp_support,
            domain.shape()[1]};
        result.support_indices_ = sclx::array<size_t, 2>{
            detail::num_interp_support,
            domain.shape()[1]};

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
            = operator_builder<T, Dimensions>::get_scratchpad_size(domain.shape(
            )[1]);

        // indices size excluded from self_scratchpad_size because it's
        // included in the builder_scratchpad_size
        size_t max_domain_copy_size = domain.elements() * sizeof(T);
        size_t self_scratchpad_size = sizeof(T) * Dimensions
                                        * detail::num_interp_support
                                        * domain.shape()[1]
                                    + max_domain_copy_size;

        size_t total_scratchpad_size
            = builder_scratchpad_size + self_scratchpad_size;

        size_t batch_scratch_pad_size
            = std::min(total_available_mem, total_scratchpad_size);
        size_t batch_size = domain.shape()[1] * batch_scratch_pad_size
                          / total_scratchpad_size;

        for (size_t i = 0; i < domain.shape()[1]; i += batch_size) {
            size_t batch_end = std::min(i + batch_size, domain.shape()[1]);
            sclx::array<T, 2> batch_domain{domain.shape()[0], batch_end - i};
            auto domain_slice = domain.get_range({i}, {batch_end});
            sclx::assign_array(domain_slice, batch_domain);

            operator_builder<T, Dimensions> builder(domain, batch_domain);
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
        return result;
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

            constexpr uint num_nodes_per_block = 128;
            constexpr uint prefetch_distance   = 16;
            static_assert(
                prefetch_distance <= detail::num_interp_support,
                "prefetch_distance must be less than or equal to "
                "detail::num_interp_support"
            );

            sclx::shape_t<1> block_shape{num_nodes_per_block};
            size_t problem_size = result.elements();
            size_t total_threads
                = std::min(max_total_threads_per_dev, problem_size);
            size_t grid_size = total_threads / (block_shape.elements());

            sclx::local_array<T, 1> local_divergence{handler, block_shape};
            sclx::local_array<sclx::index_t, 2> interp_indices_prefetch{
                handler,
                sclx::shape_t<2>{prefetch_distance, num_nodes_per_block}};
            sclx::local_array<T, 3> weights_prefetch{
                handler,
                sclx::shape_t<3>{
                    Dimensions,
                    prefetch_distance,
                    num_nodes_per_block}};
            auto field_prefetch = field.allocate_prefetch_array(
                handler,
                sclx::shape_t<2>{prefetch_distance, num_nodes_per_block}
            );

            auto& support_indices = support_indices_;
            auto& weights         = weights_;

            handler.launch<apply_divergence>(
                sclx::md_range_t<1>{result.elements()},
                result,
                [=] __device__(
                    const sclx::md_index_t<1>& index,
                    const sclx::kernel_info<1, 1>& info
                ) mutable {
                    const auto& local_thread_id = info.local_thread_id()[0];
                    uint pipeline_count         = 0;
                    for (uint p = 0; p < prefetch_distance; ++p) {
                        __pipeline_memcpy_async(
                            &interp_indices_prefetch(p, local_thread_id),
                            &support_indices(p, index[0]),
                            sizeof(sclx::index_t)
                        );
                        __pipeline_commit();
                        ++pipeline_count;
                    }
                    for (uint p = 0; p < prefetch_distance; ++p) {
                        for (int d = 0; d < Dimensions; ++d) {
                            __pipeline_memcpy_async(
                                &weights_prefetch(d, p, local_thread_id),
                                &weights(d, p, index[0]),
                                sizeof(T)
                            );
                            __pipeline_commit();
                            ++pipeline_count;
                        }
                    }
                    for (uint p = 0; p < prefetch_distance; ++p) {
                        __pipeline_wait_prior(--pipeline_count);
                    }
                    for (uint p = 0; p < prefetch_distance; ++p) {
                        field.prefetch_field_entry(
                            interp_indices_prefetch(p, local_thread_id),
                            sclx::md_index_t<2>{p, local_thread_id},
                            field_prefetch,
                            pipeline_count
                        );
                    }

                    local_divergence[local_thread_id] = 0;

                    for (uint n = 0; n < detail::num_interp_support; ++n) {
                        if (n + prefetch_distance
                            < detail::num_interp_support) {
                            __pipeline_memcpy_async(
                                &interp_indices_prefetch(
                                    n % prefetch_distance,
                                    local_thread_id
                                ),
                                &support_indices(
                                    n + prefetch_distance,
                                    index[0]
                                ),
                                sizeof(sclx::index_t)
                            );
                            __pipeline_commit();
                            ++pipeline_count;
                        }
                        T indexed_weights[Dimensions];
                        for (uint d = 0; d < Dimensions; ++d) {
                            __pipeline_wait_prior(--pipeline_count);
                            indexed_weights[d] = weights_prefetch(
                                d,
                                n % prefetch_distance,
                                local_thread_id
                            );
                        }
                        if (n + prefetch_distance
                            < detail::num_interp_support) {
                            for (uint d = 0; d < Dimensions; ++d) {
                                __pipeline_memcpy_async(
                                    &weights_prefetch(
                                        d,
                                        n % prefetch_distance,
                                        local_thread_id
                                    ),
                                    &weights(
                                        d,
                                        n + prefetch_distance,
                                        index[0]
                                    ),
                                    sizeof(T)
                                );
                                __pipeline_commit();
                                ++pipeline_count;
                            }
                        }

                        const auto& field_value = field.index_prefetch_array(
                            sclx::md_index_t<2>{
                                n % prefetch_distance,
                                local_thread_id},
                            field_prefetch,
                            pipeline_count
                        );
                        for (uint d = 0; d < Dimensions; ++d) {
                            local_divergence[local_thread_id]
                                += indexed_weights[d]
                                 * (field_value[d] - centering_offset);
                        }

                        if (n + prefetch_distance
                            < detail::num_interp_support) {
                            __pipeline_wait_prior(--pipeline_count);
                            field.prefetch_field_entry(
                                interp_indices_prefetch(
                                    n % prefetch_distance,
                                    local_thread_id
                                ),
                                sclx::md_index_t<2>{
                                    n % prefetch_distance,
                                    local_thread_id},
                                field_prefetch,
                                pipeline_count
                            );
                        }
                    }

                    result[index[0]] = local_divergence[local_thread_id];
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
            domain.shape()[1]};
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
};

}  // namespace naga::nonlocal_calculus

#include "detail/divergence.inl"