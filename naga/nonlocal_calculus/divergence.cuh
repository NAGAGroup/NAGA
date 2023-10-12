
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
#include <cuda_pipeline.h>
#include <scalix/assign_array.cuh>

namespace naga::nonlocal_calculus {

REGISTER_SCALIX_KERNEL_TAG(apply_divergence);

namespace detail {
template<auto Start, auto End, class Loop>
__device__ void divergence_unrolled_loop(Loop&& f) {
    static_assert(Start <= End, "Start must be less than or equal to End.");
    static_assert(
        std::is_integral_v<decltype(Start)>,
        "Start must be integral."
    );
    static_assert(std::is_integral_v<decltype(End)>, "End must be integral.");
    if constexpr (Start == End) {
        return;
    } else {
        f(Start);
        divergence_unrolled_loop<Start + 1, End, Loop>(std::forward<Loop>(f));
    }
}
}  // namespace detail

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

            constexpr uint num_solution_nodes_per_block = 8;
            constexpr uint num_threads
                = num_solution_nodes_per_block * detail::num_interp_support;
            constexpr uint prefetch_distance = 2;

            sclx::shape_t<2> block_shape{
                detail::num_interp_support,
                num_threads / detail::num_interp_support};
            size_t problem_size
                = result.elements() * detail::num_interp_support;
            size_t total_threads
                = std::min(max_total_threads_per_dev, problem_size);
            size_t grid_size
                = total_threads / block_shape.elements() / prefetch_distance;

            sclx::local_array<T, 2> divergence_tmp{handler, block_shape};

            sclx::local_array<T, 4> weights_prefetch{
                handler,
                sclx::shape_t<4>{
                    Dimensions,
                    detail::num_interp_support,
                    num_solution_nodes_per_block,
                    prefetch_distance}};

            sclx::local_array<T, 4> field_prefetch{
                handler,
                sclx::shape_t<4>{
                    field_dimensions,
                    detail::num_interp_support,
                    num_solution_nodes_per_block,
                    prefetch_distance}};

            sclx::local_array<T, 2> pipeline_wait_prior{
                handler,
                block_shape};

            auto& support_indices = support_indices_;
            auto& weights         = weights_;

            handler.launch<apply_divergence>(
                sclx::md_range_t<2>{
                    support_indices_.shape()[0],
                    result.elements()},
                result,
                [=] __device__(
                    const sclx::md_index_t<2>& index,
                    const sclx::kernel_info<2, 2>& info
                ) mutable {
                    const auto& local_thread_id = info.local_thread_id();
                    const auto& linear_index    = [&]() {
                        return index.as_linear(info.global_range());
                    };
                    const auto& start_idx = [&]() {
                        return info.start_index().as_linear(info.global_range()
                        );
                    };
                    if (info.stride_count() == 0) {
                        pipeline_wait_prior[local_thread_id] = 0;
                        detail::divergence_unrolled_loop<0, prefetch_distance>(
                            [&](const int& i) {
                                if (linear_index() + i * info.grid_stride()
                                    >= start_idx()
                                           + info.device_range().elements()) {
                                    return;
                                }

                                detail::divergence_unrolled_loop<0, Dimensions>(
                                    [&](const int& d) {
                                        __pipeline_memcpy_async(
                                            &weights_prefetch(
                                                d,
                                                local_thread_id[0],
                                                local_thread_id[1],
                                                i
                                            ),
                                            &weights(
                                                d,
                                                local_thread_id[0],
                                                (linear_index()
                                                 + i * info.grid_stride())
                                                    / detail::num_interp_support
                                            ),
                                            sizeof(T)
                                        );
                                        __pipeline_commit();
                                        ++pipeline_wait_prior[local_thread_id];
                                    }
                                );

                                auto support_index = support_indices(
                                    local_thread_id[0],
                                    (linear_index() + i * info.grid_stride())
                                        / detail::num_interp_support
                                );
                                detail::divergence_unrolled_loop<
                                    0,
                                    field_dimensions>([&](const int& d) {
                                    __pipeline_memcpy_async(
                                        &field_prefetch(
                                            d,
                                            local_thread_id[0],
                                            local_thread_id[1],
                                            i
                                        ),
                                        field.pointer_to(support_index) + d,
                                        sizeof(T)
                                    );
                                    __pipeline_commit();
                                    ++pipeline_wait_prior[local_thread_id];
                                });
                            }
                        );
                    }

                    T local_weights[Dimensions];
                    T support_field_value[field_dimensions];
                    detail::divergence_unrolled_loop<0, Dimensions>(
                        [&](const int& d) {
                            __pipeline_wait_prior(
                                --pipeline_wait_prior[local_thread_id]
                            );
                            local_weights[d] = weights_prefetch(
                                d,
                                local_thread_id[0],
                                local_thread_id[1],
                                info.stride_count() % prefetch_distance
                            );
                        }
                    );
                    detail::divergence_unrolled_loop<0, field_dimensions>(
                        [&](const int& d) {
                            __pipeline_wait_prior(
                                --pipeline_wait_prior[local_thread_id]
                            );
                            support_field_value[d] = field_prefetch(
                                d,
                                local_thread_id[0],
                                local_thread_id[1],
                                info.stride_count() % prefetch_distance
                            );
                        }
                    );

                    if (linear_index() + prefetch_distance * info.grid_stride()
                        < start_idx() + info.device_range().elements()) {
                        detail::divergence_unrolled_loop<0, Dimensions>(
                            [&](const int& d) {
                                __pipeline_memcpy_async(
                                    &weights_prefetch(
                                        d,
                                        local_thread_id[0],
                                        local_thread_id[1],
                                        info.stride_count() % prefetch_distance
                                    ),
                                    &weights(
                                        d,
                                        local_thread_id[0],
                                        (linear_index()
                                         + prefetch_distance
                                               * info.grid_stride())
                                            / detail::num_interp_support
                                    ),
                                    sizeof(T)
                                );
                                __pipeline_commit();
                                ++pipeline_wait_prior[local_thread_id];
                            }
                        );

                        auto support_index = support_indices(
                            local_thread_id[0],
                            (linear_index()
                             + prefetch_distance * info.grid_stride())
                                / detail::num_interp_support
                        );
                        detail::divergence_unrolled_loop<0, field_dimensions>(
                            [&](const int& d) {
                                __pipeline_memcpy_async(
                                    &field_prefetch(
                                        d,
                                        local_thread_id[0],
                                        local_thread_id[1],
                                        info.stride_count() % prefetch_distance
                                    ),
                                    field.pointer_to(support_index) + d,
                                    sizeof(T)
                                );
                                __pipeline_commit();
                                ++pipeline_wait_prior[local_thread_id];
                            }
                        );
                    }

                    if (local_thread_id[0] == 0) {
                        __pipeline_wait_prior(
                            --pipeline_wait_prior[local_thread_id]
                        );
                    }
                    {
                        T& local_div = divergence_tmp[local_thread_id];
                        T div        = 0;
                        detail::divergence_unrolled_loop<0, Dimensions>(
                            [&](const int& d) {
                                div += local_weights[d]
                                     * (support_field_value[d]
                                        - centering_offset);
                            }
                        );
                        local_div = div;
                    }

                    auto& shared_result = divergence_tmp;
                    for (uint s = 1; s < block_shape[0]; s *= 2) {
                        uint idx
                            = 2 * s * static_cast<uint>(local_thread_id[0]);
                        handler.syncthreads();

                        if (idx < block_shape[0]) {
                            shared_result(idx, local_thread_id[1])
                                += shared_result(idx + s, local_thread_id[1]);
                        }
                    }

                    if (local_thread_id[0] == 0) {
                        result[index[1]] = shared_result(0, local_thread_id[1]);
                    }
                    handler.syncthreads();
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