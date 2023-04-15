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
#include <mutex>
#include <scalix/algorithm/reduce_last_dim.cuh>

namespace naga::segmentation::detail {

template<class T, uint Dimensions>
__host__ void compute_bounds(
    point_view_t<T, Dimensions>& lower_bounds,
    point_view_t<T, Dimensions>& upper_bounds,
    const sclx::array<const T, 2>& points
) {
    auto lower_bounds_reduce = sclx::algorithm::reduce_last_dim(
        points,
        std::numeric_limits<T>::max(),
        sclx::algorithm::min<>()
    );
    auto upper_bounds_reduce = sclx::algorithm::reduce_last_dim(
        points,
        std::numeric_limits<T>::lowest(),
        sclx::algorithm::max<>()
    );

    for (uint i = 0; i < Dimensions; ++i) {
        lower_bounds[i] = lower_bounds_reduce[i];
        upper_bounds[i] = upper_bounds_reduce[i];
    }
}

template<class T, uint Dimensions>
class nd_cubic_segmentation_index_generator {
  public:
    static constexpr uint range_rank = 1;
    static constexpr uint index_rank = Dimensions;

    __host__ nd_cubic_segmentation_index_generator(
        const sclx::array<const T, 2>& points,
        const nd_cubic_segmentation<T, Dimensions>& segmentation
    )
        : points_(points),
          segmentation_(segmentation) {}

    __host__ __device__ sclx::md_index_t<index_rank>
    operator()(const sclx::md_index_t<range_rank>& index) const {
        return segmentation_.get_partition_index(&points_(0, index[0]));
    }

    __host__ __device__ sclx::md_range_t<range_rank> range() const {
        return {points_.shape()[1]};
    }

    __host__ __device__ sclx::md_range_t<index_rank> index_range() const {
        return static_cast<sclx::md_range_t<index_rank>>(segmentation_.shape());
    }

  private:
    sclx::array<const T, 2> points_;
    nd_cubic_segmentation<T, Dimensions> segmentation_;
};

template<class T, uint Dimensions>
__host__ void compute_partition_sizes(
    sclx::array<uint, Dimensions>& partition_sizes_,
    const sclx::shape_t<Dimensions>& segmentation_shape,
    sclx::array<const T, 2>& points_,
    const nd_cubic_segmentation<T, Dimensions>& segmentation
) {
    partition_sizes_ = sclx::array<uint, Dimensions>(segmentation_shape);
    sclx::fill(partition_sizes_, uint{0});
    auto fut = points_.prefetch_async(sclx::exec_topology::replicated);

    nd_cubic_segmentation_index_generator<T, Dimensions> index_generator(
        points_,
        segmentation
    );
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            index_generator,
            partition_sizes_,
            [=] __device__(const sclx::md_index_t<Dimensions>& index, const auto&) {
                atomicAdd(&partition_sizes_[index], 1);
            }
        );
    }).get();

    std::thread fut_thread([fut = std::move(fut)]() mutable { fut.wait(); });
    fut_thread.detach();
}

template<uint Dimensions>
__host__ void compute_index_offsets(
    sclx::array<size_t, Dimensions>& partition_index_offsets,
    const sclx::array<const uint, Dimensions>& partition_sizes
) {
    partition_index_offsets
        = sclx::array<size_t, Dimensions>(partition_sizes.shape());
    sclx::array<size_t, Dimensions> part_sizes_scan_result(
        partition_sizes.shape()
    );
    sclx::algorithm::inclusive_scan(
        partition_sizes,
        part_sizes_scan_result,
        uint{},
        sclx::algorithm::plus<>()
    );
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<Dimensions>(partition_index_offsets.shape()),
            partition_index_offsets,
            [=] __device__(const sclx::md_index_t<Dimensions>& index, const auto&) {
                size_t flat_index
                    = index.as_linear(partition_index_offsets.shape());
                if (flat_index == 0) {
                    partition_index_offsets[index] = 0;
                } else if (flat_index == partition_index_offsets.elements()) {
                    return;
                } else {
                    partition_index_offsets[index]
                        = part_sizes_scan_result[flat_index - 1];
                }
            }
        );
    }).get();
}

template<class T, uint Dimensions>
__host__ void assign_indices(
    sclx::array<size_t, 1>& indices_,
    const sclx::array<const uint, Dimensions>& partition_sizes_,
    const sclx::array<const size_t, Dimensions>& partition_index_offsets_,
    const sclx::array<const T, 2>& points_,
    const nd_cubic_segmentation<T, Dimensions>& segmentation
) {
    indices_ = sclx::array<size_t, 1>{points_.shape()[1]};

    std::vector<std::tuple<int, size_t, size_t>> indices_device_split;
    auto offset_device_split
        = sclx::get_device_split_info(partition_index_offsets_);

    // since the number of points in each partition may not be equal,
    // it means the writes to the indices array will not be evenly
    // distributed across the devices.
    //
    // To make sure that the writes are local to each device, we update
    // the memory split info according to the split info of the partition
    // and the index offset metadata.
    size_t last_index_slice_idx{};
    for (int d = 0; d < offset_device_split.size() - 1; ++d) {
        const auto& [device_id, slice_idx, slice_len] = offset_device_split[d];
        auto start_slice
            = partition_index_offsets_.get_slice(sclx::md_index_t<1>{slice_idx}
            );
        auto end_slice = partition_index_offsets_.get_slice(sclx::md_index_t<1>{
            slice_idx + slice_len});
        auto device_begin = *(start_slice.data());
        auto device_end   = *(end_slice.data());
        if (device_end == device_begin) {
            continue;
        }
        indices_device_split
            .emplace_back(device_id, device_begin, device_end - device_begin);
        if (d == offset_device_split.size() - 2) {
            last_index_slice_idx = device_end;
        }
    }
    if (last_index_slice_idx < indices_.elements()) {
        indices_device_split.emplace_back(
            std::get<0>(offset_device_split.back()),
            last_index_slice_idx,
            indices_.elements() - last_index_slice_idx
        );
    }
    indices_.set_primary_devices(indices_device_split);

    sclx::array<uint, Dimensions> indices_assigned(segmentation.shape());
    sclx::fill(indices_assigned, uint{0});

    sclx::array<size_t*, Dimensions> indices_container(segmentation.shape());
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        auto index_start_ptr = &indices_[0];
        handler.launch(
            sclx::md_range_t<Dimensions>(indices_container.shape()),
            indices_container,
            [=] __device__(
                const sclx::md_index_t<Dimensions>& index,
                const auto& info
            ) {
                indices_container[index]
                    = index_start_ptr + partition_index_offsets_[index];
            }
        );
    }).get();

    using generator_t = nd_cubic_segmentation_index_generator<T, Dimensions>;
    generator_t index_generator(points_, segmentation);

    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        // we have to call this as the indices_ pointers are stored
        // in the indices container, meaning the Scalix backend has no
        // way of knowing that indices_ is a result of the kernel
        //
        // I hope to make an array of arrays type data structure in the
        // future, which, along with properly registering as result memory
        // block, also abstracts away the memory splitting nonsense above
        indices_.unset_read_mostly();

        auto result_tup
            = sclx::make_array_tuple(indices_container, indices_assigned);

        handler.launch(
            index_generator,
            result_tup,
            [=] __device__(
                const sclx::md_index_t<Dimensions>& partition_index,
                const sclx::kernel_info<
                    generator_t::range_rank,
                    sclx::cuda::traits::kernel::default_block_shape.rank()>&
                    info
            ) {
                sclx::index_t point_index = info.global_thread_id()[0];

                // because of how set the device split info, we can assume
                // that these writes are local to the device, despite the
                // fact we don't explicitly specify indices_, contained in
                // indices_container, as a result of the kernel
                //
                // Also, it requires the unset_read_mostly() to be called
                // manually, which is not ideal. It actually led to my kernel
                // running slow on multiple devices before I figured out what
                // was going on.
                //
                // I hope future versions of Scalix will make this safer with
                // proper checks before launching the kernel
                auto old_value
                    = atomicAdd(&indices_assigned[partition_index], 1);
                indices_container[partition_index][old_value] = point_index;
            }
        );

        indices_.set_read_mostly();
    }).get();
}

}  // namespace naga::detail
