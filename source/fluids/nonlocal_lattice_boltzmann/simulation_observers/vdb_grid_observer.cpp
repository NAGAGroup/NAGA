// BSD 3-Clause License
//
// Copyright (c) 2024 Jack Myers
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

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <openvdb/openvdb.h>


namespace naga::fluids::nonlocal_lbm::detail {

template<class T>
struct vdb_write_data {
    const T* data;
    const bool* out_of_bounds_data;
    T grid_spacing;
    const T* domain_lb;
    const std::size_t *grid_size;
    T acoustic_normalization;
};

template<class T>
struct vdb_writer {
    static void write(const std::string& filename, const vdb_write_data<T>& write_data);
};


template<class U>
struct grid_type;

template<>
struct grid_type<float> {
    using type = openvdb::FloatGrid;
};

template<>
struct grid_type<double> {
    using type = openvdb::DoubleGrid;
};

template <class T>
void vdb_writer<T>::write(const std::string& filename, const vdb_write_data<T>& write_data) {
    auto grid_ptr = grid_type<T>::type::create();
    grid_ptr->setTransform(openvdb::math::Transform::createLinearTransform(write_data.grid_spacing));
    auto grid_accessor = grid_ptr->getAccessor();

    auto grid_values = write_data.data;
    auto grid_size   = write_data.grid_size;
    auto grid_spacing_ = write_data.grid_spacing;
    auto domain_lb_ = write_data.domain_lb;
    auto num_points = grid_size[0] * grid_size[1] * grid_size[2];
    for (size_t flat_idx = 0; flat_idx < num_points; ++flat_idx) {
        if (write_data.out_of_bounds_data[flat_idx]) {
            continue;
        }
//        if (std::abs(grid_values[flat_idx]) < 1e-2) {
//            continue;
//        }
        auto i = flat_idx % grid_size[0];
        auto j = (flat_idx / grid_size[0]) % grid_size[1];
        auto k = flat_idx / (grid_size[0] * grid_size[1]);
        auto shifted_i
            = static_cast<std::int32_t>((domain_lb_[0] + i * grid_spacing_) / grid_spacing_);
        auto shifted_j
            = static_cast<std::int32_t>((domain_lb_[1] + j * grid_spacing_) / grid_spacing_);
        auto shifted_k
            = static_cast<std::int32_t>((domain_lb_[2] + k * grid_spacing_) / grid_spacing_);

        auto grid_coord = openvdb::Coord(shifted_i, shifted_j, shifted_k);
        grid_accessor.setValue(grid_coord, grid_values[flat_idx]);
    }

    grid_ptr->setName("density");


    openvdb::io::File file(filename);

    openvdb::GridPtrVec grids;
    grids.push_back(grid_ptr);
    file.write(grids);
    file.close();
}

template struct vdb_writer<float>;
template struct vdb_writer<double>;

}