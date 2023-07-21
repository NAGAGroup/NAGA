
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

#include "../../../../../point.hpp"
#include "../../../../../mesh/experimental/closed_contour.hpp"
#include <vector>
#include <filesystem>
#include <memory>

namespace naga::experimental::fluids::nonlocal_lbm::detail {

template<uint Dimensions>
class conforming_point_cloud_impl_t;

template<uint Dimensions>
class conforming_point_cloud_t {
    static_assert(
        Dimensions == 2 || Dimensions == 3,
        "Dimensions must be 2 or 3"
    );
    static_assert(Dimensions != 3, "Not yet implemented for 3D");

  public:
    using point_t  = ::naga::point_t<double, Dimensions>;
    using normal_t = point_t;
    using index_t  = size_t;

    conforming_point_cloud_t() = default;

    using closed_contour_t = naga::experimental::mesh::closed_contour_t<double>;

    static conforming_point_cloud_t create(
        const double& approximate_spacing,
        const std::filesystem::path& domain,
        const std::vector<std::filesystem::path>& immersed_boundaries = {}
    );

    const closed_contour_t& domain() const;

    const std::vector<closed_contour_t>& immersed_boundaries() const;

    const std::vector<point_t>& points() const;

    const std::vector<normal_t>& normals() const;

    const size_t& num_bulk_points() const;

    const size_t& num_boundary_points() const;

    size_t size() const;

    bool is_boundary(const index_t &i) const;

    normal_t get_normal(const index_t &i) const;

  private:
    friend class conforming_point_cloud_impl_t<Dimensions>;
    std::shared_ptr<conforming_point_cloud_impl_t<Dimensions>> impl{};
};

extern template class conforming_point_cloud_t<2>;
}
