
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

#include "../../../../../compatability.h"
#include "../../../../../mesh/experimental/closed_contour.hpp"
#include "../../../../../point.hpp"
#include <filesystem>
#include <memory>
#include <vector>

namespace naga::experimental::fluids::nonlocal_lbm::detail {


constexpr float min_bound_dist_scale_2d             = 1.f;
constexpr float min_bound_dist_scaled_ghost_node_3d = -1.1f;
constexpr float min_bound_dist_scale_3d             = .01f;
constexpr float max_bound_dist_scale_3d             = 1.1f;

template<class T, uint Dimensions>
class conforming_point_cloud_impl_t;

template<class T, uint Dimensions>
struct input_domain_data_t {};

template<class T>
struct input_domain_data_t<T, 2> {
    using type = naga::experimental::mesh::closed_contour_t<T>;
};

template<class T>
struct input_domain_data_t<T, 3> {
    using type = naga::experimental::mesh::triangular_mesh_t<T>;
};

template<class T, uint Dimensions>
class conforming_point_cloud_t {
    static_assert(
        Dimensions == 2 || Dimensions == 3,
        "Dimensions must be 2 or 3"
    );
};

template<class T>
class conforming_point_cloud_t<T, 2> {
  public:
    using value_type = T;
    using point_t  = ::naga::point_t<value_type, 2>;
    using normal_t = point_t;
    using index_t  = size_t;

    conforming_point_cloud_t() = default;

    using input_domain_data_t = typename input_domain_data_t<value_type, 2>::type;

    static conforming_point_cloud_t create(
        const value_type& approximate_spacing,
        const std::filesystem::path& domain,
        const std::vector<std::filesystem::path>& immersed_boundaries = {}
    );

    const input_domain_data_t& domain() const;

    const std::vector<input_domain_data_t>& immersed_boundaries() const;

    const std::vector<point_t>& points() const;

    const std::vector<normal_t>& normals() const;

    const size_t& num_bulk_points() const;

    const size_t& num_boundary_points() const;

    size_t size() const;

    bool is_boundary(const index_t& i) const;

    normal_t get_normal(const index_t& i) const;

  private:
    friend class conforming_point_cloud_impl_t<value_type, 2>;
    std::shared_ptr<conforming_point_cloud_impl_t<value_type, 2>> impl{};
};

// this specialization sucks because it's interface is different
// from the above template
//
// if we followed the same interface, we would do a lot of extra
// computation in the domain provider
//
// really what we should do at some point is make the above template,
// and subsequently the 2D specialization, conform to the interface
// below

template <class T>
struct mesh_placement {
    naga::point_t<T, 3> location{{0, 0, 0}};
    naga::point_t<T, 3> rotation{{0, 0, 0}};
};

template<class T>
class conforming_point_cloud_t<T, 3> {

  public:
    static constexpr uint dimensions = 3;
    using value_type                 = T;
    using point_t                    = ::naga::point_t<value_type, dimensions>;
    using normal_t                   = point_t;
    using index_t                    = size_t;

    conforming_point_cloud_t() = default;

    using input_domain_data_t = typename input_domain_data_t<value_type, dimensions>::type;

    static conforming_point_cloud_t create(
        const value_type& approximate_spacing,
        const std::filesystem::path& domain,
        const std::vector<std::filesystem::path>& immersed_boundaries = {},
        const std::vector<mesh_placement<value_type>>& immersed_mesh_placements     = {}
    );

    const std::vector<point_t>& bulk_points() const;

    const std::vector<value_type>& bulk_to_boundary_distances() const;

    const std::vector<point_t>& boundary_points() const;

    const std::vector<normal_t>& boundary_normals() const;

    const std::vector<index_t>& closest_boundary_to_bulk() const;

    const std::vector<point_t>& ghost_points() const;

  private:
    friend class conforming_point_cloud_impl_t<value_type, dimensions>;
    std::shared_ptr<conforming_point_cloud_impl_t<value_type, dimensions>> impl{};
};

extern template class conforming_point_cloud_t<float, 2>;
extern template class conforming_point_cloud_t<float, 3>;
extern template class conforming_point_cloud_t<double, 2>;
extern template class conforming_point_cloud_t<double, 3>;

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail
