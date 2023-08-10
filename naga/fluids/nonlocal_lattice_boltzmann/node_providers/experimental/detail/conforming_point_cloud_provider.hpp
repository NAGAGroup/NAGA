
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

template <uint Dimensions>
struct input_domain_data_t{};

template <>
struct input_domain_data_t<2> {
    using type = naga::experimental::mesh::closed_contour_t<double>;
};

template <>
struct input_domain_data_t<3> {
    using type = naga::experimental::mesh::triangular_mesh_t<double>;
};

template<uint Dimensions>
class conforming_point_cloud_t {
    static_assert(
        Dimensions == 2 || Dimensions == 3,
        "Dimensions must be 2 or 3"
    );

  public:
    using point_t  = ::naga::point_t<double, Dimensions>;
    using normal_t = point_t;
    using index_t  = size_t;

    conforming_point_cloud_t() = default;

    using input_domain_data_t = typename input_domain_data_t<Dimensions>::type;

    static conforming_point_cloud_t create(
        const double& approximate_spacing,
        const std::filesystem::path& domain,
        const std::vector<std::filesystem::path>& immersed_boundaries = {}
    ) {
        conforming_point_cloud_t point_cloud;
        auto impl = conforming_point_cloud_impl_t<Dimensions>::create(
            approximate_spacing,
            domain,
            immersed_boundaries
        );
        auto impl_ptr = std::make_shared<conforming_point_cloud_impl_t<Dimensions>>(
            std::move(impl)
        );
        point_cloud.impl = std::move(impl_ptr);
        return point_cloud;
    }

    const input_domain_data_t& domain() const {
        return impl->domain();
    }

    const std::vector<input_domain_data_t>& immersed_boundaries() const {
        return impl->immersed_boundaries();
    }

    const std::vector<point_t>& points() const {
        return impl->points();
    }

    const std::vector<normal_t>& normals() const {
        return impl->normals();
    }

    const size_t& num_bulk_points() const {
        return impl->num_bulk_points();
    }

    const size_t& num_boundary_points() const {
        return impl->num_boundary_points();
    }

    size_t size() const {
        return impl->size();
    }

    bool is_boundary(const index_t &i) const {
        return impl->is_boundary(i);
    }

    normal_t get_normal(const index_t &i) const {
        return impl->get_normal(i);
    }

  private:
    friend class conforming_point_cloud_impl_t<Dimensions>;
    std::shared_ptr<conforming_point_cloud_impl_t<Dimensions>> impl{};
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

template<>
class conforming_point_cloud_t<3> {

  public:
    static constexpr uint dimensions = 3;
    using point_t  = ::naga::point_t<double, dimensions>;
    using normal_t = point_t;
    using index_t  = size_t;

    conforming_point_cloud_t() = default;

    using input_domain_data_t = typename input_domain_data_t<dimensions>::type;

    static conforming_point_cloud_t create(
        const double& approximate_spacing,
        const std::filesystem::path& domain,
        const std::vector<std::filesystem::path>& immersed_boundaries = {}
    );

    const std::vector<point_t> &bulk_points() const;

    const std::vector<double> &bulk_to_boundary_distances() const;

    const std::vector<point_t> &boundary_points() const;

    const std::vector<normal_t> &boundary_normals() const;

    const std::vector<index_t> &closest_boundary_to_bulk() const;

  private:
    friend class conforming_point_cloud_impl_t<dimensions>;
    std::shared_ptr<conforming_point_cloud_impl_t<dimensions>> impl{};
};

extern template class conforming_point_cloud_t<2>;
extern template class conforming_point_cloud_t<3>;

}
