
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
#include "../../node_provider.cuh"
#include "detail/conforming_point_cloud_provider.hpp"
#include <scalix/filesystem.hpp>
#include <scalix/fill.cuh>

namespace naga::fluids::nonlocal_lbm {
template<class T>
class d2q9_lattice;
}

namespace naga::experimental::fluids::nonlocal_lbm {

template<class Lattice>
using node_provider = ::naga::fluids::nonlocal_lbm::node_provider<Lattice>;

template<class T>
using simulation_nodes = ::naga::fluids::nonlocal_lbm::simulation_nodes<T>;

template<class T>
T distance_to_edge(
    const T* xi,
    const std::vector<T>& x1,
    const std::vector<T>& x1_normal,
    const std::vector<T>& x2,
    const std::vector<T>& x2_normal
) {
    std::vector<T> average_normal
        = {(x2_normal[0] + x1_normal[0]) / 2,
           (x2_normal[1] + x1_normal[1]) / 2};
    T norm_normal = math::loopless::norm<2>(average_normal.data());
    average_normal[0] /= norm_normal;
    average_normal[1] /= norm_normal;

    std::vector<T> x12         = {x2[0] - x1[0], x2[1] - x1[1]};
    T mag_x12                  = sqrtf(x12[0] * x12[0] + x12[1] * x12[1]);
    std::vector<T> e1          = {x12[0] / mag_x12, x12[1] / mag_x12};
    std::vector<T> edge_normal = {e1[1], -e1[0]};
    if (edge_normal[0] * average_normal[0] + edge_normal[1] * average_normal[1]
        < 0) {
        edge_normal[0] *= -1;
        edge_normal[1] *= -1;
    }
    std::vector<T> x1i = {xi[0] - x1[0], xi[1] - x1[1]};
    T mag_x1i          = sqrtf(x1i[0] * x1i[0] + x1i[1] * x1i[1]);
    T x1i_dot_e1       = x1i[0] * e1[0] + x1i[1] * e1[1];
    T x1i_dot_normal   = x1i[0] * edge_normal[0] + x1i[1] * edge_normal[1];
    T distance         = std::abs(x1i_dot_normal);
    if (x1i_dot_e1 > mag_x12) {
        T mag_x2i = sqrtf(
            (xi[0] - x2[0]) * (xi[0] - x2[0])
            + (xi[1] - x2[1]) * (xi[1] - x2[1])
        );
        distance = mag_x2i;
    } else if (x1i_dot_e1 <= 0)
        distance = mag_x1i;

    if (x1i_dot_normal <= 0)
        distance *= -1;

    return distance;
}

template<class T, class U>
T get_distance_to_contour(
    const T* p_new,
    const std::vector<U>& vertices,
    const std::vector<U>& vertex_normals,
    const std::vector<size_t>& edges
) {
    std::vector<size_t> edge = {edges[0], edges[1]};
    auto min_distance        = distance_to_edge<T>(
        p_new,
        {vertices[2 * edge[0]], vertices[2 * edge[0] + 1]},
        {vertex_normals[2 * edge[0]], vertex_normals[2 * edge[0] + 1]},
        {vertices[2 * edge[1]], vertices[2 * edge[1] + 1]},
        {vertex_normals[2 * edge[1]], vertex_normals[2 * edge[1] + 1]}
    );
    for (size_t e = 1; e < edges.size() / 2; e++) {
        edge          = {edges[2 * e], edges[2 * e + 1]};
        auto distance = distance_to_edge<T>(
            p_new,
            {vertices[2 * edge[0]], vertices[2 * edge[0] + 1]},
            {vertex_normals[2 * edge[0]], vertex_normals[2 * edge[0] + 1]},
            {vertices[2 * edge[1]], vertices[2 * edge[1] + 1]},
            {vertex_normals[2 * edge[1]], vertex_normals[2 * edge[1] + 1]}
        );
        if (std::abs(distance) < std::abs(min_distance))
            min_distance = distance;
    }

    return min_distance;
}

template<class Lattice>
class conforming_point_cloud_provider : public node_provider<Lattice> {
  public:
    using base                       = node_provider<Lattice>;
    static constexpr uint dimensions = base::dimensions;
    using value_type                 = typename base::value_type;

    conforming_point_cloud_provider(
        const value_type & approximate_spacing,
        const sclx::filesystem::path& domain,
        const std::vector<sclx::filesystem::path>& immersed_boundaries = {},
        const value_type & domain_absorption_layer_thickness                = 0,
        const value_type & domain_boundary_absorption_rate                  = 0,
        const std::vector<value_type >& immersed_boundary_layer_thicknesses = {},
        const std::vector<value_type >& immersed_boundary_absorption_rates  = {}
    ) {}

    simulation_nodes<value_type> get() const final { return {}; }
};

template<class T>
class conforming_point_cloud_provider<
    ::naga::fluids::nonlocal_lbm::d2q9_lattice<T>>
    : public node_provider<::naga::fluids::nonlocal_lbm::d2q9_lattice<T>> {
  public:
    using base = node_provider<::naga::fluids::nonlocal_lbm::d2q9_lattice<T>>;
    static constexpr uint dimensions = base::dimensions;
    using value_type                 = typename base::value_type;

    conforming_point_cloud_provider(
        const value_type & approximate_spacing,
        const sclx::filesystem::path& domain,
        const std::vector<sclx::filesystem::path>& immersed_boundaries = {},
        const value_type & domain_absorption_layer_thickness                = 0,
        const value_type & domain_boundary_absorption_rate                  = 0,
        const std::vector<value_type >& immersed_boundary_layer_thicknesses = {},
        const std::vector<value_type >& immersed_boundary_absorption_rates  = {}
    ) {

        auto conforming_point_cloud
            = detail::conforming_point_cloud_t<T, dimensions>::create(
                approximate_spacing,
                domain,
                immersed_boundaries
            );

        nodes_.points = sclx::array<value_type, 2>{
            dimensions,
            conforming_point_cloud.points().size()};
        for (size_t i = 0; i < conforming_point_cloud.points().size(); ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                nodes_.points(j, i) = conforming_point_cloud.points()[i][j];
            }
        }

        nodes_.boundary_normals = sclx::array<value_type, 2>{
            dimensions,
            conforming_point_cloud.normals().size()};
        for (size_t i = 0; i < conforming_point_cloud.normals().size(); ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                nodes_.boundary_normals(j, i)
                    = conforming_point_cloud.normals()[i][j];
            }
        }

        nodes_.num_bulk_points = conforming_point_cloud.num_bulk_points();
        nodes_.num_boundary_points
            = conforming_point_cloud.num_boundary_points();

        nodes_.nodal_spacing = approximate_spacing;

        std::vector<point_t<value_type , dimensions>> bulk_points(
            conforming_point_cloud.points().begin(),
            conforming_point_cloud.points().begin()
                + conforming_point_cloud.num_bulk_points()
        );
        auto absorption_rates
            = sclx::zeros<value_type, 1>({nodes_.num_bulk_points});
        auto contour_distances
            = sclx::zeros<value_type, 1>({nodes_.num_bulk_points});
        sclx::fill<value_type, 1>(
            contour_distances,
            std::numeric_limits<value_type>::max()
        );

        using closed_contour_t = typename detail::conforming_point_cloud_t<T,
            dimensions>::input_domain_data_t;

        std::mutex distance_mutex;

        if (domain_absorption_layer_thickness > 0) {
            closed_contour_t domain_contour = conforming_point_cloud.domain();
#pragma omp parallel for
            for (size_t i = 0; i < nodes_.num_bulk_points; ++i) {
                auto distance_to_domain_contour = get_distance_to_contour(
                    &bulk_points[i][0],
                    domain_contour.vertices(),
                    domain_contour.vertex_normals(),
                    domain_contour.edges()
                );
                distance_to_domain_contour
                    = std::abs(distance_to_domain_contour);
                std::lock_guard<std::mutex> lock(distance_mutex);
                if (distance_to_domain_contour
                    < domain_absorption_layer_thickness && distance_to_domain_contour
                        < contour_distances(i)) {
                    contour_distances(i) = distance_to_domain_contour;
                    absorption_rates(i)
                        = ::naga::math::loopless::pow<2>(
                              (domain_absorption_layer_thickness
                               - distance_to_domain_contour)
                              / domain_absorption_layer_thickness
                          )
                        * domain_boundary_absorption_rate;
                }
            }
        }

        for (size_t i = 0; i < immersed_boundaries.size(); ++i) {
            if (immersed_boundary_layer_thicknesses[i] == 0) {
                continue;
            }
            closed_contour_t immersed_boundary_contour
                = conforming_point_cloud.immersed_boundaries()[i];
#pragma omp parallel for
            for (size_t j = 0; j < nodes_.num_bulk_points; ++j) {
                auto distance_to_immersed_boundary_contour
                    = get_distance_to_contour(
                        &bulk_points[j][0],
                        immersed_boundary_contour.vertices(),
                        immersed_boundary_contour.vertex_normals(),
                        immersed_boundary_contour.edges()
                    );
                distance_to_immersed_boundary_contour
                    = std::abs(distance_to_immersed_boundary_contour);
                std::lock_guard<std::mutex> lock(distance_mutex);
                if (distance_to_immersed_boundary_contour
                        <= immersed_boundary_layer_thicknesses[i]
                    && distance_to_immersed_boundary_contour
                           < contour_distances(j)) {
                    contour_distances(j)
                        = distance_to_immersed_boundary_contour;
                    absorption_rates(j)
                        = ::naga::math::loopless::pow<2>(
                              (immersed_boundary_layer_thicknesses[i]
                               - distance_to_immersed_boundary_contour)
                              / immersed_boundary_layer_thicknesses[i]
                          )
                        * immersed_boundary_absorption_rates[i];
                }
            }
        }
        size_t num_layer_points = std::count_if(
            absorption_rates.begin(),
            absorption_rates.end(),
            [](const auto& rate) { return rate > 0; }
        );
        if (num_layer_points == 0) {
            return;
        }

        std::stable_partition(
            bulk_points.begin(),
            bulk_points.end(),
            [&](const auto& point) {
                return absorption_rates[&point - &bulk_points[0]] == 0;
            }
        );
        std::stable_partition(
            absorption_rates.begin(),
            absorption_rates.end(),
            [](const auto& rate) { return rate == 0; }
        );
        absorption_rates = sclx::array<value_type, 1>(
            sclx::shape_t<1>{num_layer_points},
            absorption_rates.data().get() + nodes_.num_bulk_points
                - num_layer_points
        );
        for (size_t i = 0; i < bulk_points.size(); ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                nodes_.points(j, i) = bulk_points[i][j];
            }
        }
        nodes_.num_layer_points = num_layer_points;
        nodes_.layer_absorption = absorption_rates;
        nodes_.num_bulk_points -= num_layer_points;
    }

    simulation_nodes<value_type> get() const final { return nodes_; }

  private:
    simulation_nodes<value_type> nodes_{};
};

template<class T>
class conforming_point_cloud_provider<
    ::naga::fluids::nonlocal_lbm::d3q27_lattice<T>>
    : public node_provider<::naga::fluids::nonlocal_lbm::d3q27_lattice<T>> {
  public:
    using base = node_provider<::naga::fluids::nonlocal_lbm::d3q27_lattice<T>>;
    static constexpr uint dimensions = base::dimensions;
    using value_type                 = typename base::value_type;

    conforming_point_cloud_provider(
        const value_type & approximate_spacing,
        const sclx::filesystem::path& domain,
        const std::vector<sclx::filesystem::path>& immersed_boundaries = {},
        const value_type & domain_absorption_layer_thickness                = 0,
        const value_type & domain_boundary_absorption_rate                  = 0,
        const std::vector<value_type >& immersed_boundary_layer_thicknesses = {},
        const std::vector<value_type >& immersed_boundary_absorption_rates  = {}
    ) {
        auto conforming_point_cloud
            = detail::conforming_point_cloud_t<T, 3>::create(
                approximate_spacing,
                domain,
                immersed_boundaries
            );

        size_t num_bulk_and_layer_points
            = conforming_point_cloud.bulk_points().size();
        size_t num_boundary_points
            = conforming_point_cloud.boundary_points().size();
        size_t num_ghost_points = conforming_point_cloud.ghost_points().size();
        size_t num_points = num_bulk_and_layer_points + num_boundary_points + num_ghost_points;

        nodes_.nodal_spacing       = approximate_spacing;
        nodes_.num_boundary_points = num_boundary_points;
        nodes_.num_ghost_nodes     = num_ghost_points;
        nodes_.points = sclx::array<value_type, 2>{dimensions, num_points};
        nodes_.boundary_normals
            = sclx::array<value_type, 2>{dimensions, num_boundary_points};

        std::vector<value_type > absorption_coefficients;
        absorption_coefficients.reserve(num_bulk_and_layer_points + num_ghost_points);

        const auto& closest_boundary_indices
            = conforming_point_cloud.closest_boundary_to_bulk();
        const auto& distance_to_outer_boundary
            = conforming_point_cloud.bulk_to_boundary_distances();
        for (uint i = 0; i < num_bulk_and_layer_points; ++i) {
            value_type peak_absorption_coefficient = 0.0;
            value_type layer_thickness             = 0.0;
            if (closest_boundary_indices[i] == 0) {
                peak_absorption_coefficient = domain_boundary_absorption_rate;
                layer_thickness             = domain_absorption_layer_thickness;
            } else {
                if (!immersed_boundary_layer_thicknesses.empty()
                    && !immersed_boundary_absorption_rates.empty()) {
                    peak_absorption_coefficient
                        = immersed_boundary_absorption_rates
                            [closest_boundary_indices[i] - 1];
                    layer_thickness = immersed_boundary_layer_thicknesses
                        [closest_boundary_indices[i] - 1];
                }
            }

            const auto& distance_to_boundary = distance_to_outer_boundary[i];

            if (distance_to_boundary > layer_thickness) {
                absorption_coefficients.push_back(0.0);
                continue;
            }

            value_type absorption_coefficient
                = peak_absorption_coefficient
                * naga::math::loopless::pow<2>(
                      1.0 - distance_to_boundary / layer_thickness
                );
            absorption_coefficients.push_back(absorption_coefficient);
        }
        for (uint i = 0; i < num_ghost_points; ++i) {
            value_type peak_absorption_coefficient = 1.0;
            value_type layer_thickness             = naga::math::abs(detail::min_bound_dist_scaled_ghost_node_3d);

            const auto& distance_to_boundary = distance_to_outer_boundary[i + num_bulk_and_layer_points];

            if (distance_to_boundary > layer_thickness) {
                absorption_coefficients.push_back(0.0);
                continue;
            }

            value_type absorption_coefficient
                = peak_absorption_coefficient
                * naga::math::loopless::pow<2>(
                      1.0 - distance_to_boundary / layer_thickness
                );
            absorption_coefficients.push_back(absorption_coefficient);
        }

        auto bulk_points = conforming_point_cloud.bulk_points();

        std::stable_partition(
            bulk_points.begin(),
            bulk_points.end(),
            [&](const naga::point_t<value_type , 3>& x) {
                size_t i = &x - &bulk_points[0];
                return absorption_coefficients[i] == 0.0;
            }
        );
        std::stable_partition(
            absorption_coefficients.begin(),
            absorption_coefficients.begin() + num_bulk_and_layer_points,
            [&](const value_type & x) {
                return x == 0.0;
            }
        );
        size_t num_layer_points = std::count_if(
            absorption_coefficients.begin(),
            absorption_coefficients.begin() + num_bulk_and_layer_points,
            [](const auto& rate) { return rate > 0; }
        );

        size_t num_bulk_points  = num_bulk_and_layer_points - num_layer_points;
        nodes_.num_bulk_points  = num_bulk_points;
        nodes_.num_layer_points = num_layer_points;

        if (num_layer_points > 0) {
            nodes_.layer_absorption
                = sclx::array<value_type, 1>{num_layer_points + num_ghost_points};
        }

        for (size_t i = 0; i < num_bulk_points; ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                nodes_.points(j, i) = bulk_points[i][j];
            }
        }

        for (size_t i = 0; i < num_layer_points; ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                nodes_.points(j, i + num_bulk_points)
                    = bulk_points[i + num_bulk_points][j];
                nodes_.layer_absorption(i)
                    = absorption_coefficients[i + num_bulk_points];
            }
        }

        const auto& boundary_points = conforming_point_cloud.boundary_points();
        for (size_t i = 0; i < num_boundary_points; ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                nodes_.points(j, i + num_bulk_and_layer_points)
                    = boundary_points[i][j];
                nodes_.boundary_normals(j, i)
                    = conforming_point_cloud.boundary_normals()[i][j];
            }
        }

        const auto& ghost_points = conforming_point_cloud.ghost_points();
        for (size_t i = 0; i < num_ghost_points; ++i) {
            for (size_t j = 0; j < dimensions; ++j) {
                nodes_.points(j, i + num_bulk_and_layer_points + num_boundary_points)
                    = ghost_points[i][j];
                nodes_.layer_absorption(i + num_layer_points)
                    = absorption_coefficients[i + num_bulk_and_layer_points];
            }
        }
    }

    simulation_nodes<value_type> get() const final { return nodes_; }

  private:
    simulation_nodes<value_type> nodes_{};
};

}  // namespace naga::experimental::fluids::nonlocal_lbm
