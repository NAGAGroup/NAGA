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

#include <future>
#include <iostream>
#include <mutex>
#include <naga/distance_functions.hpp>
#include <naga/fluids/nonlocal_lattice_boltzmann/node_providers/experimental/detail/conforming_point_cloud_provider.hpp>
#include <numeric>
#include <random>
#include <scalix/algorithm/reduce_last_dim.cuh>
#include <scalix/algorithm/transform.cuh>
#include <scalix/array.cuh>
#include <scalix/filesystem.hpp>
#include <scalix/fill.cuh>
#include <utility>

#define TINYOBJLOADER_IMPLEMENTATION  // define this in only *one* .cc
// Optional. define TINYOBJLOADER_USE_MAPBOX_EARCUT gives robust trinagulation.
// Requires C++11
// #define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

namespace naga::experimental::fluids::nonlocal_lbm::detail {

struct hashable_edge {
    size_t i;
    size_t j;
};

bool operator==(const hashable_edge& lhs, const hashable_edge& rhs) {
    return (lhs.i == rhs.i && lhs.j == rhs.j)
        || (lhs.i == rhs.j && lhs.j == rhs.i);
}

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail

namespace std {
template<>
struct hash<naga::experimental::fluids::nonlocal_lbm::detail::hashable_edge> {
    using hashable_edge
        = naga::experimental::fluids::nonlocal_lbm::detail::hashable_edge;
    size_t operator()(const hashable_edge& e) const {
        return std::hash<size_t>()(e.i) ^ std::hash<size_t>()(e.j) << 1;
    }
};
}  // namespace std

namespace naga::experimental::fluids::nonlocal_lbm::detail {

constexpr float min_bound_dist_scale_2d = 0.71f;
constexpr float min_bound_dist_scale_3d = -.6f;
constexpr float max_bound_dist_scale_3d = 0.6f;

template<class T>
struct edge_info_t {
    using value_type = T;
    using point_t    = naga::point_t<value_type, 2>;
    using vector_t   = naga::point_t<value_type, 2>;
    using normal_t   = naga::point_t<value_type, 2>;

    point_t v1;
    point_t v2;

    vector_t v12;
    vector_t normalized_edge_dir;
    value_type length;

    normal_t n1;
    normal_t n2;
    normal_t edge_normal;
};

template<class T>
edge_info_t<T> get_edge_info(
    size_t edge_index,
    const std::vector<T>& vertices,
    const std::vector<T>& vertex_normals,
    const std::vector<size_t>& edges
) {
    using value_type = T;
    edge_info_t<T> edge_info{};
    using point_t  = typename edge_info_t<T>::point_t;
    using vector_t = typename edge_info_t<T>::vector_t;
    using normal_t = typename edge_info_t<T>::normal_t;
    edge_info.v1   = point_t{
          {vertices[edges[edge_index] * 2 + 0],
           vertices[edges[edge_index] * 2 + 1]}
    };
    edge_info.v2 = point_t{
        {vertices[edges[edge_index + 1] * 2 + 0],
         vertices[edges[edge_index + 1] * 2 + 1]}
    };

    edge_info.v12 = vector_t{
        {edge_info.v2[0] - edge_info.v1[0], edge_info.v2[1] - edge_info.v1[1]}
    };
    edge_info.length = naga::math::loopless::norm<2>(edge_info.v12);
    edge_info.normalized_edge_dir = vector_t{
        {edge_info.v12[0] / edge_info.length,
         edge_info.v12[1] / edge_info.length}
    };

    edge_info.n1 = normal_t{
        {vertex_normals[edges[edge_index] * 2 + 0],
         vertex_normals[edges[edge_index] * 2 + 1]}
    };
    naga::math::loopless::normalize<2>(edge_info.n1);
    edge_info.n2 = normal_t{
        {vertex_normals[edges[edge_index + 1] * 2 + 0],
         vertex_normals[edges[edge_index + 1] * 2 + 1]}
    };
    naga::math::loopless::normalize<2>(edge_info.n2);
    edge_info.edge_normal = normal_t{
        {(edge_info.n1[0] + edge_info.n2[0]) / 2.f,
         (edge_info.n1[1] + edge_info.n2[1]) / 2.f}
    };
    naga::math::loopless::normalize<2>(edge_info.edge_normal);

    return edge_info;
}

template<class T>
void subdivide_edges_and_cache_edge_info(
    T target_length,
    std::vector<T>& vertices,
    std::vector<T>& vertex_normals,
    std::vector<size_t>& edges,
    std::vector<edge_info_t<T>>& input_edge_info
) {
    input_edge_info.clear();
    input_edge_info.reserve(edges.size() / 2);
    auto vertices_copy       = vertices;
    auto vertex_normals_copy = vertex_normals;
    auto edges_copy          = edges;
    for (size_t e = 0; e < edges_copy.size(); e += 2) {
        auto edge_info
            = get_edge_info(e, vertices_copy, vertex_normals_copy, edges_copy);
        input_edge_info.push_back(edge_info);

        T edge_len2target_len = edge_info.length / target_length;
        T acceptable_epsilon  = 0.5;
        if (edge_len2target_len < 1 + acceptable_epsilon) {
            continue;
        }
        size_t num_nodes_to_add
            = static_cast<size_t>(std::ceil(edge_len2target_len)) - 1;
        T actual_length
            = edge_info.length / static_cast<T>(num_nodes_to_add + 1);

        size_t old_edge_end = edges_copy[e + 1];
        naga::point_t<T, 2> new_vertex{
            {edge_info.v1[0] + edge_info.normalized_edge_dir[0] * actual_length,
             edge_info.v1[1] + edge_info.normalized_edge_dir[1] * actual_length}
        };
        vertices.push_back(new_vertex[0]);
        vertices.push_back(new_vertex[1]);
        vertex_normals.push_back(edge_info.edge_normal[0]);
        vertex_normals.push_back(edge_info.edge_normal[1]);
        edges[e + 1] = vertices.size() / 2 - 1;
        for (uint i = 1; i < num_nodes_to_add; ++i) {
            new_vertex[0] += edge_info.normalized_edge_dir[0] * actual_length;
            new_vertex[1] += edge_info.normalized_edge_dir[1] * actual_length;
            vertices.push_back(new_vertex[0]);
            vertices.push_back(new_vertex[1]);
            vertex_normals.push_back(edge_info.edge_normal[0]);
            vertex_normals.push_back(edge_info.edge_normal[1]);
            edges.push_back(vertices.size() / 2 - 2);
            edges.push_back(vertices.size() / 2 - 1);
        }
        edges.push_back(vertices.size() / 2 - 1);
        edges.push_back(old_edge_end);
    }
}

template<class T>
class line_t {
  public:
    using point_t = naga::point_t<T, 2>;
    line_t(point_t p1, point_t p2) : p1_(std::move(p1)), p2_(std::move(p2)) {}

    // constructs a perpendicular line that passes through perp_line_p1
    // the resulting line is defined by the perp_line_p1 and the intersection
    // of the two lines
    // y1 - y2 = m1(x1 - x2)
    // y3 - y4 = m2(x3 - x4)

    line_t perpendicular_line(const point_t& perp_line_p1) {
        auto this_slope = slope();
        auto this_b     = p1_[1] - this_slope * p1_[0];
        auto perp_slope = -1. / this_slope;
        if (this_slope == 0) {
            return line_t{perp_line_p1, point_t{{perp_line_p1[0], p1_[1]}}};
        }
        if (perp_slope == 0) {
            return line_t{perp_line_p1, point_t{{p1_[0], perp_line_p1[1]}}};
        }
        auto perp_b = perp_line_p1[1] - perp_slope * perp_line_p1[0];
        point_t intersection;
        intersection[0] = (perp_b - this_b) / (this_slope - perp_slope);
        intersection[1] = this_slope * intersection[0] + this_b;
        return line_t{perp_line_p1, intersection};
    }

    T slope() { return (p2_[1] - p1_[1]) / (p2_[0] - p1_[0]); }

    const point_t& p1() const { return p1_; }
    const point_t& p2() const { return p2_; }

  private:
    point_t p1_;
    point_t p2_;
};

template<class T>
int sgn(T x) {
    return static_cast<int>(x > 0) - static_cast<int>(x < 0);
}

template<class T>
std::tuple<T, T> distance_to_edge(
    const naga::point_t<T, 2>& xi,
    const edge_info_t<T>& edge_info
) {
    using vector_t = naga::point_t<T, 2>;
    line_t edge_line{edge_info.v1, edge_info.v2};
    auto perp_line = edge_line.perpendicular_line(xi);
    vector_t xi_to_intersection{
        {perp_line.p2()[0] - xi[0], perp_line.p2()[1] - xi[1]}
    };
    auto signed_distance_to_line
        = sgn(naga::math::loopless::dot<2>(
              xi_to_intersection,
              edge_info.edge_normal
          ))
        * naga::math::loopless::norm<2>(xi_to_intersection) * -1;

    vector_t v1_to_intersection{
        {perp_line.p2()[0] - edge_info.v1[0],
         perp_line.p2()[1] - edge_info.v1[1]}
    };
    auto signed_v1_intersection_distance = naga::math::loopless::dot<2>(
        v1_to_intersection,
        edge_info.normalized_edge_dir
    );

    if (signed_v1_intersection_distance < 0) {
        vector_t v1_to_xi{{xi[0] - edge_info.v1[0], xi[1] - edge_info.v1[1]}};
        auto v1_to_xi_dot_normal
            = naga::math::loopless::dot<2>(v1_to_xi, edge_info.edge_normal);
        T incident_angle = naga::math::asin(
            naga::math::abs(v1_to_xi_dot_normal)
            / naga::math::loopless::norm<2>(v1_to_xi)
        );
        auto signed_distance_to_v1 = sgn(v1_to_xi_dot_normal)
                                   * naga::math::loopless::norm<2>(v1_to_xi);
        return {signed_distance_to_v1, incident_angle};
    }
    if (signed_v1_intersection_distance > edge_info.length) {
        vector_t v2_to_xi{{xi[0] - edge_info.v2[0], xi[1] - edge_info.v2[1]}};
        auto v2_to_xi_dot_normal
            = naga::math::loopless::dot<2>(v2_to_xi, edge_info.edge_normal);
        T angle_from_perp_line = naga::math::asin(
            naga::math::abs(v2_to_xi_dot_normal)
            / naga::math::loopless::norm<2>(v2_to_xi)
        );
        auto signed_distance_to_v2 = sgn(v2_to_xi_dot_normal)
                                   * naga::math::loopless::norm<2>(v2_to_xi);
        return {signed_distance_to_v2, angle_from_perp_line};
    }
    return {signed_distance_to_line, naga::math::pi<T> / 2.};
}

template<class T>
std::pair<size_t, double> distance_to_boundary(
    const naga::point_t<T, 2>& xi,
    const std::vector<edge_info_t<T>>& edge_info
) {
    T min_distance              = std::numeric_limits<T>::max();
    T associated_incident_angle = 0;
    size_t edge_index           = 0;
    std::mutex min_distance_mutex;
#pragma omp parallel for
    for (size_t e = 0; e < edge_info.size(); ++e) {
        auto [distance, incident_angle] = distance_to_edge(xi, edge_info[e]);
        T diff_from_min = std::abs(std::abs(distance) - std::abs(min_distance));
        std::lock_guard<std::mutex> lock(min_distance_mutex);
        if (std::abs(distance) < std::abs(min_distance)
            || (diff_from_min < 1e-6
                && incident_angle > associated_incident_angle)) {
            min_distance              = distance;
            associated_incident_angle = incident_angle;
            edge_index                = e;
        }
    }
    return {edge_index, min_distance};
}

template<class T>
std::vector<point_t<T, 2>> generate_2d_hexagonal_grid(
    double approx_point_spacing,
    const point_t<T, 2>& lower_bound,
    const point_t<T, 2>& upper_bound
) {
    std::vector<naga::point_t<T, 2>> potential_bulk_points;

    double acceptable_epsilon = 0.1;
    if ((upper_bound[0] - lower_bound[0]) / approx_point_spacing
            < 1 + acceptable_epsilon
        || (upper_bound[1] - lower_bound[1]) / approx_point_spacing
               < 1 + acceptable_epsilon) {
        return potential_bulk_points;
    }

    size_t approx_grid_size[2]{
        static_cast<size_t>(std::ceil(
            (upper_bound[0] - lower_bound[0]) / approx_point_spacing + 1
        )),
        static_cast<size_t>(std::ceil(
            (upper_bound[1] - lower_bound[1]) / approx_point_spacing + 1
        ))
    };
    potential_bulk_points.reserve(approx_grid_size[0] * approx_grid_size[1]);

    const T l = 0.5
                  * ((upper_bound[0] - lower_bound[0])
                     / (static_cast<T>(approx_grid_size[0]) - 1))
              + 0.5
                    * ((upper_bound[1] - lower_bound[1])
                       / (static_cast<T>(approx_grid_size[1]) - 1));
    const T w = l * naga::math::sin(naga::math::pi<T> / 6.);
    const T h = l * naga::math::cos(naga::math::pi<T> / 6.);
    // first hexagonal grid
    T x_offset       = 0;
    T y              = lower_bound[1];
    bool is_even_row = true;
    while (y <= upper_bound[1]) {
        T x = is_even_row ? lower_bound[0] : lower_bound[0] - w;
        x += x_offset;
        bool is_even_column = is_even_row;
        while (x <= upper_bound[0]) {
            potential_bulk_points.emplace_back(naga::point_t<T, 2>{{x, y}});
            x += is_even_column ? l : 2 * w + l;
            is_even_column = !is_even_column;
        }
        y += h;
        is_even_row = !is_even_row;
    }
    // second hexagonal grid
    x_offset    = l;
    y           = lower_bound[1];
    is_even_row = true;
    while (y <= upper_bound[1]) {
        T x = is_even_row ? lower_bound[0] : lower_bound[0] - w;
        x += x_offset;
        bool is_even_column = is_even_row;
        while (x <= upper_bound[0]) {
            if (!is_even_column) {
                potential_bulk_points.emplace_back(naga::point_t<T, 2>{{x, y}});
            }
            x += is_even_column ? l : 2 * w + l;
            is_even_column = !is_even_column;
        }
        y += h;
        is_even_row = !is_even_row;
    }

    return potential_bulk_points;
}

template<class T>
std::vector<std::pair<size_t, T>> remove_points_outside_2d_contours(
    std::vector<point_t<T, 2>>& potential_bulk_points,
    const T& approx_point_spacing,
    const std::vector<edge_info_t<T>>& boundary_edge_info,
    T min_bound_dist_scale = min_bound_dist_scale_2d
) {
    std::vector<std::pair<size_t, T>> distances_to_edges(
        potential_bulk_points.size()
    );
    std::transform(
        potential_bulk_points.begin(),
        potential_bulk_points.end(),
        distances_to_edges.begin(),
        [&](const auto& p) {
            return distance_to_boundary(p, boundary_edge_info);
        }
    );

    std::vector<char> valid_bulk_points(potential_bulk_points.size());
    std::transform(
        distances_to_edges.begin(),
        distances_to_edges.end(),
        valid_bulk_points.begin(),
        [&](const auto& d) {
            return d.second > min_bound_dist_scale * approx_point_spacing ? 1
                                                                          : 0;
        }
    );

    size_t num_bulk = std::accumulate(
        valid_bulk_points.begin(),
        valid_bulk_points.end(),
        0
    );
    std::stable_partition(
        potential_bulk_points.begin(),
        potential_bulk_points.end(),
        [&](const auto& p) {
            const auto get_index = [&]() {
                return static_cast<size_t>(&p - &potential_bulk_points[0]);
            };
            return valid_bulk_points[get_index()];
        }
    );
    potential_bulk_points.resize(num_bulk);

    std::stable_partition(
        distances_to_edges.begin(),
        distances_to_edges.end(),
        [&](const auto& d) {
            const auto get_index = [&]() {
                return static_cast<size_t>(&d - &distances_to_edges[0]);
            };
            return valid_bulk_points[get_index()];
        }
    );
    distances_to_edges.resize(num_bulk);

    return distances_to_edges;
}

template<class T>
class conforming_point_cloud_impl_t<T, 2> {
  public:
    using point_t  = naga::point_t<T, 2>;
    using normal_t = point_t;
    using index_t  = size_t;

    using closed_contour_t =
        typename conforming_point_cloud_t<T, 2>::input_domain_data_t;

    static conforming_point_cloud_impl_t create(
        const T& approx_point_spacing,
        const std::filesystem::path& domain,
        const std::vector<std::filesystem::path>& immersed_boundaries = {}
    ) {
        std::vector<closed_contour_t> immersed_boundary_contours;
        std::vector<T> boundary_vertices;
        std::vector<T> boundary_normals;
        std::vector<size_t> boundary_edges;
        for (auto& im_obj_file : immersed_boundaries) {
            immersed_boundary_contours.emplace_back(
                closed_contour_t::import(im_obj_file)
            );
            size_t prev_vertex_count = boundary_vertices.size() / 2;
            for (auto& vertex : immersed_boundary_contours.back().vertices()) {
                boundary_vertices.push_back(vertex);
            }
            for (auto& normal :
                 immersed_boundary_contours.back().vertex_normals()) {
                boundary_normals.push_back(normal);
            }
            for (auto& edge_index : immersed_boundary_contours.back().edges()) {
                boundary_edges.push_back(prev_vertex_count + edge_index);
            }
        }

        closed_contour_t domain_contour = closed_contour_t::import(domain);
        size_t prev_vertex_count        = boundary_vertices.size() / 2;
        for (auto& vertex : domain_contour.vertices()) {
            boundary_vertices.push_back(vertex);
        }
        for (auto& normal : domain_contour.vertex_normals()) {
            // reverse normals outer boundary
            boundary_normals.push_back(normal * -1.);
        }
        for (auto& edge_index : domain_contour.edges()) {
            boundary_edges.push_back(prev_vertex_count + edge_index);
        }

        std::vector<edge_info_t<T>> boundary_edge_info;
        subdivide_edges_and_cache_edge_info(
            approx_point_spacing,
            boundary_vertices,
            boundary_normals,
            boundary_edges,
            boundary_edge_info
        );

        const auto& domain_lower_bound = domain_contour.lower_bound();
        const auto& domain_upper_bound = domain_contour.upper_bound();

        auto potential_bulk_points = generate_2d_hexagonal_grid(
            approx_point_spacing,
            domain_lower_bound,
            domain_upper_bound
        );

        //        // we also add jitter in hopes it prevents artifacting
        //        std::default_random_engine offset_rng(2);
        //        // use same seed for reproducibility
        //        std::normal_distribution<T> offset_dist(
        //            0.0,
        //            0.05 * approx_point_spacing
        //        );
        //        std::default_random_engine angle_rng(456);
        //        // use same seed for reproducibility
        //        std::uniform_real_distribution<T>
        //            angle_dist(0., naga::math::pi<T>);
        //        for (auto& p : potential_bulk_points) {
        //            auto offset = offset_dist(offset_rng);
        //            auto angle  = angle_dist(angle_rng);
        //            p[0] += offset * naga::math::cos(angle);
        //            p[1] += offset * naga::math::sin(angle);
        //        }

        remove_points_outside_2d_contours(
            potential_bulk_points,
            approx_point_spacing,
            boundary_edge_info
        );

        auto num_bulk                = potential_bulk_points.size();
        std::vector<point_t>& points = potential_bulk_points;
        points.resize(points.size() + boundary_vertices.size() / 2);
        std::vector<normal_t> normals(boundary_vertices.size() / 2);
        std::transform(
            points.begin() + num_bulk,
            points.end(),
            points.begin() + num_bulk,
            [&](const auto& p) {
                const auto get_index = [&]() {
                    return static_cast<size_t>(&p - &points[num_bulk]);
                };
                return naga::point_t<T, 2>{
                    {boundary_vertices[get_index() * 2 + 0],
                     boundary_vertices[get_index() * 2 + 1]}
                };
            }
        );
        std::transform(
            normals.begin(),
            normals.end(),
            normals.begin(),
            [&](const auto& n) {
                const auto get_index
                    = [&]() { return static_cast<size_t>(&n - &normals[0]); };
                return naga::point_t<T, 2>{
                    {boundary_normals[get_index() * 2 + 0],
                     boundary_normals[get_index() * 2 + 1]}
                };
            }
        );

        //        std::ofstream
        //        points_file("/home/gpu-dev/naga-result-data/points.csv");
        //        points_file << "x,y,distance_to_edge\n";
        //        for (size_t i = 0; i < potential_bulk_points.size();
        //        ++i) {
        //            points_file << potential_bulk_points[i][0] << ","
        //                        << potential_bulk_points[i][1] << ","
        //                        << distances_to_edges[i].second <<
        //                        "\n";
        //        }
        //        points_file.close();

        size_t num_boundaries = normals.size();

        return {
            domain_contour,
            immersed_boundary_contours,
            points,
            normals,
            num_bulk,
            num_boundaries
        };
    }

    [[nodiscard]] const closed_contour_t& domain() const { return domain_; }

    [[nodiscard]] const std::vector<closed_contour_t>&
    immersed_boundaries() const {
        return immersed_boundaries_;
    }

    [[nodiscard]] const std::vector<point_t>& points() const { return points_; }

    [[nodiscard]] const std::vector<normal_t>& normals() const {
        return normals_;
    }

    [[nodiscard]] const size_t& num_bulk_points() const {
        return num_bulk_points_;
    }

    [[nodiscard]] const size_t& num_boundary_points() const {
        return num_boundary_points_;
    }

    [[nodiscard]] size_t size() const { return points_.size(); }

    [[nodiscard]] bool is_boundary(const index_t& i) const {
        return i >= num_bulk_points_;
    }

    normal_t get_normal(const index_t& i) {
        if (!is_boundary(i)) {
            return normal_t({0., 0.});
        }
        return normals_[i - num_bulk_points_];
    }

  private:
    conforming_point_cloud_impl_t(
        closed_contour_t domain,
        std::vector<closed_contour_t> immersed_boundaries,
        std::vector<point_t> points,
        std::vector<normal_t> normals,
        size_t num_bulk_points,
        size_t num_boundary_points
    )
        : domain_(std::move(domain)),
          immersed_boundaries_(std::move(immersed_boundaries)),
          points_(std::move(points)),
          normals_(std::move(normals)),
          num_bulk_points_(num_bulk_points),
          num_boundary_points_(num_boundary_points) {}

    closed_contour_t domain_;
    std::vector<closed_contour_t> immersed_boundaries_;

    std::vector<point_t> points_;
    std::vector<normal_t> normals_;  // size matches num_boundary_points_

    size_t num_bulk_points_;  // points vector is arranged with bulk first then
                              // boundary
    size_t num_boundary_points_;
};

template<class T>
conforming_point_cloud_t<T, 2> conforming_point_cloud_t<T, 2>::create(
    const T& approximate_spacing,
    const std::filesystem::path& domain,
    const std::vector<std::filesystem::path>& immersed_boundaries
) {
    conforming_point_cloud_t point_cloud;
    auto impl = conforming_point_cloud_impl_t<T, 2>::create(
        approximate_spacing,
        domain,
        immersed_boundaries
    );
    auto impl_ptr
        = std::make_shared<conforming_point_cloud_impl_t<T, 2>>(std::move(impl)
        );
    point_cloud.impl = std::move(impl_ptr);
    return point_cloud;
}

template<class T>
const typename conforming_point_cloud_t<T, 2>::input_domain_data_t&
conforming_point_cloud_t<T, 2>::domain() const {
    return impl->domain();
}

template<class T>
const std::vector<typename conforming_point_cloud_t<T, 2>::input_domain_data_t>&
conforming_point_cloud_t<T, 2>::immersed_boundaries() const {
    return impl->immersed_boundaries();
}

template<class T>
const std::vector<typename conforming_point_cloud_t<T, 2>::point_t>&
conforming_point_cloud_t<T, 2>::points() const {
    return impl->points();
}

template<class T>
const std::vector<typename conforming_point_cloud_t<T, 2>::normal_t>&
conforming_point_cloud_t<T, 2>::normals() const {
    return impl->normals();
}

template<class T>
const size_t& conforming_point_cloud_t<T, 2>::num_bulk_points() const {
    return impl->num_bulk_points();
}

template<class T>
const size_t& conforming_point_cloud_t<T, 2>::num_boundary_points() const {
    return impl->num_boundary_points();
}

template<class T>
size_t conforming_point_cloud_t<T, 2>::size() const {
    return impl->num_boundary_points();
}

template<class T>
bool conforming_point_cloud_t<T, 2>::is_boundary(
    const conforming_point_cloud_t::index_t& i
) const {
    return impl->is_boundary(i);
}

template<class T>
typename conforming_point_cloud_t<T, 2>::normal_t
conforming_point_cloud_t<T, 2>::get_normal(
    const conforming_point_cloud_t::index_t& i
) const {
    return impl->get_normal(i);
}

template class conforming_point_cloud_t<float, 2>;
template class conforming_point_cloud_t<double, 2>;

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail

#include <sdf/sdf.hpp>

namespace naga::experimental::fluids::nonlocal_lbm::detail {

template<class T>
using triangular_mesh_t = naga::experimental::mesh::triangular_mesh_t<T>;

struct sdf_metadata {
    std::unique_ptr<sdf::SDF> sdf;
    std::unique_ptr<sdf::Points> points;
    std::unique_ptr<sdf::Triangles> faces;
};

struct edge_t {
    edge_t(size_t i, size_t j) {
        if (i > j) {
            std::swap(i, j);
        }
        this->i = i;
        this->j = j;
    }

    edge_t()                               = default;
    edge_t(const edge_t& other)            = default;
    edge_t(edge_t&& other)                 = default;
    edge_t& operator=(const edge_t& other) = default;
    edge_t& operator=(edge_t&& other)      = default;

    size_t i;
    size_t j;
};

auto operator==(const edge_t& e1, const edge_t& e2) -> bool {
    return e1.i == e2.i && e1.j == e2.j;
}
class spinlock {
  public:
    void lock() {
        for (;;) {
            if (!lock_.exchange(true, std::memory_order_acquire)) {
                break;
            }
            while (lock_.load(std::memory_order_relaxed)) {
                std::this_thread::yield();
            }
        }
    }

    void unlock() { lock_.store(false, std::memory_order_release); }

  private:
    std::atomic<bool> lock_{false};
};

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail

template<>
struct std::hash<naga::experimental::fluids::nonlocal_lbm::detail::edge_t> {
    using argument_type
        = naga::experimental::fluids::nonlocal_lbm::detail::edge_t;
    auto operator()(const argument_type& edge) const noexcept -> size_t {
        return std::hash<size_t>()(edge.i) ^ std::hash<size_t>()(edge.j);
    }
};

namespace naga::experimental::fluids::nonlocal_lbm::detail {

template<class T>
struct manifold_mesh_t {
    using value_type = T;

    static constexpr auto no_face   = std::numeric_limits<size_t>::max();
    static constexpr auto no_edge   = std::numeric_limits<size_t>::max();
    static constexpr auto no_vertex = std::numeric_limits<size_t>::max();

    using edge_t       = detail::edge_t;
    using edge_count_t = int;

    static auto import_from_obj(
        const sclx::filesystem::path& obj_path,
        const bool flip_normals = false
    ) -> manifold_mesh_t {

        using namespace detail;
        manifold_mesh_t mesh;

        tinyobj::ObjReader reader;

        if (!reader.ParseFromFile(obj_path)) {
            if (!reader.Error().empty()) {
                throw std::runtime_error("TinyObjReader: " + reader.Error());
            }
        }

        if (!reader.Warning().empty()) {
            std::cout << "TinyObjReader: " << reader.Warning();
        }

        // check if all faces are triangles
        for (const auto& shape : reader.GetShapes()) {
            for (const auto& num_face_vertices : shape.mesh.num_face_vertices) {
                if (num_face_vertices != 3) {
                    throw std::runtime_error(
                        "TinyObjReader: non-triangular face detected"
                    );
                }
            }
        }

        spinlock lock;
        std::vector<size_t> face_vertex_indices;
        std::vector<size_t> face_normal_indices;
        std::unordered_map<edge_t, std::vector<size_t>> edge_opposite_vertices;
        std::unordered_map<edge_t, std::vector<size_t>> edge_face_neighbors;
        std::unordered_map<size_t, std::vector<size_t>> vertex_face_neighbors;
        std::unordered_map<size_t, std::vector<size_t>> vertex_opposite_edges;
        std::unordered_map<size_t, std::vector<size_t>> vertex_edge_neighbors;
        std::unordered_map<edge_t, edge_count_t> edge_counts;
        size_t total_face_count = 0;
        size_t total_vertex_count;
        size_t total_normal_count;

        {
            auto attrib = reader.GetAttrib();
            auto shapes = reader.GetShapes();

            face_vertex_indices.reserve(3 * (attrib.vertices.size() - 2));
            face_normal_indices.reserve(attrib.vertices.size() - 2);
            edge_opposite_vertices.reserve(3 * (attrib.vertices.size() - 2));
            edge_face_neighbors.reserve(3 * (attrib.vertices.size() - 2));
            vertex_face_neighbors.reserve(attrib.vertices.size());
            vertex_opposite_edges.reserve(attrib.vertices.size());
            vertex_edge_neighbors.reserve(attrib.vertices.size());
            edge_counts.reserve(3 * (attrib.vertices.size() - 2));

            total_vertex_count = attrib.vertices.size() / 3;
            mesh.vertices = sclx::array<value_type, 2>{3, total_vertex_count};
            std::copy(
                attrib.vertices.begin(),
                attrib.vertices.end(),
                &mesh.vertices(0, 0)
            );
            attrib.vertices.clear();

            total_normal_count = attrib.normals.size() / 3;
            mesh.normals = sclx::array<value_type, 2>{3, total_normal_count};
            std::copy(
                attrib.normals.begin(),
                attrib.normals.end(),
                &mesh.normals(0, 0)
            );

            // get metadata for constructing mesh
            for (const auto& shape : shapes) {
#pragma omp parallel for
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size();
                     ++f) {
                    for (size_t i = 0; i < 3; ++i) {
                        // face data
                        {
                            std::lock_guard<spinlock> guard(lock);
                            face_vertex_indices.push_back(
                                shape.mesh.indices[f * 3 + i].vertex_index
                            );
                            if (i == 0) {
                                face_normal_indices.push_back(
                                    shape.mesh.indices[f * 3 + i].normal_index
                                );
                            }
                        }

                        // edge data
                        size_t edge_idx;
                        size_t edge_opposite_vertex;
                        {
                            std::lock_guard<spinlock> guard(lock);
                            edge_t edge{
                                static_cast<size_t>(
                                    shape.mesh.indices[f * 3 + i].vertex_index
                                ),
                                static_cast<size_t>(
                                    shape.mesh.indices[f * 3 + (i + 1) % 3]
                                        .vertex_index
                                )
                            };
                            auto& edge_count = edge_counts[edge];
                            ++edge_count;
                            if (edge_count > 2) {
                                throw std::runtime_error(
                                    "manifold_mesh_t: edge is "
                                    "shared by more than two triangles"
                                );
                            }
                            edge_opposite_vertices[edge].push_back(
                                shape.mesh.indices[f * 3 + (i + 2) % 3]
                                    .vertex_index
                            );
                            edge_opposite_vertex
                                = edge_opposite_vertices[edge].back();
                            edge_face_neighbors[edge].push_back(total_face_count
                            );

                            auto edge_it = edge_counts.find({edge.i, edge.j});
                            edge_idx     = static_cast<size_t>(
                                std::distance(edge_counts.begin(), edge_it)
                            );
                        }

                        // vertex data
                        {
                            std::lock_guard<spinlock> guard(lock);
                            vertex_face_neighbors[shape.mesh.indices[f * 3 + i]
                                                      .vertex_index]
                                .push_back(total_face_count);
                            vertex_edge_neighbors[shape.mesh.indices[f * 3 + i]
                                                      .vertex_index]
                                .push_back(edge_idx);
                            vertex_opposite_edges[edge_opposite_vertex]
                                .push_back(edge_idx);
                        }
                    }
                    total_face_count += 1;
                }
            }
        }
        size_t max_vertex_face_neighbors = 0;
        for (const auto& [vertex, face_neighbors] : vertex_face_neighbors) {
            max_vertex_face_neighbors
                = std::max(max_vertex_face_neighbors, face_neighbors.size());
        }
        size_t max_vertex_edge_neighbors = 0;
        for (const auto& [vertex, edge_neighbors] : vertex_edge_neighbors) {
            max_vertex_edge_neighbors
                = std::max(max_vertex_edge_neighbors, edge_neighbors.size());
        }
        auto num_unique_edges = edge_counts.size();

        mesh.triangle_vert_indices
            = sclx::array<size_t, 2>{3, total_face_count};
        std::copy(
            face_vertex_indices.begin(),
            face_vertex_indices.end(),
            &mesh.triangle_vert_indices(0, 0)
        );
        face_vertex_indices.clear();
        face_vertex_indices.shrink_to_fit();
        mesh.triangle_normal_indices = sclx::array<size_t, 1>{total_face_count};
        std::copy(
            face_normal_indices.begin(),
            face_normal_indices.end(),
            &mesh.triangle_normal_indices(0)
        );

        mesh.unique_edges = sclx::array<size_t, 2>{2, num_unique_edges};
        mesh.double_edges = sclx::array<bool, 1>{num_unique_edges};
        sclx::fill(mesh.double_edges, false);
        mesh.edge_opposite_vertices
            = sclx::array<size_t, 2>{2, num_unique_edges};
        sclx::fill(mesh.edge_opposite_vertices, no_vertex);
        mesh.edge_face_neighbors = sclx::array<size_t, 2>{2, num_unique_edges};
        sclx::fill(mesh.edge_face_neighbors, no_face);
        for (const auto& [edge, count] : edge_counts) {
            auto edge_it  = edge_counts.find(edge);
            auto edge_idx = static_cast<size_t>(
                std::distance(edge_counts.begin(), edge_it)
            );
            if (count == 2) {
                mesh.double_edges(edge_idx) = true;
            }
            mesh.unique_edges(0, edge_idx) = edge.i;
            mesh.unique_edges(1, edge_idx) = edge.j;
            const auto& opposite_vertices  = edge_opposite_vertices[edge];
            if (opposite_vertices.size() != count) {
                throw std::runtime_error(
                    "manifold_mesh_t: edge opposite "
                    "vertices size does not match edge count, "
                    "this likely indicates bug in naga"
                );
            }
            const auto& face_neighbors = edge_face_neighbors[edge];
            if (face_neighbors.size() != count) {
                throw std::runtime_error("manifold_mesh_t: edge face neighbors "
                                         "size does not match edge count, "
                                         "this likely indicates bug in naga");
            }
            for (size_t i = 0; i < count; ++i) {
                mesh.edge_opposite_vertices(i, edge_idx) = opposite_vertices[i];
                mesh.edge_face_neighbors(i, edge_idx)    = face_neighbors[i];
            }
        }
        edge_counts.clear();
        edge_counts.rehash(0);
        edge_opposite_vertices.clear();
        edge_opposite_vertices.rehash(0);
        edge_face_neighbors.clear();
        edge_face_neighbors.rehash(0);

        mesh.vertex_face_neighbors = sclx::array<size_t, 2>{
            max_vertex_face_neighbors,
            total_vertex_count
        };
        sclx::fill(mesh.vertex_face_neighbors, no_face);
        mesh.vertex_edge_neighbors = sclx::array<size_t, 2>{
            max_vertex_edge_neighbors,
            total_vertex_count
        };
        sclx::fill(mesh.vertex_edge_neighbors, no_edge);
        mesh.vertex_opposite_edges = sclx::array<size_t, 2>{
            max_vertex_face_neighbors,
            total_vertex_count
        };
        sclx::fill(mesh.vertex_opposite_edges, no_edge);
        for (size_t i = 0; i < total_vertex_count; ++i) {
            if (vertex_face_neighbors[i].size()
                != vertex_opposite_edges[i].size()) {
                throw std::runtime_error(
                    "manifold_mesh_t: vertex face neighbor "
                    "and vertex opposite edge sizes do not match"
                );
            }
            for (size_t j = 0; j < vertex_face_neighbors[i].size(); ++j) {
                mesh.vertex_face_neighbors(j, i) = vertex_face_neighbors[i][j];
                mesh.vertex_opposite_edges(j, i) = vertex_opposite_edges[i][j];
            }
            for (size_t j = 0; j < vertex_edge_neighbors[i].size(); ++j) {
                mesh.vertex_edge_neighbors(j, i) = vertex_edge_neighbors[i][j];
            }
        }

        auto lower_bounds_reduce = sclx::algorithm::reduce_last_dim(
            mesh.vertices,
            std::numeric_limits<T>::max(),
            sclx::algorithm::min<>()
        );
        auto upper_bounds_reduce = sclx::algorithm::reduce_last_dim(
            mesh.vertices,
            std::numeric_limits<T>::lowest(),
            sclx::algorithm::max<>()
        );

        for (uint i = 0; i < 3; ++i) {
            mesh.lower_bound[i] = lower_bounds_reduce[i];
            mesh.upper_bound[i] = upper_bounds_reduce[i];
        }

        if (flip_normals) {
            sclx::algorithm::transform(
                mesh.normals,
                mesh.normals,
                -1,
                sclx::algorithm::multiplies<>{}
            )
                .get();
        }

        return mesh;
    }

    sclx::array<value_type, 2> vertices;
    sclx::array<value_type, 2> normals;
    sclx::array<size_t, 2> triangle_vert_indices;
    sclx::array<size_t, 1> triangle_normal_indices;
    sclx::array<size_t, 2> unique_edges;
    sclx::array<bool, 1> double_edges;
    sclx::array<size_t, 2> vertex_face_neighbors;
    sclx::array<size_t, 2> vertex_opposite_edges;
    sclx::array<size_t, 2> vertex_edge_neighbors;
    sclx::array<size_t, 2> edge_opposite_vertices;
    sclx::array<size_t, 2> edge_face_neighbors;

    naga::point_t<value_type, 3> lower_bound;
    naga::point_t<value_type, 3> upper_bound;
};

template<class T>
sdf_metadata build_sdf(const manifold_mesh_t<T>& mesh) {
    std::unique_ptr<sdf::Points> sdf_points
        = std::make_unique<sdf::Points>(mesh.vertices.shape()[1], 3);
    for (u_int32_t p = 0; p < mesh.vertices.shape()[1]; p++) {
        sdf_points.get()[0](p, 0) = mesh.vertices(0, p);
        sdf_points.get()[0](p, 1) = mesh.vertices(1, p);
        sdf_points.get()[0](p, 2) = mesh.vertices(2, p);
    }
    std::unique_ptr<sdf::Triangles> sdf_faces
        = std::make_unique<sdf::Triangles>(
            mesh.triangle_vert_indices.shape()[1],
            3
        );
    for (u_int32_t f = 0; f < mesh.triangle_vert_indices.shape()[1]; f++) {
        sdf_faces.get()[0](f, 0) = mesh.triangle_vert_indices(0, f);
        sdf_faces.get()[0](f, 1) = mesh.triangle_vert_indices(1, f);
        sdf_faces.get()[0](f, 2) = mesh.triangle_vert_indices(2, f);
    }
    return sdf_metadata{
        std::make_unique<sdf::SDF>(*sdf_points, *sdf_faces),
        std::move(sdf_points),
        std::move(sdf_faces)
    };
}

template<class T>
std::vector<T>
get_sdf_to_points(const sdf::Points& points, const sdf::SDF& surface) {
    auto sdfs = surface(points);
    return {sdfs.data(), sdfs.data() + sdfs.size()};
}

template<class T>
std::vector<int>
get_nearest_vert_to_points(const sdf::Points& points, const sdf::SDF& surface) {
    auto nearest_verts = surface.nn(points);
    return {nearest_verts.data(), nearest_verts.data() + nearest_verts.size()};
}

// returns points and normals
template<class T>
std::tuple<std::vector<point_t<T, 3>>, std::vector<point_t<T, 3>>>
fill_face_with_nodes(
    T nodal_spacing,
    size_t f,
    const triangular_mesh_t<T>& boundary_mesh,
    std::vector<int> excluded_edges = {},
    T min_dist_scale                = min_bound_dist_scale_2d,
    T min_edge_dist_scale           = min_bound_dist_scale_2d
) {
    using vector3_t = point_t<T, 3>;
    using normal3_t = point_t<T, 3>;
    using point3_t  = point_t<T, 3>;
    using point2_t  = point_t<T, 2>;

    point3_t v1;
    point3_t v2;
    point3_t v3;
    v1 = point3_t{
        {boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 0] * 3 + 0],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 0] * 3 + 1],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 0] * 3 + 2]}
    };
    v2 = point3_t{
        {boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 1] * 3 + 0],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 1] * 3 + 1],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 1] * 3 + 2]}
    };
    v3 = point3_t{
        {boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 2] * 3 + 0],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 2] * 3 + 1],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 2] * 3 + 2]}
    };

    vector3_t v12{{v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]}};
    vector3_t v13{{v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]}};
    vector3_t v23{{v3[0] - v2[0], v3[1] - v2[1], v3[2] - v2[2]}};
    normal3_t normal = naga::math::cross(v12, v13);
    naga::math::loopless::normalize<3>(normal);

    auto v1_normal
        = &boundary_mesh.normals()[3 * boundary_mesh.face_normals()[3 * f + 0]];
    auto v2_normal
        = &boundary_mesh.normals()[3 * boundary_mesh.face_normals()[3 * f + 1]];
    auto v3_normal
        = &boundary_mesh.normals()[3 * boundary_mesh.face_normals()[3 * f + 2]];
    vector3_t normals_sum{
        {v1_normal[0] + v2_normal[0] + v3_normal[0],
         v1_normal[1] + v2_normal[1] + v3_normal[1],
         v1_normal[2] + v2_normal[2] + v3_normal[2]}
    };
    if (naga::math::loopless::dot<3>(normals_sum, normal) < 0) {
        normal[0] *= -1;
        normal[1] *= -1;
        normal[2] *= -1;
    }

    auto x = v12;
    naga::math::loopless::normalize<3>(x);
    auto& z = normal;
    auto y  = naga::math::cross(z, x);
    naga::math::loopless::normalize<3>(y);

    std::vector<T> edge_verts2d;
    // v1
    edge_verts2d.push_back(0);
    edge_verts2d.push_back(0);
    // v2
    edge_verts2d.push_back(naga::math::loopless::dot<3>(v12, x));
    edge_verts2d.push_back(naga::math::loopless::dot<3>(v12, y));
    // v3
    edge_verts2d.push_back(naga::math::loopless::dot<3>(v13, x));
    edge_verts2d.push_back(naga::math::loopless::dot<3>(v13, y));

    point2_t v1_2d{{edge_verts2d[0], edge_verts2d[1]}};
    point2_t v2_2d{{edge_verts2d[2], edge_verts2d[3]}};
    point2_t v3_2d{{edge_verts2d[4], edge_verts2d[5]}};

    T area = 0.5
           * (-v2_2d.y() * v3_2d.x() + v1_2d.y() * (-v2_2d.x() + v3_2d.x())
              + v1_2d.x() * (v2_2d.y() - v3_2d.y()) + v2_2d.x() * v3_2d.y());

    if (std::isnan(area)) {
        return {{}, {}};
    }

    point2_t lower_bound2d{
        {std::min(edge_verts2d[0], std::min(edge_verts2d[2], edge_verts2d[4])),
         std::min(edge_verts2d[1], std::min(edge_verts2d[3], edge_verts2d[5]))}
    };

    point2_t upper_bound2d{
        {std::max(edge_verts2d[0], std::max(edge_verts2d[2], edge_verts2d[4])),
         std::max(edge_verts2d[1], std::max(edge_verts2d[3], edge_verts2d[5]))}
    };

    using namespace naga::experimental::mesh;

    auto v12_edge_normal
        = calc_v12_edge_normal_of_tri(&v1_2d[0], &v2_2d[0], &v3_2d[0]);
    auto v23_edge_normal
        = calc_v12_edge_normal_of_tri(&v2_2d[0], &v3_2d[0], &v1_2d[0]);
    auto v31_edge_normal
        = calc_v12_edge_normal_of_tri(&v3_2d[0], &v1_2d[0], &v2_2d[0]);

    std::vector<T> edge_vert_normals2d;
    // v1
    edge_vert_normals2d.push_back((v12_edge_normal[0] + v31_edge_normal[0]));
    edge_vert_normals2d.push_back((v12_edge_normal[1] + v31_edge_normal[1]));
    auto normal_ref = &edge_vert_normals2d[0];
    naga::math::loopless::normalize<2>(normal_ref);
    // v2
    edge_vert_normals2d.push_back((v12_edge_normal[0] + v23_edge_normal[0]));
    edge_vert_normals2d.push_back((v12_edge_normal[1] + v23_edge_normal[1]));
    normal_ref = &edge_vert_normals2d[2];
    naga::math::loopless::normalize<2>(normal_ref);
    // v3
    edge_vert_normals2d.push_back((v23_edge_normal[0] + v31_edge_normal[0]));
    edge_vert_normals2d.push_back((v23_edge_normal[1] + v31_edge_normal[1]));
    normal_ref = &edge_vert_normals2d[4];
    naga::math::loopless::normalize<2>(normal_ref);

    point2_t v1_normal2d{{edge_vert_normals2d[0], edge_vert_normals2d[1]}};
    point2_t v2_normal2d{{edge_vert_normals2d[2], edge_vert_normals2d[3]}};
    point2_t v3_normal2d{{edge_vert_normals2d[4], edge_vert_normals2d[5]}};

    using index_t = size_t;
    std::vector<index_t> edge1{0, 1};
    std::vector<index_t> edge2{1, 2};
    std::vector<index_t> edge3{2, 0};

    auto edge1_verts2d        = edge_verts2d;
    auto edge1_vert_normals2d = edge_vert_normals2d;
    auto edge2_verts2d        = edge_verts2d;
    auto edge2_vert_normals2d = edge_vert_normals2d;
    auto edge3_verts2d        = edge_verts2d;
    auto edge3_vert_normals2d = edge_vert_normals2d;

    std::vector<edge_info_t<T>> edge_metadata;

    {
        std::vector<edge_info_t<T>> edge1_metadata;
        subdivide_edges_and_cache_edge_info(
            nodal_spacing,
            edge1_verts2d,
            edge1_vert_normals2d,  // we dont really care about the
                                   // normals here
            edge1,
            edge1_metadata
        );
        edge_metadata.insert(
            edge_metadata.end(),
            edge1_metadata.begin(),
            edge1_metadata.end()
        );
    }
    {
        std::vector<edge_info_t<T>> edge2_metadata;
        subdivide_edges_and_cache_edge_info(
            nodal_spacing,
            edge2_verts2d,
            edge2_vert_normals2d,  // we dont really care about the
                                   // normals here
            edge2,
            edge2_metadata
        );
        edge_metadata.insert(
            edge_metadata.end(),
            edge2_metadata.begin(),
            edge2_metadata.end()
        );
    }
    {
        std::vector<edge_info_t<T>> edge3_metadata;
        subdivide_edges_and_cache_edge_info(
            nodal_spacing,
            edge3_verts2d,
            edge3_vert_normals2d,  // we dont really care about the
                                   // normals here
            edge3,
            edge3_metadata
        );
        edge_metadata.insert(
            edge_metadata.end(),
            edge3_metadata.begin(),
            edge3_metadata.end()
        );
    }

    edge_metadata[0].edge_normal[0] = -v12_edge_normal[0];
    edge_metadata[0].edge_normal[1] = -v12_edge_normal[1];
    edge_metadata[1].edge_normal[0] = -v23_edge_normal[0];
    edge_metadata[1].edge_normal[1] = -v23_edge_normal[1];
    edge_metadata[2].edge_normal[0] = -v31_edge_normal[0];
    edge_metadata[2].edge_normal[1] = -v31_edge_normal[1];

    auto potential_face_points2d = generate_2d_hexagonal_grid(
        nodal_spacing,
        lower_bound2d,
        upper_bound2d
    );

    //    // we also add jitter in hopes it prevents artifacting
    //    static std::default_random_engine offset_rng(2);
    //    // use same seed for reproducibility
    //    std::normal_distribution<T> offset_dist(0.0, 0.05 * nodal_spacing);
    //    static std::default_random_engine angle_rng(456);
    //    // use same seed for reproducibility
    //    std::uniform_real_distribution<double>
    //        angle_dist(0., naga::math::pi<double>);
    //    for (auto& p : potential_face_points2d) {
    //        auto offset = offset_dist(offset_rng);
    //        auto angle  = angle_dist(angle_rng);
    //        p[0] += offset * naga::math::cos(angle);
    //        p[1] += offset * naga::math::sin(angle);
    //    }

    auto distance_to_edges = remove_points_outside_2d_contours(
        potential_face_points2d,
        nodal_spacing,
        edge_metadata,
        min_dist_scale
    );

    //    if (potential_face_points2d.empty()) {
    //        potential_face_points2d.push_back(point2_t{
    //            {v1_2d.x() + v2_2d.x() + v3_2d.x(),
    //             v1_2d.y() + v2_2d.y() + v3_2d.y()}});
    //        potential_face_points2d.back()[0] /= 3.;
    //        potential_face_points2d.back()[1] /= 3.;
    //
    //        auto point = potential_face_points2d.back();
    //
    //        distance_to_edges = remove_points_outside_2d_contours(
    //            potential_face_points2d,
    //            nodal_spacing,
    //            edge_metadata,
    //            min_dist_scale
    //        );
    //
    //        if (potential_face_points2d.empty()) {
    //            potential_face_points2d.push_back(point);
    //        }
    //    }

    std::vector<point3_t> barycentric_coordinates;
    barycentric_coordinates.reserve(potential_face_points2d.size());
    std::transform(
        potential_face_points2d.begin(),
        potential_face_points2d.end(),
        std::back_inserter(barycentric_coordinates),
        [&](const auto& p) {
            // barycentric coordinates
            T s = 1. / (2. * area)
                * ((v1_2d.y() * v3_2d.x() - v1_2d.x() * v3_2d.y())
                   + (v3_2d.y() - v1_2d.y()) * p.x()
                   + (v1_2d.x() - v3_2d.x()) * p.y());
            T t = 1. / (2. * area)
                * ((v1_2d.x() * v2_2d.y() - v1_2d.y() * v2_2d.x())
                   + (v1_2d.y() - v2_2d.y()) * p.x()
                   + (v2_2d.x() - v1_2d.x()) * p.y());
            T u = 1. - s - t;
            return point3_t{{s, t, u}};
        }
    );

    std::vector<point3_t> points3d;
    points3d.reserve(potential_face_points2d.size());
    std::transform(
        barycentric_coordinates.begin(),
        barycentric_coordinates.end(),
        std::back_inserter(points3d),
        [&](const auto& b) {
            return point3_t{
                {v1[0] + b[0] * v12[0] + b[1] * v13[0],
                 v1[1] + b[0] * v12[1] + b[1] * v13[1],
                 v1[2] + b[0] * v12[2] + b[1] * v13[2]}
            };
        }
    );
    std::vector<normal3_t> normals3d(points3d.size(), normal);

    //    std::vector<point3_t> edge_points3d;
    //    edge_points3d.reserve(edge_verts2d.size() / 2);
    //    for (size_t i = 6; i < edge1_verts2d.size(); i += 2) {
    //        if (std::find(excluded_edges.begin(), excluded_edges.end(), 0)
    //            != excluded_edges.end()) {
    //            break;
    //        }
    //        point2_t p{{edge1_verts2d[i + 0], edge1_verts2d[i + 1]}};
    //        {
    //            auto [distance_p_to_edge2, unused1]
    //                = distance_to_edge(p, edge_metadata[1]);
    //            if (std::abs(distance_p_to_edge2)
    //                < min_edge_dist_scale * nodal_spacing) {
    //                continue;
    //            }
    //        }
    //        {
    //            auto [distance_p_to_edge3, unused1]
    //                = distance_to_edge(p, edge_metadata[2]);
    //            if (std::abs(distance_p_to_edge3)
    //                < min_edge_dist_scale * nodal_spacing) {
    //                continue;
    //            }
    //        }
    //        edge_points3d.push_back(point3_t{{p[0], p[1], 0.}});
    //    }
    //    for (size_t i = 6; i < edge2_verts2d.size(); i += 2) {
    //        if (std::find(excluded_edges.begin(), excluded_edges.end(), 1)
    //            != excluded_edges.end()) {
    //            break;
    //        }
    //        point2_t p{{edge2_verts2d[i + 0], edge2_verts2d[i + 1]}};
    //        {
    //            auto [distance_p_to_edge1, unused2]
    //                = distance_to_edge(p, edge_metadata[0]);
    //            if (std::abs(distance_p_to_edge1)
    //                < min_edge_dist_scale * nodal_spacing) {
    //                continue;
    //            }
    //        }
    //        {
    //            auto [distance_p_to_edge3, unused2]
    //                = distance_to_edge(p, edge_metadata[2]);
    //            if (std::abs(distance_p_to_edge3)
    //                < min_edge_dist_scale * nodal_spacing) {
    //                continue;
    //            }
    //        }
    //        edge_points3d.push_back(point3_t{{p[0], p[1], 0.}});
    //    }
    //    for (size_t i = 6; i < edge3_verts2d.size(); i += 2) {
    //        if (std::find(excluded_edges.begin(), excluded_edges.end(), 2)
    //            != excluded_edges.end()) {
    //            break;
    //        }
    //        point2_t p{{edge3_verts2d[i + 0], edge3_verts2d[i + 1]}};
    //        {
    //            auto [distance_p_to_edge1, unused3]
    //                = distance_to_edge(p, edge_metadata[0]);
    //            if (std::abs(distance_p_to_edge1)
    //                < min_edge_dist_scale * nodal_spacing) {
    //                continue;
    //            }
    //        }
    //        {
    //            auto [distance_p_to_edge2, unused3]
    //                = distance_to_edge(p, edge_metadata[1]);
    //            if (std::abs(distance_p_to_edge2)
    //                < min_edge_dist_scale * nodal_spacing) {
    //                continue;
    //            }
    //        }
    //        edge_points3d.push_back(point3_t{{p[0], p[1], 0.}});
    //    }
    //
    //    barycentric_coordinates.clear();
    //    barycentric_coordinates.reserve(edge_points3d.size());
    //    std::transform(
    //        edge_points3d.begin(),
    //        edge_points3d.end(),
    //        std::back_inserter(barycentric_coordinates),
    //        [&](const auto& p) {
    //            // barycentric coordinates
    //            T s = 1. / (2. * area)
    //                     * ((v1_2d.y() * v3_2d.x() - v1_2d.x() * v3_2d.y())
    //                        + (v3_2d.y() - v1_2d.y()) * p.x()
    //                        + (v1_2d.x() - v3_2d.x()) * p.y());
    //            T t = 1. / (2. * area)
    //                     * ((v1_2d.x() * v2_2d.y() - v1_2d.y() * v2_2d.x())
    //                        + (v1_2d.y() - v2_2d.y()) * p.x()
    //                        + (v2_2d.x() - v1_2d.x()) * p.y());
    //            T u = 1. - s - t;
    //            return point3_t{{s, t, u}};
    //        }
    //    );
    //
    //    std::transform(
    //        edge_points3d.begin(),
    //        edge_points3d.end(),
    //        edge_points3d.begin(),
    //        [&](const auto& p) {
    //            return point3_t{
    //                {p[0] * x[0] + p[1] * y[0] + v1[0],
    //                 p[0] * x[1] + p[1] * y[1] + v1[1],
    //                 p[0] * x[2] + p[1] * y[2] + v1[2]}};
    //        }
    //    );
    //
    //    std::vector<point3_t> edge_normals3d;
    //    std::transform(
    //        barycentric_coordinates.begin(),
    //        barycentric_coordinates.end(),
    //        std::back_inserter(edge_normals3d),
    //        [&](const auto& b) {
    //            int closest_verts[2]{0, 1};
    //            if (b[1] < b[0]) {
    //                std::swap(closest_verts[0], closest_verts[1]);
    //            }
    //            if (b[2] < b[closest_verts[1]]) {
    //                if (b[2] < b[closest_verts[0]]) {
    //                    closest_verts[1] = closest_verts[0];
    //                    closest_verts[0] = 2;
    //                } else {
    //                    closest_verts[1] = 2;
    //                }
    //            }
    //            normal3_t average_normal{};
    //            for (int closest_vert : closest_verts) {
    //                if (closest_vert == 0) {
    //                    average_normal[0] += v1_normal[0];
    //                    average_normal[1] += v1_normal[1];
    //                    average_normal[2] += v1_normal[2];
    //                } else if (closest_vert == 1) {
    //                    average_normal[0] += v2_normal[0];
    //                    average_normal[1] += v2_normal[1];
    //                    average_normal[2] += v2_normal[2];
    //                } else if (closest_vert == 2) {
    //                    average_normal[0] += v3_normal[0];
    //                    average_normal[1] += v3_normal[1];
    //                    average_normal[2] += v3_normal[2];
    //                }
    //            }
    //            naga::math::loopless::normalize<3>(average_normal);
    //            return average_normal;
    //        }
    //    );
    //
    //    points3d.insert(points3d.end(), edge_points3d.begin(),
    //    edge_points3d.end()); normals3d
    //        .insert(normals3d.end(), edge_normals3d.begin(),
    //        edge_normals3d.end());

    return {points3d, normals3d};
}

template<class T>
void fill_surface_with_nodes(
    std::vector<point_t<T, 3>>& points,
    std::vector<point_t<T, 3>>& normals,
    double nodal_spacing,
    const triangular_mesh_t<T>& boundary_mesh
) {
    using point_t            = point_t<T, 3>;
    unsigned int max_threads = std::thread::hardware_concurrency();

    std::vector<hashable_edge> processed_edges(boundary_mesh.faces().size());
    std::vector<std::atomic<bool>> busy_edges(boundary_mesh.faces().size());
    std::fill(busy_edges.begin(), busy_edges.end(), true);
    std::atomic<uint> processed_edges_size{0};

    unsigned int processed_edge_cleanup_interval = max_threads * 10;
    std::vector<std::vector<size_t>> indices_to_erase(max_threads);

    std::vector<std::vector<point_t>> points_to_insert(max_threads);
    std::vector<std::vector<point_t>> normals_to_insert(max_threads);

    std::vector<std::future<size_t>> futures;

    for (size_t f = 0; f < boundary_mesh.faces().size() / 3; f++) {
        if (futures.size() == max_threads) {
            {
                auto processed_face = futures[0].get();
                std::cout
                    << "\rprocessed faces: " << processed_face + 1 << " of "
                    << boundary_mesh.faces().size() / 3 << " with "
                    << points_to_insert[processed_face % max_threads].size()
                    << " points" << std::flush;
                futures.erase(futures.begin());

                points.insert(
                    points.end(),
                    points_to_insert[processed_face % max_threads].begin(),
                    points_to_insert[processed_face % max_threads].end()
                );
                normals.insert(
                    normals.end(),
                    normals_to_insert[processed_face % max_threads].begin(),
                    normals_to_insert[processed_face % max_threads].end()
                );
            }
        }

        std::promise<size_t> fill_promise;
        auto fill_fut = fill_promise.get_future();

        std::thread([&, f, fill_promise = std::move(fill_promise)]() mutable {
            std::vector<int> excluded_edges;

            for (int e = 0; e < 3; ++e) {
                auto edge = hashable_edge{
                    boundary_mesh.faces()[f * 3 + e],
                    boundary_mesh.faces()[f * 3 + (e + 1) % 3]
                };
                auto old_processed_size = processed_edges_size.fetch_add(1);
                processed_edges[old_processed_size] = edge;
                auto end_search_it
                    = processed_edges.begin() + old_processed_size;
                auto found_it = [&]() {
                    for (auto it = processed_edges.begin(); it < end_search_it;
                         ++it) {
                        auto& busy_ref = busy_edges
                            [std::distance(processed_edges.begin(), it)];
                        while (busy_ref.load()) {
                            std::this_thread::yield();
                        }
                        if (*it == edge) {
                            return it;
                        }
                    }
                    return end_search_it;
                }();
                busy_edges[old_processed_size].store(false);
                if (found_it != end_search_it) {
                    excluded_edges.push_back(e);
                    //                    indices_to_erase[f %
                    //                    max_threads].push_back(
                    //                        std::distance(processed_edges.begin(),
                    //                        found_it) - 1
                    //                    );
                }
            }
            auto [face_points, face_normals] = fill_face_with_nodes<T>(
                nodal_spacing,
                f,
                boundary_mesh,
                excluded_edges,
                .375f,
                .05f
            );

            points_to_insert[f % max_threads]  = std::move(face_points);
            normals_to_insert[f % max_threads] = std::move(face_normals);

            fill_promise.set_value(f);
        }).detach();

        futures.push_back(std::move(fill_fut));
    }

    for (auto& fut : futures) {
        auto processed_face = fut.get();
        std::cout << "\rprocessed faces: " << processed_face + 1 << " of "
                  << boundary_mesh.faces().size() / 3 << std::flush;

        points.insert(
            points.end(),
            points_to_insert[processed_face % max_threads].begin(),
            points_to_insert[processed_face % max_threads].end()
        );

        normals.insert(
            normals.end(),
            normals_to_insert[processed_face % max_threads].begin(),
            normals_to_insert[processed_face % max_threads].end()
        );
    }
    futures.clear();

    if (points.size() > boundary_mesh.vertices().size() / 2) {
        return;
    }
    for (size_t i = 0; i < boundary_mesh.vertices().size(); i += 3) {
        std::vector<point_t> shared_normals;
        auto start_search = boundary_mesh.faces().begin();
        while (start_search != boundary_mesh.faces().end()) {
            auto found_it
                = std::find(start_search, boundary_mesh.faces().end(), i / 3);
            if (found_it == boundary_mesh.faces().end()) {
                break;
            }
            auto face_index
                = std::distance(boundary_mesh.faces().begin(), found_it) / 3;
            auto vertex_index
                = std::distance(boundary_mesh.faces().begin(), found_it) % 3;
            auto normal_index
                = boundary_mesh.face_normals()[face_index * 3 + vertex_index];
            shared_normals.push_back(point_t{
                {boundary_mesh.normals()[normal_index * 3 + 0],
                 boundary_mesh.normals()[normal_index * 3 + 1],
                 boundary_mesh.normals()[normal_index * 3 + 2]}
            });

            start_search = found_it + 1;
        }
        point_t average_normal{{0., 0., 0.}};
        for (const auto& normal : shared_normals) {
            average_normal[0] += normal[0];
            average_normal[1] += normal[1];
            average_normal[2] += normal[2];
        }
        naga::math::loopless::normalize<3>(average_normal);
        points.push_back(point_t{
            {boundary_mesh.vertices()[i + 0],
             boundary_mesh.vertices()[i + 1],
             boundary_mesh.vertices()[i + 2]}
        });
        normals.push_back(average_normal);
    }
}

template<class T>
class conforming_point_cloud_impl_t<T, 3> {
  public:
    using point_t  = naga::point_t<T, 3>;
    using normal_t = point_t;
    using index_t  = size_t;

    using triangular_mesh_t =
        typename conforming_point_cloud_t<T, 3>::input_domain_data_t;

    static conforming_point_cloud_impl_t create(
        const T& nodal_spacing,
        const std::filesystem::path& domain_obj,
        const std::vector<std::filesystem::path>& immersed_boundary_objs = {}
    ) {
        auto domain_mesh = manifold_mesh_t<T>::import_from_obj(domain_obj);
        std::vector<manifold_mesh_t<T>> immersed_boundary_meshes;
        for (const auto& immersed_boundary_obj : immersed_boundary_objs) {
            immersed_boundary_meshes.push_back(
                manifold_mesh_t<T>::import_from_obj(immersed_boundary_obj)
            );
        }

        sdf_metadata domain_sdf = build_sdf(domain_mesh);
        std::vector<sdf_metadata> immersed_boundary_sdfs;
        for (const auto& immersed_boundary_mesh : immersed_boundary_meshes) {
            immersed_boundary_sdfs.push_back(build_sdf(immersed_boundary_mesh));
        }

        sclx::algorithm::transform(
            domain_mesh.normals,
            domain_mesh.normals,
            -1.f,
            sclx::algorithm::multiplies<>{}
        );

        point_t lower_bound{
            {domain_mesh.vertices[0],
             domain_mesh.vertices[1],
             domain_mesh.vertices[2]}
        };
        point_t upper_bound{
            {domain_mesh.vertices[0],
             domain_mesh.vertices[1],
             domain_mesh.vertices[2]}
        };
        for (size_t i = 0; i < domain_mesh.vertices.elements(); i += 3) {
            lower_bound[0]
                = std::min(lower_bound[0], domain_mesh.vertices[i + 0]);
            lower_bound[1]
                = std::min(lower_bound[1], domain_mesh.vertices[i + 1]);
            lower_bound[2]
                = std::min(lower_bound[2], domain_mesh.vertices[i + 2]);
            upper_bound[0]
                = std::max(upper_bound[0], domain_mesh.vertices[i + 0]);
            upper_bound[1]
                = std::max(upper_bound[1], domain_mesh.vertices[i + 1]);
            upper_bound[2]
                = std::max(upper_bound[2], domain_mesh.vertices[i + 2]);
        }
        lower_bound[0] -= 2.f * nodal_spacing;
        lower_bound[1] -= 2.f * nodal_spacing;
        lower_bound[2] -= 2.f * nodal_spacing;
        upper_bound[0] += 2.f * nodal_spacing;
        upper_bound[1] += 2.f * nodal_spacing;
        upper_bound[2] += 2.f * nodal_spacing;
        size_t approx_grid_size[3]{
            static_cast<size_t>(
                std::ceil((upper_bound[0] - lower_bound[0]) / nodal_spacing)
            ),
            static_cast<size_t>(
                std::ceil((upper_bound[1] - lower_bound[1]) / nodal_spacing)
            ),
            static_cast<size_t>(
                std::ceil((upper_bound[2] - lower_bound[2]) / nodal_spacing)
            )
        };

        std::vector<point_t> bulk_points;
        std::vector<T> bulk_to_boundary_distances;
        std::vector<index_t> closest_boundary_to_bulk;
        std::vector<int> closest_vertices_in_boundary;

        auto potential_grid_size = 2 * approx_grid_size[0] * approx_grid_size[1]
                                 * approx_grid_size[2];
        size_t batch_bytes        = naga::math::loopless::pow<30>(size_t{2});
        size_t elements_per_batch = batch_bytes / sizeof(point_t);
        size_t num_batches = (potential_grid_size + elements_per_batch - 1)
                           / elements_per_batch;
        for (size_t b = 0; b < num_batches; ++b) {
            sdf::Points potential_bulk_points;
            std::vector<point_t> fill_points(std::min(
                elements_per_batch,
                potential_grid_size - b * elements_per_batch
            ));
            potential_bulk_points = sdf::Points(fill_points.size(), 3);
#pragma omp parallel for
            for (size_t p = 0; p < fill_points.size(); ++p) {
                auto linear_id = (p / 2) + b * elements_per_batch;
                auto i         = linear_id % approx_grid_size[0];
                auto j
                    = (linear_id / approx_grid_size[0]) % approx_grid_size[1];
                auto k
                    = (linear_id / (approx_grid_size[0] * approx_grid_size[1]))
                    % (approx_grid_size[2]);
                point_t new_point{
                    {lower_bound[0] + static_cast<T>(i) * nodal_spacing,
                     lower_bound[1] + static_cast<T>(j) * nodal_spacing,
                     lower_bound[2] + static_cast<T>(k) * nodal_spacing}
                };
                if (p % 2 == 1) {
                    new_point[0] += nodal_spacing / 2.f;
                    new_point[1] += nodal_spacing / 2.f;
                    new_point[2] += naga::math::sqrt(2) * nodal_spacing / 2.f;
                }
                fill_points[p] = new_point;

                potential_bulk_points(static_cast<uint>(p), 0)
                    = static_cast<float>(new_point[0]);
                potential_bulk_points(static_cast<uint>(p), 1)
                    = static_cast<float>(new_point[1]);
                potential_bulk_points(static_cast<uint>(p), 2)
                    = static_cast<float>(new_point[2]);
            }

            std::vector<T> min_distance_to_boundary
                = get_sdf_to_points<T>(potential_bulk_points, *domain_sdf.sdf);
            auto closest_vertices_in_boundary_batch
                = get_nearest_vert_to_points<T>(
                    potential_bulk_points,
                    *domain_sdf.sdf
                );
            std::vector<uint> closest_boundary_indices(
                potential_bulk_points.rows(),
                0
            );

            uint boundary_index = 1;
            for (const auto& sdf : immersed_boundary_sdfs) {
                std::vector<T> distance_to_boundary
                    = get_sdf_to_points<T>(potential_bulk_points, *sdf.sdf);
                std::vector<int> closest_vertices
                    = get_nearest_vert_to_points<T>(
                        potential_bulk_points,
                        *domain_sdf.sdf
                    );
#pragma omp parallel for
                for (size_t i = 0; i < distance_to_boundary.size(); ++i) {
                    distance_to_boundary[i] *= -1;
                    constexpr T epsilon      = 1e-6;
                    const auto& distance     = distance_to_boundary[i];
                    const auto& min_distance = min_distance_to_boundary[i];
                    if (std::abs(std::abs(distance) - std::abs(min_distance))
                        < epsilon) {
                        closest_boundary_indices[i]
                            = distance < min_distance
                                ? boundary_index
                                : closest_boundary_indices[i];
                    } else if (std::abs(distance) < std::abs(min_distance)) {
                        closest_boundary_indices[i] = boundary_index;
                    }
                }
                std::transform(
                    distance_to_boundary.begin(),
                    distance_to_boundary.end(),
                    min_distance_to_boundary.begin(),
                    [&](const T& distance) {
                        size_t i = &distance - &distance_to_boundary[0];
                        return closest_boundary_indices[i] == boundary_index
                                 ? distance
                                 : min_distance_to_boundary[i];
                    }
                );
                std::transform(
                    closest_vertices.begin(),
                    closest_vertices.end(),
                    closest_vertices_in_boundary_batch.begin(),
                    [&](const int& closest_vertex) {
                        size_t i = &closest_vertex - &closest_vertices[0];
                        return closest_vertices_in_boundary_batch[i]
                                    == boundary_index
                                 ? closest_vertex
                                 : closest_vertices_in_boundary_batch[i];
                    }
                );
                boundary_index++;
            }

            std::mutex bulk_points_mutex;
#pragma omp parallel for
            for (uint i = 0; i < potential_bulk_points.rows(); i++) {
                const auto& distance_to_boundary = min_distance_to_boundary[i];
                if (distance_to_boundary
                    < min_bound_dist_scale_3d * nodal_spacing) {
                    continue;
                }

                uint bulk_index = 0;
                {
                    std::lock_guard<std::mutex> lock(bulk_points_mutex);
                    bulk_points.emplace_back();
                    bulk_index = bulk_points.size() - 1;
                    bulk_to_boundary_distances.push_back(distance_to_boundary);
                    closest_vertices_in_boundary.push_back(static_cast<index_t>(
                        closest_vertices_in_boundary_batch[i]
                    ));
                    closest_boundary_to_bulk.push_back(
                        closest_boundary_indices[i]
                    );
                }

                bulk_points[bulk_index][0] = potential_bulk_points(i, 0);
                bulk_points[bulk_index][1] = potential_bulk_points(i, 1);
                bulk_points[bulk_index][2] = potential_bulk_points(i, 2);
            }
        }

        std::vector<point_t> boundary_points;
        std::vector<normal_t> boundary_normals;
        std::vector<size_t> bulk_indices_to_erase;
        for (size_t i = 0; i < bulk_points.size(); ++i) {
            if (bulk_to_boundary_distances[i]
                    >= min_bound_dist_scale_3d * nodal_spacing
                && bulk_to_boundary_distances[i]
                       <= max_bound_dist_scale_3d * nodal_spacing) {

                boundary_points.push_back(bulk_points[i]);
                const auto& boundary_point = boundary_points.back();
                auto mesh                  = &domain_mesh;
                if (closest_boundary_to_bulk[i] != 0) {
                    mesh = &immersed_boundary_meshes
                               [closest_boundary_to_bulk[i] - 1];
                }

                boundary_normals.emplace_back();
                bulk_indices_to_erase.push_back(i);
                auto& face_normal = boundary_normals.back();

                auto closest_vertex_index  = closest_vertices_in_boundary[i];
                auto closest_face_distance = std::numeric_limits<T>::max();
                size_t closest_face_idx;

                for (size_t n = 0; n < mesh->vertex_face_neighbors.shape()[0];
                     ++n) {
                    auto f
                        = mesh->vertex_face_neighbors(n, closest_vertex_index);
                    if (f == manifold_mesh_t<T>::no_face) {
                        continue;
                    }
                    naga::point_t<T, 3> centroid{{0, 0, 0}};
                    for (size_t v = 0; v < 3; ++v) {
                        centroid[0] += mesh->vertices(
                            0,
                            mesh->triangle_vert_indices(v, f)
                        );
                        centroid[1] += mesh->vertices(
                            1,
                            mesh->triangle_vert_indices(v, f)
                        );
                        centroid[2] += mesh->vertices(
                            2,
                            mesh->triangle_vert_indices(v, f)
                        );
                    }
                    centroid[0] /= 3;
                    centroid[1] /= 3;
                    centroid[2] /= 3;

                    auto distance
                        = naga::distance_functions::loopless::euclidean<3>{
                        }(boundary_point, centroid);
                    if (distance < closest_face_distance) {
                        closest_face_distance = distance;
                        closest_face_idx      = f;
                    }
                }
                face_normal = &mesh->normals(
                    0,
                    mesh->triangle_normal_indices(closest_face_idx)
                );

//                if (bulk_to_boundary_distances[i] < 0.f) {
//                    auto new_position_x = boundary_points.back()[0]
//                                        - boundary_normals.back()[0]
//                                              * bulk_to_boundary_distances[i];
//                    auto new_position_y = boundary_points.back()[1]
//                                        - boundary_normals.back()[1]
//                                              * bulk_to_boundary_distances[i];
//                    auto new_position_z = boundary_points.back()[2]
//                                        - boundary_normals.back()[2]
//                                              * bulk_to_boundary_distances[i];
//                    boundary_points.back()[0]     = new_position_x;
//                    boundary_points.back()[1]     = new_position_y;
//                    boundary_points.back()[2]     = new_position_z;
//                    bulk_to_boundary_distances[i] = 0.f;
//                }
            }
        }
        std::sort(
            bulk_indices_to_erase.begin(),
            bulk_indices_to_erase.end(),
            std::greater<>{}
        );
        for (const auto& i : bulk_indices_to_erase) {
            bulk_points.erase(bulk_points.begin() + i);
            bulk_to_boundary_distances.erase(
                bulk_to_boundary_distances.begin() + i
            );
            closest_boundary_to_bulk.erase(
                closest_boundary_to_bulk.begin() + i
            );
        }

        return {
            std::move(bulk_points),
            std::move(bulk_to_boundary_distances),
            std::move(closest_boundary_to_bulk),
            std::move(boundary_points),
            std::move(boundary_normals)
        };
    }

    const std::vector<point_t>& bulk_points() const { return bulk_points_; }

    const std::vector<T>& bulk_to_boundary_distances() const {
        return bulk_to_boundary_distances_;
    }

    const std::vector<index_t>& closest_boundary_to_bulk() const {
        return closest_boundary_to_bulk_;
    }

    const std::vector<point_t>& boundary_points() const {
        return boundary_points_;
    }

    const std::vector<normal_t>& boundary_normals() const {
        return boundary_normals_;
    }

  private:
    conforming_point_cloud_impl_t(
        std::vector<point_t>&& bulk_points,
        std::vector<T>&& bulk_to_boundary_distances,
        std::vector<index_t>&& closest_boundary_to_bulk,
        std::vector<point_t>&& boundary_points,
        std::vector<normal_t>&& boundary_normals
    )
        : bulk_points_(std::move(bulk_points)),
          bulk_to_boundary_distances_(std::move(bulk_to_boundary_distances)),
          closest_boundary_to_bulk_(std::move(closest_boundary_to_bulk)),
          boundary_points_(std::move(boundary_points)),
          boundary_normals_(std::move(boundary_normals)) {}

    std::vector<point_t> bulk_points_;
    std::vector<T> bulk_to_boundary_distances_;
    std::vector<index_t> closest_boundary_to_bulk_;
    std::vector<point_t> boundary_points_;
    std::vector<normal_t> boundary_normals_;
};

template<class T>
conforming_point_cloud_t<T, 3> conforming_point_cloud_t<T, 3>::create(
    const T& approximate_spacing,
    const std::filesystem::path& domain,
    const std::vector<std::filesystem::path>& immersed_boundaries
) {
    conforming_point_cloud_t point_cloud;
    auto impl = conforming_point_cloud_impl_t<T, dimensions>::create(
        approximate_spacing,
        domain,
        immersed_boundaries
    );
    auto impl_ptr
        = std::make_shared<conforming_point_cloud_impl_t<T, dimensions>>(
            std::move(impl)
        );
    point_cloud.impl = std::move(impl_ptr);
    return point_cloud;
}

template<class T>
const std::vector<typename conforming_point_cloud_t<T, 3>::point_t>&
conforming_point_cloud_t<T, 3>::bulk_points() const {
    return impl->bulk_points();
}

template<class T>
const std::vector<T>&
conforming_point_cloud_t<T, 3>::bulk_to_boundary_distances() const {
    return impl->bulk_to_boundary_distances();
}

template<class T>
const std::vector<typename conforming_point_cloud_t<T, 3>::index_t>&
conforming_point_cloud_t<T, 3>::closest_boundary_to_bulk() const {
    return impl->closest_boundary_to_bulk();
}

template<class T>
const std::vector<typename conforming_point_cloud_t<T, 3>::point_t>&
conforming_point_cloud_t<T, 3>::boundary_points() const {
    return impl->boundary_points();
}

template<class T>
const std::vector<typename conforming_point_cloud_t<T, 3>::normal_t>&
conforming_point_cloud_t<T, 3>::boundary_normals() const {
    return impl->boundary_normals();
}

template class conforming_point_cloud_t<float, 3>;
template class conforming_point_cloud_t<double, 3>;

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail
