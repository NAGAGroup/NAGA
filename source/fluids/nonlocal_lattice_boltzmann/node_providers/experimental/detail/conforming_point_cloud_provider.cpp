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
#include <naga/fluids/nonlocal_lattice_boltzmann/node_providers/experimental/detail/conforming_point_cloud_provider.hpp>
#include <numeric>
#include <random>
#include <utility>

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

struct edge_info_t {
    using point_t  = naga::point_t<double, 2>;
    using vector_t = naga::point_t<double, 2>;
    using normal_t = naga::point_t<double, 2>;

    point_t v1;
    point_t v2;

    vector_t v12;
    vector_t normalized_edge_dir;
    double length;

    normal_t n1;
    normal_t n2;
    normal_t edge_normal;
};

edge_info_t get_edge_info(
    size_t edge_index,
    const std::vector<double>& vertices,
    const std::vector<double>& vertex_normals,
    const std::vector<size_t>& edges
) {
    edge_info_t edge_info{};
    using point_t  = edge_info_t::point_t;
    using vector_t = edge_info_t::vector_t;
    using normal_t = edge_info_t::normal_t;
    edge_info.v1   = point_t{
          {vertices[edges[edge_index] * 2 + 0],
           vertices[edges[edge_index] * 2 + 1]}};
    edge_info.v2 = point_t{
        {vertices[edges[edge_index + 1] * 2 + 0],
         vertices[edges[edge_index + 1] * 2 + 1]}};

    edge_info.v12 = vector_t{
        {edge_info.v2[0] - edge_info.v1[0], edge_info.v2[1] - edge_info.v1[1]}};
    edge_info.length = naga::math::loopless::norm<2>(edge_info.v12);
    edge_info.normalized_edge_dir = vector_t{
        {edge_info.v12[0] / edge_info.length,
         edge_info.v12[1] / edge_info.length}};

    edge_info.n1 = normal_t{
        {vertex_normals[edges[edge_index] * 2 + 0],
         vertex_normals[edges[edge_index] * 2 + 1]}};
    naga::math::loopless::normalize<2>(edge_info.n1);
    edge_info.n2 = normal_t{
        {vertex_normals[edges[edge_index + 1] * 2 + 0],
         vertex_normals[edges[edge_index + 1] * 2 + 1]}};
    naga::math::loopless::normalize<2>(edge_info.n2);
    edge_info.edge_normal = normal_t{
        {(edge_info.n1[0] + edge_info.n2[0]) / 2.,
         (edge_info.n1[1] + edge_info.n2[1]) / 2.}};
    naga::math::loopless::normalize<2>(edge_info.edge_normal);

    return edge_info;
}

void subdivide_edges_and_cache_edge_info(
    double target_length,
    std::vector<double>& vertices,
    std::vector<double>& vertex_normals,
    std::vector<size_t>& edges,
    std::vector<edge_info_t>& input_edge_info
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

        double edge_len2target_len = edge_info.length / target_length;
        double acceptable_epsilon  = 0.1;
        if (edge_len2target_len < 1 + acceptable_epsilon) {
            continue;
        }
        size_t num_nodes_to_add
            = static_cast<size_t>(std::ceil(edge_len2target_len)) - 1;
        double actual_length
            = edge_info.length / static_cast<double>(num_nodes_to_add + 1);

        size_t old_edge_end = edges_copy[e + 1];
        naga::point_t<double, 2> new_vertex{
            {edge_info.v1[0] + edge_info.normalized_edge_dir[0] * actual_length,
             edge_info.v1[1]
                 + edge_info.normalized_edge_dir[1] * actual_length}};
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

class line_t {
  public:
    using point_t = naga::point_t<double, 2>;
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

    double slope() { return (p2_[1] - p1_[1]) / (p2_[0] - p1_[0]); }

    const point_t& p1() const { return p1_; }
    const point_t& p2() const { return p2_; }

  private:
    point_t p1_;
    point_t p2_;
};

int sgn(double x) { return static_cast<int>(x > 0) - static_cast<int>(x < 0); }

std::tuple<double, double> distance_to_edge(
    const naga::point_t<double, 2>& xi,
    const edge_info_t& edge_info
) {
    using vector_t = naga::point_t<double, 2>;
    line_t edge_line{edge_info.v1, edge_info.v2};
    auto perp_line = edge_line.perpendicular_line(xi);
    vector_t xi_to_intersection{
        {perp_line.p2()[0] - xi[0], perp_line.p2()[1] - xi[1]}};
    auto signed_distance_to_line
        = sgn(naga::math::loopless::dot<2>(
              xi_to_intersection,
              edge_info.edge_normal
          ))
        * naga::math::loopless::norm<2>(xi_to_intersection) * -1;

    vector_t v1_to_intersection{
        {perp_line.p2()[0] - edge_info.v1[0],
         perp_line.p2()[1] - edge_info.v1[1]}};
    auto signed_v1_intersection_distance = naga::math::loopless::dot<2>(
        v1_to_intersection,
        edge_info.normalized_edge_dir
    );

    if (signed_v1_intersection_distance < 0) {
        vector_t v1_to_xi{{xi[0] - edge_info.v1[0], xi[1] - edge_info.v1[1]}};
        auto v1_to_xi_dot_normal
            = naga::math::loopless::dot<2>(v1_to_xi, edge_info.edge_normal);
        double incident_angle = naga::math::asin(
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
        double angle_from_perp_line = naga::math::asin(
            naga::math::abs(v2_to_xi_dot_normal)
            / naga::math::loopless::norm<2>(v2_to_xi)
        );
        auto signed_distance_to_v2 = sgn(v2_to_xi_dot_normal)
                                   * naga::math::loopless::norm<2>(v2_to_xi);
        return {signed_distance_to_v2, angle_from_perp_line};
    }
    return {signed_distance_to_line, naga::math::pi<double> / 2.};
}

std::pair<size_t, double> distance_to_boundary(
    const naga::point_t<double, 2>& xi,
    const std::vector<edge_info_t>& edge_info
) {
    double min_distance              = std::numeric_limits<double>::max();
    double associated_incident_angle = 0;
    size_t edge_index                = 0;
    std::mutex min_distance_mutex;
#pragma omp parallel for
    for (size_t e = 0; e < edge_info.size(); ++e) {
        auto [distance, incident_angle] = distance_to_edge(xi, edge_info[e]);
        double diff_from_min
            = std::abs(std::abs(distance) - std::abs(min_distance));
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

std::vector<point_t<double, 2>> generate_2d_hexagonal_grid(
    const double& approx_point_spacing,
    const point_t<double, 2>& lower_bound,
    const point_t<double, 2>& upper_bound
) {
    std::vector<naga::point_t<double, 2>> potential_bulk_points;

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
        ))};
    potential_bulk_points.reserve(approx_grid_size[0] * approx_grid_size[1]);

    const double l = 0.5
                       * ((upper_bound[0] - lower_bound[0])
                          / (static_cast<double>(approx_grid_size[0]) - 1))
                   + 0.5
                         * ((upper_bound[1] - lower_bound[1])
                            / (static_cast<double>(approx_grid_size[1]) - 1));
    const double w = l * naga::math::sin(naga::math::pi<double> / 6.);
    const double h = l * naga::math::cos(naga::math::pi<double> / 6.);
    // first hexagonal grid
    double x_offset  = 0;
    double y         = lower_bound[1];
    bool is_even_row = true;
    while (y <= upper_bound[1]) {
        double x = is_even_row ? lower_bound[0] : lower_bound[0] - w;
        x += x_offset;
        bool is_even_column = is_even_row;
        while (x <= upper_bound[0]) {
            potential_bulk_points.emplace_back(naga::point_t<double, 2>{{x, y}}
            );
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
        double x = is_even_row ? lower_bound[0] : lower_bound[0] - w;
        x += x_offset;
        bool is_even_column = is_even_row;
        while (x <= upper_bound[0]) {
            if (!is_even_column) {
                potential_bulk_points.emplace_back(naga::point_t<double, 2>{
                    {x, y}});
            }
            x += is_even_column ? l : 2 * w + l;
            is_even_column = !is_even_column;
        }
        y += h;
        is_even_row = !is_even_row;
    }

    return potential_bulk_points;
}

std::vector<std::pair<size_t, double>> remove_points_outside_2d_contours(
    std::vector<point_t<double, 2>>& potential_bulk_points,
    const double& approx_point_spacing,
    const std::vector<edge_info_t>& boundary_edge_info
) {
    std::vector<std::pair<size_t, double>> distances_to_edges(
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
            return d.second > 0.3 * approx_point_spacing ? 1 : 0;
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

template<>
class conforming_point_cloud_impl_t<2> {
  public:
    using point_t  = naga::point_t<double, 2>;
    using normal_t = point_t;
    using index_t  = size_t;

    using closed_contour_t = conforming_point_cloud_t<2>::input_domain_data_t;

    static conforming_point_cloud_impl_t create(
        const double& approx_point_spacing,
        const std::filesystem::path& domain,
        const std::vector<std::filesystem::path>& immersed_boundaries = {}
    ) {
        std::vector<closed_contour_t> immersed_boundary_contours;
        std::vector<double> boundary_vertices;
        std::vector<double> boundary_normals;
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

        std::vector<edge_info_t> boundary_edge_info;
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

        // we also add jitter in hopes it prevents artifacting
        //        std::default_random_engine offset_rng(2); // use same seed for
        //        reproducibility std::normal_distribution<double>
        //        offset_dist(0.0, 0.05 * approx_point_spacing);
        //        std::default_random_engine angle_rng(456); // use same seed
        //        for reproducibility std::uniform_real_distribution<double>
        //        angle_dist(0., naga::math::pi<double>); for (auto& p :
        //        potential_bulk_points) {
        //            auto offset = offset_dist(offset_rng);
        //            auto angle = angle_dist(angle_rng);
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
                return naga::point_t<double, 2>{
                    {boundary_vertices[get_index() * 2 + 0],
                     boundary_vertices[get_index() * 2 + 1]}};
            }
        );
        std::transform(
            normals.begin(),
            normals.end(),
            normals.begin(),
            [&](const auto& n) {
                const auto get_index
                    = [&]() { return static_cast<size_t>(&n - &normals[0]); };
                return naga::point_t<double, 2>{
                    {boundary_normals[get_index() * 2 + 0],
                     boundary_normals[get_index() * 2 + 1]}};
            }
        );

        //        std::ofstream
        //        points_file("/home/gpu-dev/naga-result-data/points.csv");
        //        points_file << "x,y,distance_to_edge\n";
        //        for (size_t i = 0; i < potential_bulk_points.size(); ++i) {
        //            points_file << potential_bulk_points[i][0] << ","
        //                        << potential_bulk_points[i][1] << ","
        //                        << distances_to_edges[i].second << "\n";
        //        }
        //        points_file.close();

        size_t num_boundaries = normals.size();

        return {
            domain_contour,
            immersed_boundary_contours,
            points,
            normals,
            num_bulk,
            num_boundaries};
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

conforming_point_cloud_t<2> conforming_point_cloud_t<2>::create(
    const double& approximate_spacing,
    const std::filesystem::path& domain,
    const std::vector<std::filesystem::path>& immersed_boundaries
) {
    conforming_point_cloud_t point_cloud;
    auto impl = conforming_point_cloud_impl_t<2>::create(
        approximate_spacing,
        domain,
        immersed_boundaries
    );
    auto impl_ptr
        = std::make_shared<conforming_point_cloud_impl_t<2>>(std::move(impl));
    point_cloud.impl = std::move(impl_ptr);
    return point_cloud;
}

const typename conforming_point_cloud_t<2>::input_domain_data_t&
conforming_point_cloud_t<2>::domain() const {
    return impl->domain();
}

const std::vector<typename conforming_point_cloud_t<2>::input_domain_data_t>&
conforming_point_cloud_t<2>::immersed_boundaries() const {
    return impl->immersed_boundaries();
}

const std::vector<typename conforming_point_cloud_t<2>::point_t>&
conforming_point_cloud_t<2>::points() const {
    return impl->points();
}

const std::vector<typename conforming_point_cloud_t<2>::normal_t>&
conforming_point_cloud_t<2>::normals() const {
    return impl->normals();
}

const size_t& conforming_point_cloud_t<2>::num_bulk_points() const {
    return impl->num_bulk_points();
}

const size_t& conforming_point_cloud_t<2>::num_boundary_points() const {
    return impl->num_boundary_points();
}

size_t conforming_point_cloud_t<2>::size() const {
    return impl->num_boundary_points();
}

bool conforming_point_cloud_t<2>::is_boundary(
    const conforming_point_cloud_t::index_t& i
) const {
    return impl->is_boundary(i);
}

typename conforming_point_cloud_t<2>::normal_t
conforming_point_cloud_t<2>::get_normal(
    const conforming_point_cloud_t::index_t& i
) const {
    return impl->get_normal(i);
}

template class conforming_point_cloud_t<2>;

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail

#include <sdf/sdf.hpp>

namespace naga::experimental::fluids::nonlocal_lbm::detail {

using triangular_mesh_t = naga::experimental::mesh::triangular_mesh_t<double>;

struct sdf_metadata {
    std::unique_ptr<sdf::SDF> sdf;
    std::unique_ptr<sdf::Points> points;
    std::unique_ptr<sdf::Triangles> faces;
};

sdf_metadata
build_sdf(const std::vector<double>& points, const std::vector<size_t>& faces) {
    std::unique_ptr<sdf::Points> sdf_points
        = std::make_unique<sdf::Points>(points.size() / 3, 3);
    for (u_int32_t p = 0; p < points.size() / 3; p++) {
        sdf_points.get()[0](p, 0) = points[p * 3];
        sdf_points.get()[0](p, 1) = points[p * 3 + 1];
        sdf_points.get()[0](p, 2) = points[p * 3 + 2];
    }
    std::unique_ptr<sdf::Triangles> sdf_faces
        = std::make_unique<sdf::Triangles>(faces.size() / 3, 3);
    for (u_int32_t f = 0; f < faces.size() / 3; f++) {
        sdf_faces.get()[0](f, 0) = faces[f * 3];
        sdf_faces.get()[0](f, 1) = faces[f * 3 + 1];
        sdf_faces.get()[0](f, 2) = faces[f * 3 + 2];
    }
    return sdf_metadata{
        std::make_unique<sdf::SDF>(*sdf_points, *sdf_faces),
        std::move(sdf_points),
        std::move(sdf_faces)};
}

std::vector<double>
get_sdf_to_points(const sdf::Points& points, const sdf::SDF& surface) {
    auto sdfs = surface(points);
    return {sdfs.data(), sdfs.data() + sdfs.size()};
}

// returns points and normals
std::tuple<std::vector<point_t<double, 3>>, std::vector<point_t<double, 3>>>
fill_face_with_nodes(
    double nodal_spacing,
    size_t f,
    const triangular_mesh_t& boundary_mesh,
    std::vector<int> excluded_edges = {}
) {
    using vector3_t = point_t<double, 3>;
    using normal3_t = point_t<double, 3>;
    using point3_t  = point_t<double, 3>;
    using point2_t  = point_t<double, 2>;

    point3_t v1{
        {boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 0] * 3 + 0],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 0] * 3 + 1],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 0] * 3 + 2]}};
    point3_t v2{
        {boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 1] * 3 + 0],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 1] * 3 + 1],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 1] * 3 + 2]}};
    point3_t v3{
        {boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 2] * 3 + 0],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 2] * 3 + 1],
         boundary_mesh.vertices()[boundary_mesh.faces()[f * 3 + 2] * 3 + 2]}};

    vector3_t v12{{v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]}};
    vector3_t v13{{v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]}};
    normal3_t normal = naga::math::cross(v12, v13);
    naga::math::loopless::normalize<3>(normal);

    auto v1_normal
        = &boundary_mesh.normals()[3 * boundary_mesh.face_normals()[3 * f + 0]];
    if (naga::math::loopless::dot<3>(normal, v1_normal) < 0) {
        normal[0] *= -1;
        normal[1] *= -1;
        normal[2] *= -1;
    }
    auto v2_normal
        = &boundary_mesh.normals()[3 * boundary_mesh.face_normals()[3 * f + 1]];
    auto v3_normal
        = &boundary_mesh.normals()[3 * boundary_mesh.face_normals()[3 * f + 2]];

    auto x = v12;
    naga::math::loopless::normalize<3>(x);
    auto& z = normal;
    auto y  = naga::math::cross(z, x);
    naga::math::loopless::normalize<3>(y);

    std::vector<double> edge_verts2d;
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

    double area
        = 0.5
        * (-v2_2d.y() * v3_2d.x() + v1_2d.y() * (-v2_2d.x() + v3_2d.x())
           + v1_2d.x() * (v2_2d.y() - v3_2d.y()) + v2_2d.x() * v3_2d.y());

    if (std::isnan(area)) {
        return {{}, {}};
    }

    point2_t lower_bound2d{
        {std::min(edge_verts2d[0], std::min(edge_verts2d[2], edge_verts2d[4])),
         std::min(
             edge_verts2d[1],
             std::min(edge_verts2d[3], edge_verts2d[5])
         )}};

    point2_t upper_bound2d{
        {std::max(edge_verts2d[0], std::max(edge_verts2d[2], edge_verts2d[4])),
         std::max(
             edge_verts2d[1],
             std::max(edge_verts2d[3], edge_verts2d[5])
         )}};

    using namespace naga::experimental::mesh;

    auto v12_edge_normal
        = calc_v12_edge_normal_of_tri(&v1_2d[0], &v2_2d[0], &v3_2d[0]);
    auto v23_edge_normal
        = calc_v12_edge_normal_of_tri(&v2_2d[0], &v3_2d[0], &v1_2d[0]);
    auto v31_edge_normal
        = calc_v12_edge_normal_of_tri(&v3_2d[0], &v1_2d[0], &v2_2d[0]);

    std::vector<double> edge_vert_normals2d;
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

    std::vector<edge_info_t> edge_metadata;

    {
        std::vector<edge_info_t> edge1_metadata;
        subdivide_edges_and_cache_edge_info(
            nodal_spacing,
            edge1_verts2d,
            edge1_vert_normals2d,  // we dont really care about the normals here
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
        std::vector<edge_info_t> edge2_metadata;
        subdivide_edges_and_cache_edge_info(
            nodal_spacing,
            edge2_verts2d,
            edge2_vert_normals2d,  // we dont really care about the normals here
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
        std::vector<edge_info_t> edge3_metadata;
        subdivide_edges_and_cache_edge_info(
            nodal_spacing,
            edge3_verts2d,
            edge3_vert_normals2d,  // we dont really care about the normals here
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
    auto distance_to_edges = remove_points_outside_2d_contours(
        potential_face_points2d,
        nodal_spacing,
        edge_metadata
    );

    std::vector<point3_t> barycentric_coordinates;
    barycentric_coordinates.reserve(potential_face_points2d.size());
    std::transform(
        potential_face_points2d.begin(),
        potential_face_points2d.end(),
        std::back_inserter(barycentric_coordinates),
        [&](const auto& p) {
            // barycentric coordinates
            double s = 1. / (2. * area)
                     * ((v1_2d.y() * v3_2d.x() - v1_2d.x() * v3_2d.y())
                        + (v3_2d.y() - v1_2d.y()) * p.x()
                        + (v1_2d.x() - v3_2d.x()) * p.y());
            double t = 1. / (2. * area)
                     * ((v1_2d.x() * v2_2d.y() - v1_2d.y() * v2_2d.x())
                        + (v1_2d.y() - v2_2d.y()) * p.x()
                        + (v2_2d.x() - v1_2d.x()) * p.y());
            double u = 1. - s - t;
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
                 v1[2] + b[0] * v12[2] + b[1] * v13[2]}};
        }
    );
    std::vector<normal3_t> normals3d(points3d.size(), normal);

    std::vector<point3_t> edge_points3d;
    edge_points3d.reserve(edge_verts2d.size() / 2);
    for (size_t i = 6; i < edge1_verts2d.size(); i += 2) {
        if (std::find(excluded_edges.begin(), excluded_edges.end(), 0)
            != excluded_edges.end()) {
            break;
        }
        point2_t p{{edge1_verts2d[i + 0], edge1_verts2d[i + 1]}};
        auto [distance_p_to_edge2, unused1]
            = distance_to_edge(p, edge_metadata[1]);
        if (std::abs(distance_p_to_edge2) < 0.3 * nodal_spacing) {
            continue;
        }
        edge_points3d.push_back(point3_t{{p[0], p[1], 0.}});
    }
    for (size_t i = 6; i < edge2_verts2d.size(); i += 2) {
        if (std::find(excluded_edges.begin(), excluded_edges.end(), 1)
            != excluded_edges.end()) {
            break;
        }
        point2_t p{{edge2_verts2d[i + 0], edge2_verts2d[i + 1]}};
        auto [distance_p_to_edge3, unused2]
            = distance_to_edge(p, edge_metadata[2]);
        if (std::abs(distance_p_to_edge3) < 0.3 * nodal_spacing) {
            continue;
        }
        edge_points3d.push_back(point3_t{{p[0], p[1], 0.}});
    }
    for (size_t i = 6; i < edge3_verts2d.size(); i += 2) {
        if (std::find(excluded_edges.begin(), excluded_edges.end(), 2)
            != excluded_edges.end()) {
            break;
        }
        point2_t p{{edge3_verts2d[i + 0], edge3_verts2d[i + 1]}};
        auto [distance_p_to_edge1, unused3]
            = distance_to_edge(p, edge_metadata[0]);
        if (std::abs(distance_p_to_edge1) < 0.3 * nodal_spacing) {
            continue;
        }
        edge_points3d.push_back(point3_t{{p[0], p[1], 0.}});
    }

    barycentric_coordinates.clear();
    barycentric_coordinates.reserve(edge_points3d.size());
    std::transform(
        edge_points3d.begin(),
        edge_points3d.end(),
        std::back_inserter(barycentric_coordinates),
        [&](const auto& p) {
            // barycentric coordinates
            double s = 1. / (2. * area)
                     * ((v1_2d.y() * v3_2d.x() - v1_2d.x() * v3_2d.y())
                        + (v3_2d.y() - v1_2d.y()) * p.x()
                        + (v1_2d.x() - v3_2d.x()) * p.y());
            double t = 1. / (2. * area)
                     * ((v1_2d.x() * v2_2d.y() - v1_2d.y() * v2_2d.x())
                        + (v1_2d.y() - v2_2d.y()) * p.x()
                        + (v2_2d.x() - v1_2d.x()) * p.y());
            double u = 1. - s - t;
            return point3_t{{s, t, u}};
        }
    );

    std::transform(
        edge_points3d.begin(),
        edge_points3d.end(),
        edge_points3d.begin(),
        [&](const auto& p) {
            return point3_t{
                {p[0] * x[0] + p[1] * y[0] + v1[0],
                 p[0] * x[1] + p[1] * y[1] + v1[1],
                 p[0] * x[2] + p[1] * y[2] + v1[2]}};
        }
    );

    std::vector<point3_t> edge_normals3d;
    std::transform(
        barycentric_coordinates.begin(),
        barycentric_coordinates.end(),
        std::back_inserter(edge_normals3d),
        [&](const auto& b) {
            int closest_verts[2]{0, 1};
            if (b[1] < b[0]) {
                std::swap(closest_verts[0], closest_verts[1]);
            }
            if (b[2] < b[closest_verts[1]]) {
                if (b[2] < b[closest_verts[0]]) {
                    closest_verts[1] = closest_verts[0];
                    closest_verts[0] = 2;
                } else {
                    closest_verts[1] = 2;
                }
            }
            normal3_t average_normal{};
            for (int closest_vert : closest_verts) {
                if (closest_vert == 0) {
                    average_normal[0] += v1_normal[0];
                    average_normal[1] += v1_normal[1];
                    average_normal[2] += v1_normal[2];
                } else if (closest_vert == 1) {
                    average_normal[0] += v2_normal[0];
                    average_normal[1] += v2_normal[1];
                    average_normal[2] += v2_normal[2];
                } else if (closest_vert == 2) {
                    average_normal[0] += v3_normal[0];
                    average_normal[1] += v3_normal[1];
                    average_normal[2] += v3_normal[2];
                }
            }
            naga::math::loopless::normalize<3>(average_normal);
            return average_normal;
        }
    );

    points3d.insert(points3d.end(), edge_points3d.begin(), edge_points3d.end());
    normals3d
        .insert(normals3d.end(), edge_normals3d.begin(), edge_normals3d.end());

    return {points3d, normals3d};
}

void fill_surface_with_nodes(
    std::vector<point_t<double, 3>>& points,
    std::vector<point_t<double, 3>>& normals,
    double nodal_spacing,
    const triangular_mesh_t& boundary_mesh
) {
    using point_t            = point_t<double, 3>;
    unsigned int max_threads = std::thread::hardware_concurrency();

    std::vector<hashable_edge> processed_edges(boundary_mesh.faces().size());
    std::vector<std::atomic<bool>> valid_edges(boundary_mesh.faces().size());
    for (auto& valid_edge : valid_edges) {
        valid_edge = false;
    }
    std::atomic<uint> processed_edges_size{0};

    unsigned int processed_edge_cleanup_interval = max_threads * 10;
    std::vector<size_t> indices_to_erase(boundary_mesh.faces().size());
    std::atomic<uint> indices_to_erase_size{0};

    std::vector<std::vector<point_t>> points_to_insert(max_threads);
    std::vector<std::vector<point_t>> normals_to_insert(max_threads);

    std::vector<std::future<size_t>> futures;

    for (size_t f = 0; f < boundary_mesh.faces().size() / 3; f++) {
        if (futures.size() == max_threads) {
            {
                auto processed_face = futures[0].get();
                std::cout << "\rprocessed faces: " << processed_face + 1
                          << " of " << boundary_mesh.faces().size() / 3
                          << std::flush;
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

            if (f % processed_edge_cleanup_interval == 0
                && indices_to_erase_size > 100) {
                for (auto& fut : futures) {
                    auto processed_face = fut.get();
                    std::cout << "\rprocessed faces: " << processed_face + 1
                              << " of " << boundary_mesh.faces().size() / 3
                              << std::flush;

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

                std::for_each(
                    indices_to_erase.begin(),
                    indices_to_erase.begin() + indices_to_erase_size,
                    [&](const auto& i) {
                        processed_edges[i]
                            = processed_edges[--processed_edges_size];
                        valid_edges[i].store(
                            valid_edges[processed_edges_size].load()
                        );
                        valid_edges[processed_edges_size].store(false);
                    }
                );

                indices_to_erase_size = 0;
            }
        }

        std::promise<size_t> fill_promise;
        auto fill_fut = fill_promise.get_future();

        std::thread([&, f, fill_promise = std::move(fill_promise)]() mutable {
            std::vector<int> excluded_edges;

            for (int e = 0; e < 3; ++e) {
                auto edge = hashable_edge{
                    boundary_mesh.faces()[f * 3 + e],
                    boundary_mesh.faces()[f * 3 + (e + 1) % 3]};
                auto old_processed_size = processed_edges_size.fetch_add(1);
                processed_edges[old_processed_size] = edge;
                valid_edges[old_processed_size].store(true);
                auto found_ptr = processed_edges.begin() + old_processed_size;
                for (auto it = processed_edges.begin(); it != found_ptr; ++it) {
                    while (
                        !valid_edges[std::distance(processed_edges.begin(), it)]
                             .load()
                    ) {}
                    if (*it == edge) {
                        found_ptr = it;
                        break;
                    }
                }
                if (found_ptr != processed_edges.begin() + old_processed_size) {
                    excluded_edges.push_back(e);
                    auto old_erase_size = indices_to_erase_size.fetch_add(2);
                    indices_to_erase[old_erase_size]
                        = std::distance(processed_edges.begin(), found_ptr);
                    indices_to_erase[old_erase_size + 1] = old_processed_size;
                }
            }
            auto [face_points, face_normals] = fill_face_with_nodes(
                nodal_spacing,
                f,
                boundary_mesh,
                excluded_edges
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
}

template<>
class conforming_point_cloud_impl_t<3> {
  public:
    using point_t  = naga::point_t<double, 3>;
    using normal_t = point_t;
    using index_t  = size_t;

    using triangular_mesh_t = conforming_point_cloud_t<3>::input_domain_data_t;

    static conforming_point_cloud_impl_t create(
        const double& nodal_spacing,
        const std::filesystem::path& domain_obj,
        const std::vector<std::filesystem::path>& immersed_boundary_objs = {}
    ) {
        auto domain_mesh = triangular_mesh_t::import(domain_obj);
        std::vector<triangular_mesh_t> immersed_boundary_meshes;
        for (const auto& immersed_boundary_obj : immersed_boundary_objs) {
            immersed_boundary_meshes.push_back(
                triangular_mesh_t::import(immersed_boundary_obj)
            );
        }

        sdf_metadata domain_sdf
            = build_sdf(domain_mesh.vertices(), domain_mesh.faces());
        std::vector<sdf_metadata> immersed_boundary_sdfs;
        for (const auto& immersed_boundary_mesh : immersed_boundary_meshes) {
            immersed_boundary_sdfs.push_back(build_sdf(
                immersed_boundary_mesh.vertices(),
                immersed_boundary_mesh.faces()
            ));
        }

        domain_mesh.flip_normals();

        point_t lower_bound{
            {domain_mesh.vertices()[0],
             domain_mesh.vertices()[1],
             domain_mesh.vertices()[2]}};
        point_t upper_bound{
            {domain_mesh.vertices()[0],
             domain_mesh.vertices()[1],
             domain_mesh.vertices()[2]}};
        for (size_t i = 0; i < domain_mesh.vertices().size(); i += 3) {
            lower_bound[0]
                = std::min(lower_bound[0], domain_mesh.vertices()[i + 0]);
            lower_bound[1]
                = std::min(lower_bound[1], domain_mesh.vertices()[i + 1]);
            lower_bound[2]
                = std::min(lower_bound[2], domain_mesh.vertices()[i + 2]);
            upper_bound[0]
                = std::max(upper_bound[0], domain_mesh.vertices()[i + 0]);
            upper_bound[1]
                = std::max(upper_bound[1], domain_mesh.vertices()[i + 1]);
            upper_bound[2]
                = std::max(upper_bound[2], domain_mesh.vertices()[i + 2]);
        }
        size_t approx_grid_size[3]{
            static_cast<size_t>(
                std::ceil((upper_bound[0] - lower_bound[0]) / nodal_spacing)
            ),
            static_cast<size_t>(
                std::ceil((upper_bound[1] - lower_bound[1]) / nodal_spacing)
            ),
            static_cast<size_t>(
                std::ceil((upper_bound[2] - lower_bound[2]) / nodal_spacing)
            )};

        std::vector<point_t> boundary_points;
        boundary_points.reserve(
            approx_grid_size[0] * approx_grid_size[1]
            + approx_grid_size[0] * approx_grid_size[2]
            + approx_grid_size[1] * approx_grid_size[2]
        );
        std::vector<normal_t> boundary_normals;
        boundary_normals.reserve(
            approx_grid_size[0] * approx_grid_size[1]
            + approx_grid_size[0] * approx_grid_size[2]
            + approx_grid_size[1] * approx_grid_size[2]
        );
        fill_surface_with_nodes(
            boundary_points,
            boundary_normals,
            nodal_spacing,
            domain_mesh
        );

        for (const auto& immersed_boundary_mesh : immersed_boundary_meshes) {
            fill_surface_with_nodes(
                boundary_points,
                boundary_normals,
                nodal_spacing,
                immersed_boundary_mesh
            );
        }

        sdf::Points potential_bulk_points;
        {
            std::vector<point_t> fill_points(
                2 * approx_grid_size[0] * approx_grid_size[1]
                * approx_grid_size[2]
            );
            std::transform(
                fill_points.begin(),
                fill_points.begin() + fill_points.size() / 2,
                fill_points.begin(),
                [&](const point_t& p) {
                    size_t linear_id = &p - &fill_points[0];
                    size_t i         = linear_id % approx_grid_size[0];
                    size_t j         = (linear_id / approx_grid_size[0])
                             % approx_grid_size[1];
                    size_t k = (linear_id
                                / (approx_grid_size[0] * approx_grid_size[1]))
                             % (approx_grid_size[2]);
                    point_t new_point{
                        {lower_bound[0]
                             + static_cast<double>(i) * nodal_spacing,
                         lower_bound[1]
                             + static_cast<double>(j) * nodal_spacing,
                         lower_bound[2]
                             + static_cast<double>(k) * nodal_spacing}};
                    return new_point;
                }
            );
            std::transform(
                fill_points.begin() + fill_points.size() / 2,
                fill_points.end(),
                fill_points.begin() + fill_points.size() / 2,
                [&](const point_t& p) {
                    size_t linear_id
                        = &p - &fill_points[fill_points.size() / 2];
                    size_t i = linear_id % approx_grid_size[0];
                    size_t j = (linear_id / approx_grid_size[0])
                             % approx_grid_size[1];
                    size_t k = (linear_id
                                / (approx_grid_size[0] * approx_grid_size[1]))
                             % (approx_grid_size[2]);
                    point_t new_point{
                        {lower_bound[0]
                             + static_cast<double>(i) * nodal_spacing,
                         lower_bound[1]
                             + static_cast<double>(j) * nodal_spacing,
                         lower_bound[2]
                             + static_cast<double>(k) * nodal_spacing}};
                    new_point[0] += nodal_spacing / 2.0;
                    new_point[1] += nodal_spacing / 2.0;
                    new_point[2] += naga::math::sqrt(2) * nodal_spacing / 2.0;
                    return new_point;
                }
            );
            potential_bulk_points = sdf::Points(fill_points.size(), 3);
            for (uint i = 0; i < fill_points.size(); ++i) {
                potential_bulk_points(i, 0)
                    = static_cast<float>(fill_points[i][0]);
                potential_bulk_points(i, 1)
                    = static_cast<float>(fill_points[i][1]);
                potential_bulk_points(i, 2)
                    = static_cast<float>(fill_points[i][2]);
            }
        }

        std::vector<double> min_distance_to_boundary
            = get_sdf_to_points(potential_bulk_points, *domain_sdf.sdf);
        std::vector<uint> closest_boundary_indices(
            potential_bulk_points.rows(),
            0
        );

        uint boundary_index = 1;
        for (const auto& sdf : immersed_boundary_sdfs) {
            std::vector<double> distance_to_boundary
                = get_sdf_to_points(potential_bulk_points, *sdf.sdf);
            std::transform(
                distance_to_boundary.begin(),
                distance_to_boundary.end(),
                distance_to_boundary.begin(),
                [](const double& d) { return -d; }
            );
            std::transform(
                closest_boundary_indices.begin(),
                closest_boundary_indices.end(),
                closest_boundary_indices.begin(),
                [&](const uint& closest_boundary_index) {
                    size_t i = &closest_boundary_index
                             - &closest_boundary_indices[0];
                    constexpr double epsilon = 1e-6;
                    const auto& distance     = distance_to_boundary[i];
                    const auto& min_distance = min_distance_to_boundary[i];
                    if (std::abs(std::abs(distance) - std::abs(min_distance))
                        < epsilon) {
                        return distance < min_distance ? boundary_index
                                                       : closest_boundary_index;
                    } else if (std::abs(distance) < std::abs(min_distance)) {
                        return boundary_index;
                    } else {
                        return closest_boundary_index;
                    }
                }
            );
            std::transform(
                distance_to_boundary.begin(),
                distance_to_boundary.end(),
                min_distance_to_boundary.begin(),
                [&](const double& distance) {
                    size_t i = &distance - &distance_to_boundary[0];
                    return closest_boundary_indices[i] == boundary_index
                             ? distance
                             : min_distance_to_boundary[i];
                }
            );
            boundary_index++;
        }

        std::vector<point_t> bulk_points;
        bulk_points.reserve(potential_bulk_points.rows());
        std::vector<double> bulk_to_boundary_distances;
        bulk_to_boundary_distances.reserve(potential_bulk_points.rows());
        std::vector<index_t> closest_boundary_to_bulk;
        closest_boundary_to_bulk.reserve(potential_bulk_points.rows());

        for (uint i = 0; i < potential_bulk_points.rows(); i++) {
            const auto& distance_to_boundary = min_distance_to_boundary[i];
            if (distance_to_boundary < 0.1f * nodal_spacing) {
                continue;
            }

            bulk_points.emplace_back();
            bulk_points.back()[0] = potential_bulk_points(i, 0);
            bulk_points.back()[1] = potential_bulk_points(i, 1);
            bulk_points.back()[2] = potential_bulk_points(i, 2);

            bulk_to_boundary_distances.push_back(distance_to_boundary);
            closest_boundary_to_bulk.push_back(closest_boundary_indices[i]);
        }

        return {
            std::move(bulk_points),
            std::move(bulk_to_boundary_distances),
            std::move(closest_boundary_to_bulk),
            std::move(boundary_points),
            std::move(boundary_normals)};
    }

    const std::vector<point_t>& bulk_points() const { return bulk_points_; }

    const std::vector<double>& bulk_to_boundary_distances() const {
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
        std::vector<double>&& bulk_to_boundary_distances,
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
    std::vector<double> bulk_to_boundary_distances_;
    std::vector<index_t> closest_boundary_to_bulk_;
    std::vector<point_t> boundary_points_;
    std::vector<normal_t> boundary_normals_;
};

conforming_point_cloud_t<3> conforming_point_cloud_t<3>::create(
    const double& approximate_spacing,
    const std::filesystem::path& domain,
    const std::vector<std::filesystem::path>& immersed_boundaries
) {
    conforming_point_cloud_t point_cloud;
    auto impl = conforming_point_cloud_impl_t<dimensions>::create(
        approximate_spacing,
        domain,
        immersed_boundaries
    );
    auto impl_ptr = std::make_shared<conforming_point_cloud_impl_t<dimensions>>(
        std::move(impl)
    );
    point_cloud.impl = std::move(impl_ptr);
    return point_cloud;
}

const std::vector<conforming_point_cloud_t<3>::point_t>&
conforming_point_cloud_t<3>::bulk_points() const {
    return impl->bulk_points();
}

const std::vector<double>&
conforming_point_cloud_t<3>::bulk_to_boundary_distances() const {
    return impl->bulk_to_boundary_distances();
}

const std::vector<conforming_point_cloud_t<3>::index_t>&
conforming_point_cloud_t<3>::closest_boundary_to_bulk() const {
    return impl->closest_boundary_to_bulk();
}

const std::vector<conforming_point_cloud_t<3>::point_t>&
conforming_point_cloud_t<3>::boundary_points() const {
    return impl->boundary_points();
}

const std::vector<conforming_point_cloud_t<3>::normal_t>&
conforming_point_cloud_t<3>::boundary_normals() const {
    return impl->boundary_normals();
}

template class conforming_point_cloud_t<3>;

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail
