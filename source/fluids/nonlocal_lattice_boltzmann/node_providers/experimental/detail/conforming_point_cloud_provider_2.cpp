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

#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/lloyd_optimize_mesh_2.h>
#include <naga/fluids/nonlocal_lattice_boltzmann/node_providers/experimental/detail/conforming_point_cloud_provider.hpp>
#include <utility>

namespace naga::experimental::fluids::nonlocal_lbm::detail {

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_mesh_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;

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
        double actual_length = edge_info.length / (num_nodes_to_add + 1);

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

double distance_to_edge(
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
        auto signed_distance_to_v1
            = sgn(naga::math::loopless::dot<2>(v1_to_xi, edge_info.edge_normal))
            * naga::math::loopless::norm<2>(v1_to_xi);
        return signed_distance_to_v1;
    }
    if (signed_v1_intersection_distance > edge_info.length) {
        vector_t v2_to_xi{{xi[0] - edge_info.v2[0], xi[1] - edge_info.v2[1]}};
        auto signed_distance_to_v2
            = sgn(naga::math::loopless::dot<2>(v2_to_xi, edge_info.edge_normal))
            * naga::math::loopless::norm<2>(v2_to_xi);
        return signed_distance_to_v2;
    }
    return signed_distance_to_line;
}

std::pair<size_t, double> distance_to_boundary(
    const naga::point_t<double, 2>& xi,
    const std::vector<edge_info_t>& edge_info
) {
    double min_distance = std::numeric_limits<double>::max();
    size_t edge_index   = 0;
    std::mutex min_distance_mutex;
#pragma omp parallel for
    for (size_t e = 0; e < edge_info.size(); ++e) {
        double distance = distance_to_edge(xi, edge_info[e]);
        std::lock_guard<std::mutex> lock(min_distance_mutex);
        if (std::abs(distance) < std::abs(min_distance)) {
            min_distance = distance;
            edge_index   = e;
        }
    }
    return {edge_index, min_distance};
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

        const double l = approx_point_spacing;
        const double w = l * naga::math::sin(naga::math::pi<double> / 6.);
        const double h = l * naga::math::cos(naga::math::pi<double> / 6.);

        auto domain_lower_bound = domain_contour.lower_bound();
        auto domain_upper_bound = domain_contour.upper_bound();
        size_t approx_grid_size[2]{
            static_cast<size_t>(std::ceil(
                (domain_upper_bound[0] - domain_lower_bound[0])
                    / approx_point_spacing
                + 1
            )),
            static_cast<size_t>(std::ceil(
                (domain_upper_bound[1] - domain_lower_bound[1])
                    / approx_point_spacing
                + 1
            ))};

        std::vector<naga::point_t<double, 2>> potential_bulk_points;
        potential_bulk_points.reserve(
            approx_grid_size[0] * approx_grid_size[1]
        );
        // first hexagonal grid
        double x_offset  = 0;
        double y         = domain_lower_bound[1];
        bool is_even_row = true;
        while (y <= domain_upper_bound[1]) {
            double x = is_even_row ? domain_lower_bound[0]
                                   : domain_lower_bound[0] - w;
            x += x_offset;
            bool is_even_column = is_even_row;
            while (x <= domain_upper_bound[0]) {
                potential_bulk_points.emplace_back(naga::point_t<double, 2>{
                    {x, y}});
                x += is_even_column ? l : 2 * w + l;
                is_even_column = !is_even_column;
            }
            y += h;
            is_even_row = !is_even_row;
        }
        // second hexagonal grid
        x_offset    = l;
        y           = domain_lower_bound[1];
        is_even_row = true;
        while (y <= domain_upper_bound[1]) {
            double x = is_even_row ? domain_lower_bound[0]
                                   : domain_lower_bound[0] - w;
            x += x_offset;
            bool is_even_column = is_even_row;
            while (x <= domain_upper_bound[0]) {
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

        std::vector<point_t>& points = potential_bulk_points;
        points.resize(num_bulk + boundary_vertices.size() / 2);
        std::vector<normal_t> normals(boundary_vertices.size() / 2);
        std::transform(
            points.begin() + num_bulk,
            points.end(),
            points.begin() + num_bulk,
            [&](const auto& p) {
                const auto get_index
                    = [&]() { return static_cast<size_t>(&p - &points[num_bulk]); };
                return naga::point_t<double,2>{{boundary_vertices[get_index() * 2 + 0],
                         boundary_vertices[get_index() * 2 + 1]}};
            }
        );
        std::transform(
            normals.begin(),
            normals.end(),
            normals.begin(),
            [&](const auto& n) {
                const auto get_index = [&]() {
                    return static_cast<size_t>(&n - &normals[0]);
                };
                return naga::point_t<double,2>{{boundary_normals[get_index() * 2 + 0],
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
