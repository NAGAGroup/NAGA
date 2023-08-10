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

namespace naga::experimental::fluids::nonlocal_lbm::detail {

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_mesh_vertex_base_2<K> Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> Tds;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, Tds> CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT> Criteria;

typedef CDT::Vertex_handle Vertex_handle;
typedef CDT::Point Point;

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
        CDT cdt;

        std::list<Point> list_of_seeds;

        std::vector<closed_contour_t> immersed_boundary_contours;
        for (auto& im_obj_file : immersed_boundaries) {
            CDT seed_point_cdt;
            immersed_boundary_contours.emplace_back(
                closed_contour_t::import(im_obj_file)
            );
            const auto& contour = immersed_boundary_contours.back();
            for (size_t e = 0; e < contour.edges().size(); e += 2) {
                const double* point_raw1
                    = &contour.vertices()[2 * contour.edges()[e]];
                const double* point_raw2
                    = &contour.vertices()[2 * contour.edges()[e + 1]];

                Vertex_handle v1
                    = cdt.insert(Point(point_raw1[0], point_raw1[1]));
                Vertex_handle v2
                    = cdt.insert(Point(point_raw2[0], point_raw2[1]));
                cdt.insert_constraint(v1, v2);

                Vertex_handle v3
                    = seed_point_cdt.insert(Point(point_raw1[0], point_raw1[1])
                    );
                Vertex_handle v4
                    = seed_point_cdt.insert(Point(point_raw2[0], point_raw2[1])
                    );
                seed_point_cdt.insert_constraint(v3, v4);
            }

            CGAL::refine_Delaunay_mesh_2(seed_point_cdt, Criteria());

            for (const auto& v : seed_point_cdt.finite_vertex_handles()) {
                auto edge_circulator       = seed_point_cdt.incident_edges(v);
                auto edge_circulator_begin = edge_circulator;
                bool is_boundary           = false;

                do {
                    const auto& e = *edge_circulator;
                    if (seed_point_cdt.is_infinite(e.first->vertex(e.second))) {
                        is_boundary = true;
                        break;
                    }
                } while (++edge_circulator != edge_circulator_begin);

                if (!is_boundary) {
                    list_of_seeds.push_back(v->point());
                    break;
                }
            }
        }

        closed_contour_t domain_contour
            = closed_contour_t::import(domain);
        std::vector<point_t> points;
        std::vector<normal_t> normals;
        uint num_boundaries;
        uint num_bulk;

        {
            std::vector<bool> boundary_flags;

            const auto& contour = domain_contour;
            for (size_t e = 0; e < contour.edges().size(); e += 2) {
                const double* point_raw1
                    = &contour.vertices()[2 * contour.edges()[e]];
                Vertex_handle v1
                    = cdt.insert(Point(point_raw1[0], point_raw1[1]));
                const double* point_raw2
                    = &contour.vertices()[2 * contour.edges()[e + 1]];
                Vertex_handle v2
                    = cdt.insert(Point(point_raw2[0], point_raw2[1]));
                cdt.insert_constraint(v1, v2);
            }

            CGAL::refine_Delaunay_mesh_2(
                cdt,
                list_of_seeds.begin(),
                list_of_seeds.end(),
                Criteria(0.125, approx_point_spacing)
            );

            CGAL::lloyd_optimize_mesh_2(
                cdt,
                CGAL::parameters::time_limit           = 30,
                CGAL::parameters::max_iteration_number = 10,
                CGAL::parameters::seeds_begin          = list_of_seeds.begin(),
                CGAL::parameters::seeds_end            = list_of_seeds.end()
            );

            for (const auto& v : cdt.finite_vertex_handles()) {
                auto edge_circulator       = cdt.incident_edges(v);
                auto edge_circulator_begin = edge_circulator;
                std::vector<CDT::Edge> boundary_edges(2);
                bool is_boundary = false;

                do {
                    const auto& e        = *edge_circulator;
                    const auto& mirror_e = cdt.mirror_edge(e);

                    if (!cdt.is_constrained(e)) {
                        continue;
                    }

                    if (cdt.is_infinite(e.first->vertex(e.second))
                        || !e.first->is_in_domain()) {
                        is_boundary       = true;
                        boundary_edges[0] = e;
                    }
                    if (cdt.is_infinite(mirror_e.first->vertex(mirror_e.second))
                        || !mirror_e.first->is_in_domain()) {
                        is_boundary       = true;
                        boundary_edges[1] = mirror_e;
                    }
                } while (++edge_circulator != edge_circulator_begin);

                points.push_back(point_t({v->point().x(), v->point().y()}));

                if (!is_boundary) {
                    normals.emplace_back(normal_t({0., 0.}));
                    boundary_flags.push_back(false);
                    continue;
                }

                std::vector<double> normal1(2);
                normal1[0]
                    = boundary_edges[0]
                          .first->vertex((boundary_edges[0].second + 1) % 3)
                          ->point()
                          .y()
                    - boundary_edges[0]
                          .first->vertex((boundary_edges[0].second + 2) % 3)
                          ->point()
                          .y();
                normal1[1]
                    = boundary_edges[0]
                          .first->vertex((boundary_edges[0].second + 2) % 3)
                          ->point()
                          .x()
                    - boundary_edges[0]
                          .first->vertex((boundary_edges[0].second + 1) % 3)
                          ->point()
                          .x();
                double norm1 = std::sqrt(
                    normal1[0] * normal1[0] + normal1[1] * normal1[1]
                );
                normal1[0] /= norm1;
                normal1[1] /= norm1;

                std::vector<double> normal2(2);
                normal2[0]
                    = boundary_edges[1]
                          .first->vertex((boundary_edges[1].second + 1) % 3)
                          ->point()
                          .y()
                    - boundary_edges[1]
                          .first->vertex((boundary_edges[1].second + 2) % 3)
                          ->point()
                          .y();
                normal2[1]
                    = boundary_edges[1]
                          .first->vertex((boundary_edges[1].second + 2) % 3)
                          ->point()
                          .x()
                    - boundary_edges[1]
                          .first->vertex((boundary_edges[1].second + 1) % 3)
                          ->point()
                          .x();
                double norm2 = std::sqrt(
                    normal2[0] * normal2[0] + normal2[1] * normal2[1]
                );
                normal2[0] /= norm2;
                normal2[1] /= norm2;

                std::vector<double> normal(2);
                normal[0] = normal1[0] + normal2[0];
                normal[1] = normal1[1] + normal2[1];
                double norm
                    = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1]);

                // invert the normal to be compatible with what LBM simulation
                // expects
                normal[0] /= -norm;
                normal[1] /= -norm;
                normals.push_back(normal_t({normal[0], normal[1]}));

                boundary_flags.push_back(true);
            }

            num_boundaries = std::count(
                boundary_flags.begin(),
                boundary_flags.end(),
                true
            );
            num_bulk = boundary_flags.size() - num_boundaries;
            std::stable_partition(
                points.begin(),
                points.end(),
                [&](const auto& v) { return !boundary_flags[&v - &points[0]]; }
            );
            std::stable_partition(
                normals.begin(),
                normals.end(),
                [&](const auto& v) { return !boundary_flags[&v - &normals[0]]; }
            );
            normals.erase(normals.begin(), normals.begin() + num_bulk);
        }

        return {
            domain_contour,
            immersed_boundary_contours,
            points,
            normals,
            num_bulk,
            num_boundaries};
    }

    const closed_contour_t& domain() const { return domain_; }

    const std::vector<closed_contour_t>& immersed_boundaries() const {
        return immersed_boundaries_;
    }

    const std::vector<point_t>& points() const { return points_; }

    const std::vector<normal_t>& normals() const { return normals_; }

    const size_t& num_bulk_points() const { return num_bulk_points_; }

    const size_t& num_boundary_points() const { return num_boundary_points_; }

    size_t size() const { return points_.size(); }

    bool is_boundary(const index_t& i) { return i >= num_bulk_points_; }

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

template class conforming_point_cloud_t<2>;

}  // namespace naga::experimental::fluids::nonlocal_lbm::detail
