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

#include <naga/fluids/nonlocal_lattice_boltzmann/node_providers/experimental/detail/conforming_point_cloud_provider.hpp>
#include <naga/math.hpp>
#include <naga/mesh/experimental/triangular_mesh.hpp>
#include <naga/point.hpp>
#include <scalix/filesystem.hpp>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/make_mesh_3.h>
#include <sdf/sdf.hpp>

namespace naga::experimental::fluids::nonlocal_lbm::detail {

// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Mesh_polyhedron_3<K>::type Polyhedron;
typedef CGAL::Polyhedral_mesh_domain_with_features_3<K> Mesh_domain;
typedef CGAL::Parallel_tag Concurrency_tag;

// Triangulation
typedef CGAL::
    Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<
    Tr,
    Mesh_domain::Corner_index,
    Mesh_domain::Curve_index>
    C3t3;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

using namespace CGAL::parameters;

// HalfEdgeDS
typedef Polyhedron::HalfedgeDS HalfedgeDS;

using triangular_mesh_t = naga::experimental::mesh::triangular_mesh_t<double>;

triangular_mesh_t extract_surface_mesh(const C3t3& c3t3) {
    size_t number_of_surface_verts = 0;

    std::vector<double> surface_verts;
    surface_verts.reserve(c3t3.triangulation().number_of_vertices() * 3);

    std::vector<double> vertex_normals;
    surface_verts.reserve(c3t3.triangulation().number_of_vertices() * 3);

    std::vector<size_t> surface_faces;
    surface_faces.reserve(c3t3.triangulation().number_of_vertices() * 3);

    std::unordered_map<void*, size_t> vertex_map;
    vertex_map.reserve(c3t3.triangulation().number_of_vertices());

    for (const auto& c : c3t3.triangulation().finite_cell_handles()) {
        if (!c3t3.is_in_complex(c)) {
            continue;
        }
        for (int i = 0; i < 4; ++i) {
            if (!c3t3.is_in_complex(c->neighbor(i))) {
                std::vector<Tr::Vertex_handle> face_verts;
                for (int j = 0; j < 3; ++j) {
                    auto v = c->vertex((i + j + 1) % 4);
                    face_verts.push_back(v);
                }
                if ((i % 2) != 0) {
                    std::swap(face_verts[2], face_verts[1]);
                }
                double face_normal[3];
                double v12[3]{
                    face_verts[1]->point().x() - face_verts[0]->point().x(),
                    face_verts[1]->point().y() - face_verts[0]->point().y(),
                    face_verts[1]->point().z() - face_verts[0]->point().z()};
                double v13[3]{
                    face_verts[2]->point().x() - face_verts[0]->point().x(),
                    face_verts[2]->point().y() - face_verts[0]->point().y(),
                    face_verts[2]->point().z() - face_verts[0]->point().z()};
                naga::math::cross(face_normal, v12, v13);
                auto norm = naga::math::loopless::norm<3>(face_normal);
                if (norm == 0) {
                    continue;
                }
                size_t f_idx = surface_faces.size();
                for (auto v : face_verts) {
                    void* v_ptr = static_cast<void*>(&*v);
                    if (!vertex_map.count(v_ptr)) {
                        vertex_map[v_ptr] = number_of_surface_verts++;
                        surface_verts.push_back(v->point().x());
                        surface_verts.push_back(v->point().y());
                        surface_verts.push_back(v->point().z());
                        vertex_normals.push_back(0);
                        vertex_normals.push_back(0);
                        vertex_normals.push_back(0);
                    }
                    surface_faces.push_back(vertex_map[v_ptr]);
                }
                naga::math::loopless::normalize<3>(face_normal, norm);
                vertex_normals[surface_faces[f_idx + 0] * 3 + 0]
                    += face_normal[0];
                vertex_normals[surface_faces[f_idx + 0] * 3 + 1]
                    += face_normal[1];
                vertex_normals[surface_faces[f_idx + 0] * 3 + 2]
                    += face_normal[2];
                vertex_normals[surface_faces[f_idx + 1] * 3 + 0]
                    += face_normal[0];
                vertex_normals[surface_faces[f_idx + 1] * 3 + 1]
                    += face_normal[1];
                vertex_normals[surface_faces[f_idx + 1] * 3 + 2]
                    += face_normal[2];
                vertex_normals[surface_faces[f_idx + 2] * 3 + 0]
                    += face_normal[0];
                vertex_normals[surface_faces[f_idx + 2] * 3 + 1]
                    += face_normal[1];
                vertex_normals[surface_faces[f_idx + 2] * 3 + 2]
                    += face_normal[2];
            }
        }
    }

    for (size_t i = 0; i < vertex_normals.size(); i += 3) {
        auto vertex_normal = &vertex_normals[i];
        auto norm          = naga::math::loopless::norm<3>(vertex_normal);
        if (norm == 0.0) {
            continue;
        }
        naga::math::loopless::normalize<3>(vertex_normal, norm);
    }
    surface_verts.shrink_to_fit();
    surface_faces.shrink_to_fit();
    return {surface_verts, vertex_normals, surface_faces, {}};
}

// A modifier creating a triangle with the incremental builder.
template<class HDS>
class Poly_builder : public CGAL::Modifier_base<HDS> {
  public:
    Poly_builder(const std::vector<sclx::filesystem::path>& paths) {
        for (const auto& path : paths) {
            mesh_.emplace_back(
                naga::experimental::mesh::triangular_mesh_t<double>::import(path
                )
            );
        }
    }

    Poly_builder(std::vector<triangular_mesh_t> mesh)
        : mesh_(std::move(mesh)) {}

    Poly_builder(const C3t3& c3t3) {
        mesh_.push_back(extract_surface_mesh(c3t3));
    }

    void operator()(HDS& hds) {
        // Postcondition: hds is a valid polyhedral surface.
        CGAL::Polyhedron_incremental_builder_3<HDS> B(hds, true);
        size_t num_vertices = 0;
        size_t num_faces    = 0;
        for (auto& mesh : mesh_) {
            num_vertices += mesh.vertices().size() / 3;
            num_faces += mesh.faces().size() / 3;
        }
        B.begin_surface(num_vertices, num_faces);
        typedef typename HDS::Vertex Vertex;
        typedef typename Vertex::Point Point;
        for (auto& mesh : mesh_) {
            for (size_t i = 0; i < mesh.vertices().size(); i += 3) {
                B.add_vertex(Point(
                    mesh.vertices()[i + 0],
                    mesh.vertices()[i + 1],
                    mesh.vertices()[i + 2]
                ));
            }
            for (size_t i = 0; i < mesh.faces().size(); i += 3) {
                B.begin_facet();
                B.add_vertex_to_facet(mesh.faces()[i + 0]);
                B.add_vertex_to_facet(mesh.faces()[i + 1]);
                B.add_vertex_to_facet(mesh.faces()[i + 2]);
                B.end_facet();
            }
        }
    }

  private:
    std::vector<triangular_mesh_t> mesh_{};
};

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

        Polyhedron immersed_boundaries_poly;
        Poly_builder<HalfedgeDS> immersed_boundaries_poly_builder(
            immersed_boundary_meshes
        );
        immersed_boundaries_poly.delegate(immersed_boundaries_poly_builder);

        // Mesh criteria
        Mesh_criteria criteria(
            edge_size              = nodal_spacing,
            facet_angle            = 30,
            facet_size             = nodal_spacing,
            facet_distance         = nodal_spacing / 10.,
            cell_radius_edge_ratio = 3,
            cell_size              = nodal_spacing
        );

        std::unique_ptr<Mesh_domain> immersed_boundaries_domain_ptr;
        triangular_mesh_t immersed_boundaries_subdomain_mesh;
        if (!immersed_boundary_objs.empty()) {
            immersed_boundaries_domain_ptr = std::move(
                std::make_unique<Mesh_domain>(immersed_boundaries_poly)
            );
            Mesh_domain& immersed_boundaries_domain
                = *immersed_boundaries_domain_ptr;
            immersed_boundaries_domain.detect_features();

            C3t3 immersed_c3t3 = CGAL::make_mesh_3<C3t3>(
                immersed_boundaries_domain,
                criteria,
                no_perturb(),
                no_exude()
            );
//            CGAL::refine_mesh_3(
//                immersed_c3t3,
//                immersed_boundaries_domain,
//                criteria
//            );

            immersed_boundaries_subdomain_mesh
                = extract_surface_mesh(immersed_c3t3);
        }

        Polyhedron immersed_boundaries_subdomain_poly;
        Poly_builder<HalfedgeDS> immersed_boundaries_subdomain_poly_builder(
            {immersed_boundaries_subdomain_mesh}
        );
        immersed_boundaries_subdomain_poly.delegate(
            immersed_boundaries_subdomain_poly_builder
        );

        Polyhedron domain_poly;
        Poly_builder<HalfedgeDS> domain_poly_builder({domain_mesh});
        domain_poly.delegate(domain_poly_builder);

        std::unique_ptr<Mesh_domain> domain_ptr;
        if (!immersed_boundary_objs.empty()) {
            domain_ptr = std::move(std::make_unique<Mesh_domain>(
                immersed_boundaries_subdomain_poly,
                domain_poly
            ));
        } else {
            domain_ptr = std::move(std::make_unique<Mesh_domain>(domain_poly));
        }
        Mesh_domain& domain = *domain_ptr;
        C3t3 c3t3           = CGAL::make_mesh_3<C3t3>(
            domain,
            criteria,
            no_perturb(),
            no_exude()
        );
//        CGAL::refine_mesh_3(c3t3, domain, criteria);

        triangular_mesh_t domain_boundary_mesh = extract_surface_mesh(c3t3);

        sdf::Points potential_bulk_points;
        {
            std::unordered_map<void*, index_t> vertex_to_index;
            size_t vertex_count = 0;
            std::vector<double> potential_bulk_points_flat;
            potential_bulk_points_flat.reserve(
                c3t3.triangulation().number_of_vertices() * 3
            );
            for (const auto& c : c3t3.triangulation().finite_cell_handles()) {
                if (!c3t3.is_in_complex(c)) {
                    continue;
                }
                for (int i = 0; i < 4; ++i) {
                    auto v = c->vertex(i);
                    void* v_ptr = static_cast<void*>(&*v);
                    if (vertex_to_index.count(v_ptr)) {
                        continue;
                    }
                    if (!immersed_boundary_objs.empty()
                        && immersed_boundaries_domain_ptr
                               ->is_in_domain_object()(
                                   {v->point().x(),
                                    v->point().y(),
                                    v->point().z()}
                               )) {
                        continue;
                    }
                    potential_bulk_points_flat.push_back(v->point().x());
                    potential_bulk_points_flat.push_back(v->point().y());
                    potential_bulk_points_flat.push_back(v->point().z());
                    vertex_to_index[v_ptr] = vertex_count++;
                }
            }

            potential_bulk_points
                = sdf::Points(potential_bulk_points_flat.size() / 3, 3);
            for (u_int32_t p = 0; p < potential_bulk_points_flat.size() / 3;
                 p++) {
                potential_bulk_points(p, 0)
                    = static_cast<float>(potential_bulk_points_flat[p * 3 + 0]);
                potential_bulk_points(p, 1)
                    = static_cast<float>(potential_bulk_points_flat[p * 3 + 1]);
                potential_bulk_points(p, 2)
                    = static_cast<float>(potential_bulk_points_flat[p * 3 + 2]);
            }
        }

        std::vector<double> min_distance_to_boundary
            = get_sdf_to_points(potential_bulk_points, *domain_sdf.sdf);
        std::transform(
            min_distance_to_boundary.begin(),
            min_distance_to_boundary.end(),
            min_distance_to_boundary.begin(),
            [](const double& d) { return std::abs(d); }
        );
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
                [](const double& d) { return std::abs(d); }
            );
            std::transform(
                closest_boundary_indices.begin(),
                closest_boundary_indices.end(),
                closest_boundary_indices.begin(),
                [&](const uint& closest_boundary_index) {
                    size_t i = &closest_boundary_index
                             - &closest_boundary_indices[0];
                    if (distance_to_boundary[i] < min_distance_to_boundary[i]) {
                        return boundary_index;
                    }
                    return closest_boundary_index;
                }
            );
            std::transform(
                min_distance_to_boundary.begin(),
                min_distance_to_boundary.end(),
                min_distance_to_boundary.begin(),
                [&](const double& distance) {
                    size_t i = &distance - &min_distance_to_boundary[0];
                    return std::min(distance, distance_to_boundary[i]);
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

        size_t approx_num_boundary_points
            = domain_boundary_mesh.vertices().size() + immersed_boundaries_subdomain_mesh.vertices().size();
        std::vector<point_t> boundary_points;
        boundary_points.reserve(approx_num_boundary_points);
        std::vector<normal_t> boundary_normals;
        boundary_normals.reserve(approx_num_boundary_points);

        domain_boundary_mesh.flip_normals();
        for (size_t i = 0; i < domain_boundary_mesh.vertices().size(); i+=3) {
            boundary_points.emplace_back();
            boundary_points.back()[0] = domain_boundary_mesh.vertices()[i + 0];
            boundary_points.back()[1] = domain_boundary_mesh.vertices()[i + 1];
            boundary_points.back()[2] = domain_boundary_mesh.vertices()[i + 2];

            boundary_normals.emplace_back();
            boundary_normals.back()[0] = domain_boundary_mesh.normals()[i + 0];
            boundary_normals.back()[1] = domain_boundary_mesh.normals()[i + 1];
            boundary_normals.back()[2] = domain_boundary_mesh.normals()[i + 2];
        }

        for (size_t i = 0; i < immersed_boundaries_subdomain_mesh.vertices().size(); i+=3) {
            boundary_points.emplace_back();
            boundary_points.back()[0] = immersed_boundaries_subdomain_mesh.vertices()[i + 0];
            boundary_points.back()[1] = immersed_boundaries_subdomain_mesh.vertices()[i + 1];
            boundary_points.back()[2] = immersed_boundaries_subdomain_mesh.vertices()[i + 2];

            boundary_normals.emplace_back();
            boundary_normals.back()[0] = immersed_boundaries_subdomain_mesh.normals()[i + 0];
            boundary_normals.back()[1] = immersed_boundaries_subdomain_mesh.normals()[i + 1];
            boundary_normals.back()[2] = immersed_boundaries_subdomain_mesh.normals()[i + 2];
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
