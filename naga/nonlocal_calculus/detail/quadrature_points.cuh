
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

#include "../../math.cuh"
#include "../../point_map.cuh"

namespace naga::nonlocal_calculus::detail {

constexpr uint num_radial_quad_points = 4;
constexpr uint num_theta_quad_points  = 4;
constexpr uint num_phi_quad_points    = 4;
constexpr uint num_interp_support     = 32;

constexpr uint num_quad_points_2d
    = num_radial_quad_points * num_theta_quad_points;
constexpr uint num_quad_points_3d = num_quad_points_2d * num_phi_quad_points;

template<class T>
__constant__ static T const_radial_quad_weights[num_radial_quad_points]{
    0.652145,
    0.652145,
    0.347855,
    0.347855};

template<class T>
__constant__ static T const_radial_quad_points[num_radial_quad_points]{
    0.339981,
    -0.339981,
    0.861136,
    -0.861136,
};

template<class T>
__constant__ static T unscaled_quad_points_2d[2 * num_quad_points_2d]{};
template<class T>
static bool is_quad_points_2d_init = false;

template<class T>
__constant__ static T unscaled_quad_points_3d[3 * num_quad_points_3d]{};
template<class T>
static bool is_quad_points_3d_init = false;

template<class T>
__host__ void calc_unscaled_quad_point_2d(size_t q_idx, T* x_k) {

    uint r_idx = q_idx % num_radial_quad_points;
    uint t_idx = q_idx / num_radial_quad_points;

    T r                     = (const_radial_quad_points<T>[r_idx] + 1.f) / 2.f;
//    constexpr T theta_scale = 2 * math::pi<T> / (num_theta_quad_points - 1);
    T theta = (const_radial_quad_points<T>[t_idx] + 1.f) * 2.f * math::pi<T> / 2.f;

    x_k[0] = r * math::cos(theta);
    x_k[1] = r * math::sin(theta);
}

template<class T>
__host__ void init_quad_points_2d() {
    if (is_quad_points_2d_init<T>) {
        return;
    }

    T quad_points[2 * num_quad_points_2d];

    for (uint q_idx = 0; q_idx < num_quad_points_2d; ++q_idx) {
        calc_unscaled_quad_point_2d(q_idx, quad_points + 2 * q_idx);
    }

    int num_devices       = sclx::cuda::traits::device_count();
    int current_device    = sclx::cuda::traits::current_device();
    auto current_location = std::experimental::source_location::current();
    for (int dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        sclx::cuda::set_device(dev_idx);
        auto err = cudaMemcpyToSymbol(
            unscaled_quad_points_2d<T>,
            quad_points,
            2 * num_quad_points_2d * sizeof(T),
            0,
            cudaMemcpyHostToDevice
        );
        sclx::cuda::cuda_exception::raise_if_not_success(
            err,
            current_location,
            "naga::nonlocal_calculus::detail::"
        );
    }
    sclx::cuda::set_device(current_device);
    is_quad_points_2d_init<T> = true;
}

template<class T>
__host__ void calc_unscaled_quad_point_3d(size_t q_idx, T x_k[3]) {

    uint r_idx   = q_idx % num_radial_quad_points;
    uint t_idx   = (q_idx / num_radial_quad_points) % num_theta_quad_points;
    uint phi_idx = q_idx / num_quad_points_2d;

    T r = (const_radial_quad_points<T>[r_idx] + 1.f) / 2.f;

//    constexpr T theta_scale = 2 * math::pi<T> / (num_theta_quad_points - 1);
    T theta = (const_radial_quad_points<T>[t_idx] + 1.f) * 2.f * math::pi<T> / 2.f;

//    constexpr T phi_scale = math::pi<T> / (num_phi_quad_points - 1);
    T phi = (const_radial_quad_points<T>[phi_idx] + 1.f) * math::pi<T> / 2.f;

    x_k[0] = r * math::sin(phi) * math::cos(theta);
    x_k[1] = r * math::sin(phi) * math::sin(theta);
    x_k[2] = r * math::cos(phi);
}

template<class T>
__host__ void init_quad_points_3d() {
    if (is_quad_points_3d_init<T>) {
        return;
    }

    T quad_points[3 * num_quad_points_3d];

    for (uint q_idx = 0; q_idx < num_quad_points_3d; ++q_idx) {
        calc_unscaled_quad_point_3d(q_idx, quad_points + 3 * q_idx);
    }

    int num_devices       = sclx::cuda::traits::device_count();
    int current_device    = sclx::cuda::traits::current_device();
    auto current_location = std::experimental::source_location::current();
    for (int dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        sclx::cuda::set_device(dev_idx);
        auto err = cudaMemcpyToSymbol(
            unscaled_quad_points_3d<T>,
            quad_points,
            3 * num_quad_points_3d * sizeof(T),
            0,
            cudaMemcpyHostToDevice
        );
        sclx::cuda::cuda_exception::raise_if_not_success(
            err,
            current_location,
            "naga::nonlocal_calculus::detail::"
        );
    }
    sclx::cuda::set_device(current_device);
    is_quad_points_3d_init<T> = true;
}

template<class T, uint Dimensions>
class quadrature_point_map {
  public:
    using point_type = point_t<T, Dimensions>;

    __host__ quadrature_point_map(
        const sclx::array<const T, 2>& query_points,
        const sclx::array<const T, 1>& interaction_radii
    )
        : query_points_(query_points),
          interaction_radii_(interaction_radii) {
        static_assert(
            Dimensions == 2 || Dimensions == 3,
            "quadrature_point_map only supports 2D and 3D"
        );

        if (query_points.shape()[0] != Dimensions) {
            sclx::throw_exception<std::invalid_argument>(
                "query_points must have shape (Dimensions, num_query_points)",
                "naga::nonlocal_calculus::quadrature_point_map::"
            );
        }

        if constexpr (Dimensions == 2) {
            init_quad_points_2d<T>();
            if (!is_host_unscaled_quad_points_init_) {
                cudaMemcpyFromSymbol(
                    host_unscaled_quad_points_,
                    unscaled_quad_points_2d<T>,
                    2 * num_quad_points_2d * sizeof(T),
                    0,
                    cudaMemcpyDeviceToHost
                );
                is_host_unscaled_quad_points_init_ = true;
            }
        } else {
            init_quad_points_3d<T>();
            if (!is_host_unscaled_quad_points_init_) {
                cudaMemcpyFromSymbol(
                    host_unscaled_quad_points_,
                    unscaled_quad_points_3d<T>,
                    3 * num_quad_points_3d * sizeof(T),
                    0,
                    cudaMemcpyDeviceToHost
                );
                is_host_unscaled_quad_points_init_ = true;
            }
        }
    }

    __host__ __device__ point_type operator[](const sclx::index_t& index
    ) const {
        const T* unscaled_quad_points;
#ifdef __CUDA_ARCH__
        if constexpr (Dimensions == 2) {
            unscaled_quad_points = unscaled_quad_points_2d<T>;
        } else {
            unscaled_quad_points = unscaled_quad_points_3d<T>;
        }
#else
        unscaled_quad_points = host_unscaled_quad_points_;
#endif
        point_t<T, Dimensions> point;
        T interaction_radius = interaction_radii_[index / num_quad_points_];
        for (uint d = 0; d < Dimensions; ++d) {
            point[d] = query_points_(d, index / num_quad_points_);
            point[d] += interaction_radius
                      * unscaled_quad_points
                            [Dimensions * (index % num_quad_points_) + d];
        }

        return point;
    }

    __host__ __device__ point_type operator[](const sclx::md_index_t<1>& index
    ) const {
        return (*this)[index[0]];
    }

    __host__ __device__ size_t size() const {
        return query_points_.shape()[1] * num_quad_points_;
    }

  private:
    sclx::array<const T, 2> query_points_;
    sclx::array<const T, 1> interaction_radii_;

    static constexpr int num_quad_points_
        = Dimensions == 2 ? num_quad_points_2d : num_quad_points_3d;
    static T host_unscaled_quad_points_[Dimensions * num_quad_points_];
    static bool is_host_unscaled_quad_points_init_;
};

template<class T, uint Dimensions>
T quadrature_point_map<T, Dimensions>::host_unscaled_quad_points_
    [Dimensions * num_quad_points_];
template<class T, uint Dimensions>
bool quadrature_point_map<T, Dimensions>::is_host_unscaled_quad_points_init_
    = false;

template<class T>
void get_closest_neighbor_distances(
    const sclx::array<T, 2>& knn_distances_squared,
    const sclx::array<T, 1>& closest_neighbor_distances
) {
    sclx::execute_kernel([&](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<1>(closest_neighbor_distances.shape()),
            closest_neighbor_distances,
            [=] __device__(const sclx::md_index_t<1>& idx, const auto&) {
                closest_neighbor_distances[idx]
                    = math::sqrt(knn_distances_squared(1, idx[0]));
            }
        );
    });
}

}  // namespace naga::nonlocal_calculus::detail
