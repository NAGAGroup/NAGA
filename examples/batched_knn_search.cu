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
#include <chrono>
#include <naga/particle_segmentation/nearest_neighbors.cuh>

template<class T, uint Dimensions>
class grid_point_generator {
  public:
    __host__ grid_point_generator(
        sclx::array<T, 2> grid,
        T spacing,
        size_t grid_size,
        T offset = 0
    )
        : grid(grid),
          spacing(spacing),
          grid_size(grid_size),
          offset(offset) {}

    template<class KernelInfo = void>
    __host__ __device__ void
    operator()(const sclx::md_index_t<1>& idx, const KernelInfo&) const {
        sclx::index_t linear_index = idx[0];
        for (uint i = 0; i < Dimensions; ++i) {
            grid(i, idx[0]) = (linear_index % grid_size) * spacing + offset;
            linear_index /= grid_size;
        }
    }

  private:
    sclx::array<T, 2> grid;
    T spacing;
    size_t grid_size;
    T offset;
};

int main() {
    size_t big_grid_size     = 200;
    size_t small_grid_size   = 4;
    float big_grid_spacing   = 1.0f;
    float small_grid_spacing = 1.0f / small_grid_size;
    constexpr uint dims      = 3;
    sclx::array<float, 2> grid{
        dims,
        static_cast<size_t>(
            std::pow(big_grid_size, dims) + std::pow(small_grid_size, dims)
        )};

    uint k = static_cast<uint>(std::pow(small_grid_size, dims));

    // fill large grid
    auto big_grid_slice = grid.get_range(
        {0},
        {static_cast<size_t>(std::pow(big_grid_size, dims))}
    );
    sclx::execute_kernel([=](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<1>{
                static_cast<size_t>(std::pow(big_grid_size, dims))},
            big_grid_slice,
            grid_point_generator<float, dims>{
                big_grid_slice,
                big_grid_spacing,
                big_grid_size}
        );
    });

    // fill small grid such that all nearest neighbors to the first point in the
    // large grid are in the small grid
    auto small_grid_slice = grid.get_range(
        {static_cast<size_t>(std::pow(big_grid_size, dims))},
        {static_cast<size_t>(
            std::pow(big_grid_size, dims) + std::pow(small_grid_size, dims)
        )}
    );
    sclx::execute_kernel([=](sclx::kernel_handler& handler) {
        handler.launch(
            sclx::md_range_t<1>{small_grid_size * small_grid_size},
            small_grid_slice,
            grid_point_generator<float, dims>{
                small_grid_slice,
                small_grid_spacing,
                small_grid_size,
                -0.5f * small_grid_spacing * small_grid_size}
        );
    });

    uint partitioner_sizes[]
        = {static_cast<uint>(k / 2.f), k, k * 2, k * 4, k * 8, k * 16};

    size_t big_grid_point_count = std::pow(big_grid_size, dims);
    std::cout << "Number of points in big grid: " << big_grid_point_count
              << std::endl
              << std::endl;

    std::cout << "We expect all nearest points to the first point to be in "
                 "the small grid, except for the first point itself."
              << std::endl;

    for (auto& part_size : partitioner_sizes) {
        std::cout << "Partitioner size: " << part_size << std::endl;
        using point_map_t   = naga::default_point_map<float, dims>;
        using partitioner_t = naga::knn::rectangular_partitioner<point_map_t>;

        auto start = std::chrono::high_resolution_clock::now();

        // build partitioner
        partitioner_t partitioner(grid, part_size);
        auto end_part_build = std::chrono::high_resolution_clock::now();

        // find nearest neighbors
        auto [distances_squared, indices] = naga::batched_nearest_neighbors(
            k,
            point_map_t{big_grid_slice},
            partitioner
        );
        auto end_knn = std::chrono::high_resolution_clock::now();

        auto first_nn = indices.get_slice(sclx::md_index_t<1>{0});
        for (auto& i : first_nn) {
            if (i != 0 && i < big_grid_point_count) {
                std::cerr
                    << "The above expectation was violated, indicating a bug."
                    << std::endl;
                std::cerr << "Offending index: " << i << std::endl;
            }
        }

        std::chrono::duration<double> part_build_time = end_part_build - start;
        std::chrono::duration<double> knn_time   = end_knn - end_part_build;
        std::chrono::duration<double> total_time = end_knn - start;

        std::cout << "Partitioner build time: "
                  << part_build_time.count() * 1000 << " ms" << std::endl;
        std::cout << "KNN search time: " << knn_time.count() * 1000 << " ms"
                  << std::endl;
        std::cout << "Total time: " << total_time.count() * 1000 << " ms"
                  << std::endl
                  << std::endl
                  << std::endl;
    }

    return 0;
}
