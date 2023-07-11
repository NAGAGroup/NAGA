
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

#include "../../../point.cuh"
#include "../lattices.cuh"
#include "subtask_factory.h"
#include <scalix/execute_kernel.cuh>

namespace naga::fluids::nonlocal_lbm::detail {

template<class T, uint Dimensions>
class pml_div_Q1_field_map {
  public:
    using point_type = point_t<T, Dimensions>;

    __host__ __device__ pml_div_Q1_field_map(
        const T* c0,
        const sclx::array<const T, 1>& absorption_coeff,
        const sclx::array<const T, 1>& Q1,
        size_t pml_start_index,
        size_t pml_end_index
    )
        : absorption_coeff(absorption_coeff),
          Q1(Q1),
          pml_start_index(pml_start_index),
          pml_end_index(pml_end_index) {

        for (uint d = 0; d < Dimensions; d++) {
            this->c0[d] = c0[d];
        }
    }

    __host__ __device__ point_type operator[](sclx::index_t i) const {
        point_type c;
        if (i < pml_start_index || i >= pml_end_index) {
            for (uint d = 0; d < Dimensions; d++) {
                c[d] = 0.f;
            }
        } else {
            size_t pml_index = i - pml_start_index;
            T coeff          = absorption_coeff[pml_index];
            for (uint d = 0; d < Dimensions; d++) {
                c[d] = -c0[d] * coeff * Q1[pml_index];
            }
        }

        return c;
    }

    __host__ __device__ point_type operator[](const sclx::md_index_t<1>& index
    ) const {
        return (*this)[index[0]];
    }

    __host__ __device__ size_t size() const {
        return absorption_coeff.elements();
    }

  private:
    T c0[Dimensions];
    sclx::array<const T, 1> absorption_coeff;
    sclx::array<const T, 1> Q1;
    size_t pml_start_index;
    size_t pml_end_index;
};

/**
 * @brief Adds the PML absorption term to distribution, sans divergence terms
 */
template<class Lattice>
class partial_pml_2d_subtask;

// Clion can't find implementation of the static `create` method in
// the following class. This is a bug in Clion.
//
// You can find the implementation in the .inl variant of this file.

template<class Lattice>
struct subtask_factory<partial_pml_2d_subtask<Lattice>> {
    static partial_pml_2d_subtask<Lattice> create(
        const simulation_engine<Lattice>& engine,
        sclx::kernel_handler& handler
    );
};

}  // namespace naga::fluids::nonlocal_lbm::detail