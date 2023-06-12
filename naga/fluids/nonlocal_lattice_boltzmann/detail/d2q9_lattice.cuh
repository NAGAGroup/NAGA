
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

#include "lattices.cuh"

namespace naga::fluids::nonlocal_lbm {
template<class T>
struct d2q9_lattice;
}

namespace naga::fluids::nonlocal_lbm::detail {

template <class T>
struct d2q9_lattice_velocities: lattice_velocities_t<T, 2, 9> {
    T vals[9][2] = {
        { 0,  0},
        {-1,  1},
        {-1,  0},
        {-1, -1},
        { 0, -1},
        { 1, -1},
        { 1,  0},
        { 1,  1},
        { 0,  1}
    };
};

template <class T>
struct d2q9_lattice_weights: lattice_weights_t<T, 9> {
    T vals[9] = {4.0 / 9.0,
                 1.0 / 36.0,
                 1.0 / 9.0,
                 1.0 / 36.0,
                 1.0 / 9.0,
                 1.0 / 36.0,
                 1.0 / 9.0,
                 1.0 / 36.0,
                 1.0 / 9.0};
};

template<class T>
struct lattice_interface<d2q9_lattice<T>> {
    static constexpr uint size       = d2q9_lattice<T>::size;
    static constexpr uint dimensions = d2q9_lattice<T>::dimensions;
    using value_type                 = typename d2q9_lattice<T>::value_type;

    static constexpr d2q9_lattice_velocities<T> lattice_velocities() {
        return d2q9_lattice_velocities<T>{};
    }

    static constexpr d2q9_lattice_weights<T> lattice_weights() {
        return d2q9_lattice_weights<T>{};
    }

    static constexpr int get_bounce_back_idx(const int &alpha) {
        typedef enum {
            r  = 0,
            nw = 1,
            w  = 2,
            sw = 3,
            s  = 4,
            se = 5,
            e  = 6,
            ne = 7,
            n  = 8,
        } lattice_directions;
        switch (alpha) {
        case lattice_directions::r: return lattice_directions::r;
        case lattice_directions::n: return lattice_directions::s;
        case lattice_directions::ne: return lattice_directions::sw;
        case lattice_directions::e: return lattice_directions::w;
        case lattice_directions::se: return lattice_directions::nw;
        case lattice_directions::s: return lattice_directions::n;
        case lattice_directions::sw: return lattice_directions::ne;
        case lattice_directions::w: return lattice_directions::e;
        case lattice_directions::nw: return lattice_directions::se;
        default: return -1;
        }
    }
};

}  // namespace naga::fluids::nonlocal_lbm::detail