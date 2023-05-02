
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

#include "detail/divergence.cuh"

namespace naga::nonlocal_calculus {

template<class T, uint Dimensions>
class divergence_operator {
  public:
    friend class operator_builder<T, Dimensions>;

    static divergence_operator create(const sclx::array<T, 2>& domain) {
        operator_builder<T, Dimensions> builder(domain);
        return builder.template create<detail::divergence_operator_type>();
    }

    void apply(
        const sclx::array<T, 2>& field,
        const sclx::array<T, 1>& result,
        const T& centering_offset = T(0)
    ) const {
        if (result.elements() != weights_.shape()[2]) {
            sclx::throw_exception<std::invalid_argument>(
                "Result array has incorrect shape.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }
        if (Dimensions != field.shape()[0]) {
            sclx::throw_exception<std::invalid_argument>(
                "Field array has incorrect shape.",
                "naga::nonlocal_calculus::divergence_operator::"
            );
        }

        sclx::execute_kernel([&](const sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<1>{result.shape()[0]},
                result,
                [=,
                 *this] __device__(const sclx::md_index_t<1>& index, const auto&) {
                    T divergence = 0;
                    for (uint idx = 0;
                         idx < support_indices_.shape()[0] * Dimensions;
                         ++idx) {
                        uint d = idx % Dimensions;
                        uint s = idx / Dimensions;
                        divergence += weights_(d, s, index[0])
                                    * (field(d, support_indices_(s, index[0]))
                                       - centering_offset);
                    }
                    result[index] = divergence;
                }
            );
        });
    }

  private:
    divergence_operator() = default;

    static divergence_operator create(
        const sclx::array<T, 2>& domain,
        const sclx::array<sclx::index_t, 2>& support_indices,
        const sclx::array<T, 2>& quad_interp_weights,
        const sclx::array<T, 1>& interaction_radii
    ) {
        divergence_operator op;
        op.weights_ = sclx::array<T, 3>{
            Dimensions,
            support_indices.shape()[0],
            domain.shape()[1]};
        op.support_indices_ = support_indices;

        detail::compute_divergence_weights<T, Dimensions>(
            op.weights_,
            domain,
            interaction_radii,
            quad_interp_weights,
            support_indices
        );

        return op;
    }

    sclx::array<T, 3> weights_;
    sclx::array<sclx::index_t, 2> support_indices_;
};

}  // namespace naga::nonlocal_calculus

#include "detail/divergence.inl"