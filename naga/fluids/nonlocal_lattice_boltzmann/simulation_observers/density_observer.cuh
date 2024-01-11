
// BSD 3-Clause License
//
// Copyright (c) 2023
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

#include "../../../interpolation/radial_point_method.cuh"
#include "../../../segmentation/nd_cubic_segmentation.cuh"
#include "../simulation_observer.cuh"

#include <AudioFile.h>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <numeric>
#include <utility>

namespace naga::fluids::nonlocal_lbm {

template<class Lattice>
class density_observer : public simulation_observer<Lattice> {
  public:
    using lattice_type               = Lattice;
    static constexpr uint dimensions = lattice_traits<lattice_type>::dimensions;
    using base                           = simulation_observer<Lattice>;
    using value_type                     = typename base::value_type;
    using simulation_domain_t            = typename base::simulation_domain_t;
    using problem_parameters_t           = typename base::problem_parameters_t;
    using solution_t                     = typename base::solution_t;

    auto density_values() const { return density_values_; }

    auto time_values() const { return time_values_; }

    density_observer(
        const naga::point_t<value_type, dimensions> & location,
        const simulation_domain_t& domain,
        const value_type& time_multiplier = 1
    )
        :
          time_multiplier_(time_multiplier),
          current_density_{1}
    {

        observer_location_ = sclx::array<value_type, 2>{dimensions, 1};
        std::copy(
            &location[0],
            &location[0] + dimensions,
            &observer_location_(0, 0)
        );

        using namespace naga::segmentation;
        nd_cubic_segmentation<value_type, dimensions> domain_segmentation(
            domain.points,
            uint{32}
        );
        naga::default_point_map<value_type, dimensions> location_map{
            observer_location_};
        auto [distances_squared, indices]
            = batched_nearest_neighbors(32, location_map, domain_segmentation);

        interpolator_ = std::make_shared<interpolator_t>(
            interpolator_t::create_interpolator(
                domain.points,
                indices,
                location_map,
                domain.nodal_spacing
            )
        );
    }

    void update(
        const value_type& time,
        const simulation_domain_t&,
        const problem_parameters_t& params,
        const solution_t& solution
    ) {
        auto scaled_time = time * time_multiplier_;
        interpolator_->interpolate(
            solution.macroscopic_values.fluid_density,
            current_density_,
            params.nominal_density
        );
        density_values_.push_back(current_density_[0]);
        time_values_.push_back(scaled_time);
    }

    template<class Archive>
    void save_state(Archive& ar) const {
        sclx::serialize_array<Archive, value_type, 2>(ar, observer_location_);
        sclx::serialize_array<Archive, value_type, 1>(ar, current_density_);
        ar(density_values_);
        ar(time_values_);
        ar(*interpolator_);
        ar(time_multiplier_);
    }

    template<class Archive>
    void load_state(Archive& ar) {
        sclx::deserialize_array<Archive, value_type, 2>(ar, observer_location_);
        sclx::deserialize_array<Archive, value_type, 1>(ar, current_density_);
        ar(density_values_);
        ar(time_values_);
        ar(*interpolator_);
        ar(time_multiplier_);
    }

  private:
    sclx::array<value_type, 2> observer_location_;
    sclx::array<value_type, 1> current_density_;
    std::vector<value_type> density_values_;
    std::vector<value_type> time_values_;

    using interpolator_t = naga::interpolation::radial_point_method<value_type>;
    std::shared_ptr<interpolator_t> interpolator_;

    value_type time_multiplier_;
};

}  // namespace naga::fluids::nonlocal_lbm
