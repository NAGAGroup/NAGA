
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

#include "../simulation_observer.cuh"

#include <AudioFile.h>
#include <numeric>
#include <utility>

namespace naga::fluids::nonlocal_lbm {

enum class channel_configuration { mono, stereo };

template<class T, uint Dimensions, channel_configuration ChannelConfig>
struct audio_sink_traits;

template<class T, uint Dimensions>
struct audio_sink_traits<T, Dimensions, channel_configuration::mono> {
    static constexpr uint channel_count = 1;
    using location_type                 = naga::point_t<T, Dimensions>;
    using signal_history_type           = std::deque<T>;
};

template<class T, uint Dimensions>
struct audio_sink_traits<T, Dimensions, channel_configuration::stereo> {
    static constexpr uint channel_count = 2;
    using location_type
        = std::pair<naga::point_t<T, Dimensions>, naga::point_t<T, Dimensions>>;
    using signal_history_type = std::deque<std::pair<T, T>>;
};

template<class Lattice, channel_configuration ChannelConfig>
class audio_sink_observer : public simulation_observer<Lattice> {
  public:
    using lattice_type               = Lattice;
    static constexpr uint dimensions = lattice_traits<lattice_type>::dimensions;
    static constexpr auto channel_config = ChannelConfig;
    using base                           = simulation_observer<Lattice>;
    using value_type                     = typename base::value_type;
    using simulation_domain_t            = typename base::simulation_domain_t;
    using problem_parameters_t           = typename base::problem_parameters_t;
    using solution_t                     = typename base::solution_t;

    static constexpr uint channel_count
        = audio_sink_traits<value_type, dimensions, channel_config>::
            channel_count;
    using location_type =
        typename audio_sink_traits<value_type, dimensions, channel_config>::
            location_type;

    audio_sink_observer(
        sclx::filesystem::path output_wav_file,
        const value_type& sample_rate,
        const location_type& location,
        const value_type& nominal_density,
        const simulation_domain_t& domain,
        const value_type& time_multiplier = 1,
        uint moving_avg_samples           = 10,
        size_t save_frequency             = 10
    )
        : sample_rate_(sample_rate),
          output_wav_file_(std::move(output_wav_file)),
          save_frequency_(save_frequency),
          time_multiplier_(time_multiplier) {

        std::vector<const naga::point_t<value_type, dimensions>*> sink_locations(
            channel_count
        );
        if constexpr (channel_config == channel_configuration::mono) {
            const typename audio_sink_traits<
                value_type,
                dimensions,
                channel_configuration::mono>::location_type& location_ref
                = location;
            sink_locations[0] = &location_ref;
        } else {
            const typename audio_sink_traits<
                value_type,
                dimensions,
                channel_configuration::stereo>::location_type& location_ref
                = location;
            sink_locations[0] = &location_ref.first;
            sink_locations[0] = &location_ref.second;
        }
        sink_locations_ = sclx::array<value_type, 2>{dimensions, channel_count};
        for (int i = 0; i < channel_count; ++i) {
            std::copy(
                &((*sink_locations[i])[0]),
                &((*sink_locations[i])[0]) + dimensions,
                &sink_locations_(0, i)
            );
        }

        using namespace naga::segmentation;
        nd_cubic_segmentation<value_type, dimensions> domain_segmentation(
            domain.points,
            32
        );
        naga::default_point_map<value_type, 3> location_map{sink_locations_};
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

        signal_history_.resize(moving_avg_samples);
        std::fill(
            signal_history_.begin(),
            signal_history_.end(),
            nominal_density
        );

        audio_buffer_.resize(channel_count);
    }

    void update(
        const value_type& time,
        const simulation_domain_t&,
        const problem_parameters_t&,
        const solution_t& solution
    ) {
        auto frame
            = static_cast<size_t>(time_multiplier_ * time * sample_rate_);
        if (frame < audio_buffer_[0].size()) {
            return;
        }
        interpolator_->interpolate(
            solution.macroscopic_values.fluid_density,
            sink_signal_
        );
        if constexpr (ChannelConfig == channel_configuration::mono) {
            value_type moving_avg = std::reduce(
                                        signal_history_.begin(),
                                        signal_history_.end(),
                                        0.0
                                    )
                                  / signal_history_.size();

            signal_history_.pop_front();
            signal_history_.push_back(sink_signal_[0]);

            sink_signal_[0] -= moving_avg;

            audio_buffer_[0].push_back(sink_signal_[0]);
        } else {
            value_type moving_avg[2];

            moving_avg[0] = std::reduce(
                                signal_history_.begin(),
                                signal_history_.end(),
                                0.0,
                                [](value_type acc,
                                   const std::pair<value_type, value_type>& pair
                                ) { return acc + pair.first; }
                            )
                          / signal_history_.size();

            moving_avg[1] = std::reduce(
                                signal_history_.begin(),
                                signal_history_.end(),
                                0.0,
                                [](value_type acc,
                                   const std::pair<value_type, value_type>& pair
                                ) { return acc + pair.second; }
                            )
                          / signal_history_.size();

            signal_history_.pop_front();
            signal_history_.push_back({sink_signal_[0], sink_signal_[1]});

            sink_signal_[0] -= moving_avg[0];
            sink_signal_[1] -= moving_avg[1];

            audio_buffer_[0].push_back(sink_signal_[0]);
            audio_buffer_[1].push_back(sink_signal_[1]);
        }

        for (const auto& signal : sink_signal_) {
            if (std::abs(signal) > max_signal_) {
                max_signal_ = std::abs(signal);
            }
        }

        if (audio_buffer_[0].size() % save_frequency_ == 0) {
            save();
        }
    }

    void save() {
        AudioFile<value_type> audio_file;
        auto audio_buffer_copy = audio_buffer_;
        for (auto& channel : audio_buffer_copy) {
            std::transform(
                channel.begin(),
                channel.end(),
                channel.begin(),
                [&](value_type signal) { return signal / max_signal_; }
            );
        }
        audio_file.setAudioBuffer(audio_buffer_copy);
        audio_file.setSampleRate(sample_rate_);
        audio_file.setBitDepth(16);
        audio_file.save(output_wav_file_.string());
    }

    ~audio_sink_observer() { save(); }

  private:
    sclx::array<value_type, 2> sink_locations_;
    sclx::array<value_type, 1> sink_signal_{channel_count};

    using interpolator_t = naga::interpolation::radial_point_method<value_type>;
    std::shared_ptr<interpolator_t> interpolator_;
    using dequeue_t =
        typename audio_sink_traits<value_type, dimensions, channel_config>::
            signal_history_type;
    dequeue_t signal_history_;

    value_type time_multiplier_;
    typename AudioFile<value_type>::AudioBuffer audio_buffer_;
    sclx::filesystem::path output_wav_file_;
    value_type sample_rate_;
    size_t save_frequency_;
    value_type max_signal_ = 0.0;
};

}  // namespace naga::fluids::nonlocal_lbm
