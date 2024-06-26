#include <naga/fluids/nonlocal_lattice_boltzmann.cuh>
#include <naga/path.hpp>
#include <naga/regions/hypersphere.cuh>

template<class Lattice>
struct problem_traits {
    using lattice_type = Lattice;
    using value_type   = typename naga::fluids::nonlocal_lbm::lattice_traits<
          lattice_type>::value_type;
    static constexpr uint dimensions
        = naga::fluids::nonlocal_lbm::lattice_traits<lattice_type>::dimensions;

    using sim_engine_t
        = naga::fluids::nonlocal_lbm::simulation_engine<lattice_type>;

    using density_source_t
        = naga::fluids::nonlocal_lbm::density_source<lattice_type>;

    using velocity_source_t
        = naga::fluids::nonlocal_lbm::velocity_source<lattice_type>;
    using simulation_domain_t  = typename sim_engine_t::simulation_domain_t;
    using problem_parameters_t = typename sim_engine_t::problem_parameters_t;
    using solution_t           = typename sim_engine_t::solution_t;
    using region_t = naga::regions::hypersphere<value_type, dimensions>;

    using node_provider_t = naga::experimental::fluids::nonlocal_lbm::
        conforming_point_cloud_provider<lattice_type>;

    using path_t = naga::path_t<value_type, dimensions>;

    class zero_path_t : public path_t {
      public:
        using base       = path_t;
        using point_type = typename base::point_type;

        static std::shared_ptr<path_t> create(const point_type& origin) {
            return std::shared_ptr<path_t>(new zero_path_t(origin));
        }

        const point_type& operator()(const value_type&) final {
            return origin_;
        }

      private:
        zero_path_t(point_type origin) : origin_(std::move(origin)) {}
        naga::point_t<value_type, dimensions> origin_;
    };

    class csv_path_t : public path_t {
      public:
        using base       = path_t;
        using point_type = typename base::point_type;

        static std::shared_ptr<path_t> create(
            const std::string& csv_file,
            const value_type& desired_step_size,
            point_type origin = point_type{},
            point_type lower_bound
            = point_type{{-std::numeric_limits<value_type>::infinity(), -std::numeric_limits<value_type>::infinity(), -std::numeric_limits<value_type>::infinity()}
            },
            point_type upper_bound
            = point_type{{std::numeric_limits<value_type>::infinity(), std::numeric_limits<value_type>::infinity(), std::numeric_limits<value_type>::infinity()}
            }
        ) {
            return std::shared_ptr<path_t>(new csv_path_t(
                csv_file,
                desired_step_size,
                origin,
                lower_bound,
                upper_bound
            ));
        }

        const point_type& operator()(const value_type& t) final {
            auto index = static_cast<size_t>(t / step_size_);
            if (index >= points_.size()) {
                return points_.back();
            }
            return points_[index];
        }

      private:
        csv_path_t(
            const std::string& csv_file,
            const value_type& desired_step_size,
            point_type origin,
            point_type lower_bound,
            point_type upper_bound
        )
            : step_size_{desired_step_size} {
            std::vector<point_type> loaded_points;
            std::vector<value_type> loaded_times;
            std::ifstream file(csv_file);
            // discard first line
            std::string line;
            std::getline(file, line);
            while (std::getline(file, line)) {
                std::stringstream line_stream(line);
                std::string cell;
                std::getline(line_stream, cell, ',');
                value_type time = std::stod(cell);
                point_type point;
                for (int i = 0; i < dimensions; ++i) {
                    std::getline(line_stream, cell, ',');
                    point[i] = std::stod(cell);
                }
                loaded_times.push_back(time);
                loaded_points.push_back(point);
            }
            value_type duration = loaded_times.back();
            for (value_type t = 0.0; t <= duration; t += step_size_) {
                // find first loaded time <= t
                auto lower_ptr = std::lower_bound(
                    loaded_times.begin(),
                    loaded_times.end(),
                    t
                );
                auto lower_time = *lower_ptr;
                auto lower_index
                    = std::distance(loaded_times.begin(), lower_ptr);
                auto upper_index = lower_index != loaded_times.size() - 1
                                     ? lower_index + 1
                                     : lower_index;
                auto lower_weight
                    = 1.0
                    - (t - lower_time)
                          / (loaded_times[upper_index] - lower_time);
                auto upper_weight = 1.0 - lower_weight;
                point_type point;
                for (int i = 0; i < dimensions; ++i) {
                    point[i] = lower_weight * loaded_points[lower_index][i]
                             + upper_weight * loaded_points[upper_index][i];
                }
                points_.push_back(point);
            }

            std::transform(
                points_.begin(),
                points_.end(),
                points_.begin(),
                [&](auto point) {
                    point[0] += origin[0];
                    point[1] += origin[1];
                    point[2] += origin[2];
                    return point;
                }
            );

            // clip to bounds
            std::transform(
                points_.begin(),
                points_.end(),
                points_.begin(),
                [&](auto point) {
                    for (int i = 0; i < dimensions; ++i) {
                        if (point[i] < lower_bound[i]) {
                            point[i] = lower_bound[i];
                        }
                        if (point[i] > upper_bound[i]) {
                            point[i] = upper_bound[i];
                        }
                    }
                    return point;
                }
            );
        }
        std::vector<point_type> points_;
        point_type current_point_;
        value_type step_size_;
    };

    class circular_path : public path_t {
      public:
        using base       = path_t;
        using point_type = typename base::point_type;

        static std::shared_ptr<path_t>
        create(const value_type& radius, const value_type& period) {
            return std::shared_ptr<path_t>(new circular_path(radius, period));
        }

        const point_type& operator()(const value_type& t) final {
            current_point_[0]
                = radius_
                * naga::math::cos(2 * naga::math::pi<value_type> * t / period_);
            current_point_[1]
                = radius_
                * naga::math::sin(2 * naga::math::pi<value_type> * t / period_);

            return current_point_;
        }

      private:
        circular_path(value_type radius, value_type period)
            : radius_(radius),
              period_(period) {}
        value_type radius_;
        value_type period_;
        point_type current_point_;
    };

    class linear_path : public path_t {
      public:
        using base       = path_t;
        using point_type = typename base::point_type;

        static std::shared_ptr<path_t>
        create(const point_type& start_pos, const point_type& velocity) {
            return std::shared_ptr<path_t>(new linear_path(start_pos, velocity)
            );
        }

        const point_type& operator()(const value_type& t) final {
            for (int i = 0; i < dimensions; ++i) {
                current_point_[i] = start_pos_[i] + velocity_[i] * t;
            }

            return current_point_;
        }

      private:
        linear_path(point_type start_pos, point_type velocity)
            : start_pos_(std::move(start_pos)),
              velocity_(std::move(velocity)) {}

        point_type start_pos_;
        point_type velocity_;
        point_type current_point_;
    };

    class spherical_audio_source_v2 : public velocity_source_t {
      public:
        spherical_audio_source_v2(
            const sclx::filesystem::path& wav_file,
            const value_type& amplitude,
            const value_type& source_radius,
            const value_type& max_frequency,
            const value_type& time_multiplier = 1.0,
            const size_t& frame_offset        = 0,
            std::shared_ptr<path_t> path
            = zero_path_t::create(naga::point_t<value_type, dimensions>{}),
            bool loop_audio = false
        )
            : amplitude_(amplitude),
              source_outer_radius_(source_radius),
              source_inner_radius_(source_radius * .5f),
              max_frequency_(max_frequency),
              time_multiplier_(time_multiplier),
              frame_offset_(frame_offset),
              loop_audio_(loop_audio) {
            AudioFile<value_type> audio_file(wav_file.string());
            if (!audio_file.isMono()) {
                sclx::throw_exception<std::runtime_error>(
                    "Audio file must be mono",
                    "spherical_audio_source::"
                );
            }
            sample_rate_ = audio_file.getSampleRate();
            if (sample_rate_ < max_frequency_ * 8) {
                sclx::throw_exception<std::runtime_error>(
                    "Sample rate must be at least 4 times the max frequency",
                    "spherical_audio_source::"
                );
            }

            audio_samples_ = sclx::array<value_type, 1>{
                static_cast<const size_t&>(audio_file.getNumSamplesPerChannel())
            };

            std::copy(
                audio_file.samples[0].begin(),
                audio_file.samples[0].end(),
                audio_samples_.begin()
            );

            auto max_source_term = *std::max_element(
                audio_samples_.begin(),
                audio_samples_.end()
            );

            if (max_source_term == 0.0) {
                sclx::throw_exception<std::runtime_error>(
                    "Audio file must not be silent",
                    "spherical_audio_source::"
                );
            }

            std::transform(
                audio_samples_.begin(),
                audio_samples_.end(),
                audio_samples_.begin(),
                [=](const auto& sample) { return sample / max_source_term; }
            );

            path_ = std::move(path);

            std::cout << "Audio file has " << audio_samples_.elements()
                      << " samples at " << audio_file.getSampleRate()
                      << " Hz\n\n";
        }

        std::future<void> add_velocity_source(
            const simulation_domain_t& domain,
            const problem_parameters_t& params,
            const solution_t& solution,
            const value_type& time,
            sclx::array<value_type, 2>& velocity_terms
        ) final {
            // if (odd_sim_frame_) {
            //     odd_sim_frame_ = !odd_sim_frame_;
            //     return std::async(std::launch::deferred, []() {});
            // }
            // odd_sim_frame_ = !odd_sim_frame_;
            region_t source_region{
                source_outer_radius_,
                (*path_)(time * time_multiplier_)
            };

            auto lower_frame_number = std::floor(
                time * time_multiplier_ * static_cast<value_type>(sample_rate_)
            );
            auto upper_frame_number = lower_frame_number + 1;
            auto fractional_frame_number
                = time * time_multiplier_
                * static_cast<value_type>(sample_rate_);
            auto lower_weight
                = 1.0 - (fractional_frame_number - lower_frame_number);
            auto upper_weight
                = 1.0 - (upper_frame_number - fractional_frame_number);

            auto frame_number
                = static_cast<size_t>(lower_frame_number) + frame_offset_;
            const auto& amplitude       = amplitude_;
            const auto& audio_samples   = audio_samples_;
            const auto& nominal_density = params.nominal_density;
            const auto& fluid_velocity
                = solution.macroscopic_values.fluid_velocity;
            const auto& points = domain.points;

            auto audio_sample_upper
                = amplitude * audio_samples(frame_number + 1);
            auto audio_sample_lower = amplitude * audio_samples(frame_number);
            auto audio_sample       = upper_weight * audio_sample_upper
                              + lower_weight * audio_sample_lower;

            if (loop_audio_) {
                frame_number = frame_number % audio_samples.elements();
            }

            if (frame_number >= audio_samples.elements()) {
                if (!has_finished_) {
                    std::cout << "Audio source has finished at frame "
                              << frame_number << "\n\n";
                    has_finished_ = true;
                }
                return std::async(std::launch::deferred, []() {});
            }

            auto bulk_end = domain.num_bulk_points + domain.num_layer_points;

            return sclx::execute_kernel([=](sclx::kernel_handler& handler
                                        ) mutable {
                sclx::local_array<value_type, 2> local_points(
                    handler,
                    {dimensions,
                     sclx::cuda::traits::kernel::default_block_shape[0]}
                );

                auto& source_outer_radius = source_outer_radius_;
                auto& source_inner_radius = source_inner_radius_;
                handler.launch(
                    sclx::md_range_t<1>{velocity_terms.shape()[1]},
                    velocity_terms,
                    [=] __device__(
                        const sclx::md_index_t<>& idx,
                        const sclx::kernel_info<>& info
                    ) mutable {
                        if (idx[0] >= bulk_end) {
                            return;
                        }

                        for (int i = 0; i < dimensions; ++i) {
                            local_points(i, info.local_thread_linear_id())
                                = points(i, idx[0]);
                        }

                        if (source_region.contains(
                                &local_points(0, info.local_thread_linear_id())
                            )) {
                            auto perturbation = audio_sample;
                            auto fluid_speed
                                = naga::math::loopless::norm<dimensions>(
                                    &fluid_velocity(0, idx[0])
                                );
                            //                            perturbation +=
                            //                            fluid_speed;

                            naga::point_t<value_type, dimensions>
                                center_to_point;
                            for (int i = 0; i < dimensions; ++i) {
                                center_to_point[i]
                                    = local_points(
                                          i,
                                          info.local_thread_linear_id()
                                      )
                                    - source_region.center()[i];
                            }
                            auto distance = naga::math::sqrt(
                                naga::math::loopless::dot<dimensions>(
                                    center_to_point,
                                    center_to_point
                                )
                            );
                            if (distance < source_inner_radius) {
                                return;
                            }

                            auto source_thickness
                                = source_outer_radius - source_inner_radius;

                            for (int i = 0; i < dimensions; ++i) {
                                velocity_terms(i, idx[0]) = perturbation
                                                          * center_to_point[i]
                                                          / distance;
                            }
                        }
                    }
                );
            });
        }

        const size_t& sample_rate() const { return sample_rate_; }

        value_type audio_length() const {
            return static_cast<value_type>(audio_samples_.elements())
                 / static_cast<value_type>(sample_rate_);
        }

        size_t sample_count() const { return audio_samples_.elements(); }

      private:
        sclx::array<value_type, 1> audio_samples_;
        value_type amplitude_;
        value_type source_outer_radius_;
        value_type source_inner_radius_;
        value_type max_frequency_;
        value_type time_multiplier_;
        size_t sample_rate_;
        size_t frame_offset_;
        bool loop_audio_;
        bool has_finished_ = false;
        std::shared_ptr<path_t> path_;
        bool odd_sim_frame_ = false;
    };

    class spherical_audio_source : public density_source_t {
      public:
        spherical_audio_source(
            const sclx::filesystem::path& wav_file,
            const value_type& amplitude,
            const value_type& source_radius,
            const value_type& max_frequency,
            const value_type& time_multiplier = 1.0,
            const size_t& frame_offset        = 0,
            std::shared_ptr<path_t> path
            = zero_path_t::create(naga::point_t<value_type, dimensions>{}),
            bool loop_audio = false
        )
            : amplitude_(amplitude),
              source_radius_(source_radius),
              max_frequency_(max_frequency),
              time_multiplier_(time_multiplier),
              frame_offset_(frame_offset),
              loop_audio_(loop_audio) {
            AudioFile<value_type> audio_file(wav_file.string());
            if (!audio_file.isMono()) {
                sclx::throw_exception<std::runtime_error>(
                    "Audio file must be mono",
                    "spherical_audio_source::"
                );
            }
            sample_rate_ = audio_file.getSampleRate();
            if (sample_rate_ < max_frequency_ * 8) {
                sclx::throw_exception<std::runtime_error>(
                    "Sample rate must be at least 4 times the max frequency",
                    "spherical_audio_source::"
                );
            }

            audio_samples_ = sclx::array<value_type, 1>{
                static_cast<const size_t&>(audio_file.getNumSamplesPerChannel())
            };

            std::copy(
                audio_file.samples[0].begin(),
                audio_file.samples[0].end(),
                audio_samples_.begin()
            );

            auto max_source_term = *std::max_element(
                audio_samples_.begin(),
                audio_samples_.end()
            );

            if (max_source_term == 0.0) {
                sclx::throw_exception<std::runtime_error>(
                    "Audio file must not be silent",
                    "spherical_audio_source::"
                );
            }

            std::transform(
                audio_samples_.begin(),
                audio_samples_.end(),
                audio_samples_.begin(),
                [=](const auto& sample) { return sample / max_source_term; }
            );

            path_ = std::move(path);

            std::cout << "Audio file has " << audio_samples_.elements()
                      << " samples at " << audio_file.getSampleRate()
                      << " Hz\n\n";
        }

        std::future<void> add_density_source(
            const simulation_domain_t& domain,
            const problem_parameters_t& params,
            const solution_t& solution,
            const value_type& time,
            sclx::array<value_type, 1>& source_terms
        ) final {
            region_t source_region{
                source_radius_,
                (*path_)(time * time_multiplier_)
            };

            value_type lower_frame_number = std::floor(
                time * time_multiplier_ * static_cast<value_type>(sample_rate_)
            );
            value_type upper_frame_number = lower_frame_number + 1;
            auto fractional_frame_number
                = time * time_multiplier_
                * static_cast<value_type>(sample_rate_);
            auto lower_weight
                = 1.0 - (fractional_frame_number - lower_frame_number);
            auto upper_weight
                = 1.0 - (upper_frame_number - fractional_frame_number);

            auto frame_number
                = static_cast<size_t>(lower_frame_number) + frame_offset_;
            const auto& amplitude     = amplitude_;
            const auto& audio_samples = audio_samples_;
            const auto& density = solution.macroscopic_values.fluid_density;
            const auto& nominal_density = params.nominal_density;
            const auto& points          = domain.points;

            auto audio_sample_upper
                = amplitude * audio_samples(frame_number + 1);
            auto audio_sample_lower = amplitude * audio_samples(frame_number);
            auto audio_sample       = upper_weight * audio_sample_upper
                              + lower_weight * audio_sample_lower;

            if (loop_audio_) {
                frame_number = frame_number % audio_samples.elements();
            }

            if (frame_number >= audio_samples.elements()) {
                if (!has_finished_) {
                    std::cout << "Audio source has finished at frame "
                              << frame_number << "\n\n";
                    has_finished_ = true;
                }
                return std::async(std::launch::deferred, []() {});
            }

            auto bulk_end = domain.num_bulk_points + domain.num_layer_points;

            return sclx::execute_kernel([=](sclx::kernel_handler& handler
                                        ) mutable {
                sclx::local_array<value_type, 2> local_points(
                    handler,
                    {dimensions,
                     sclx::cuda::traits::kernel::default_block_shape[0]}
                );

                auto& source_radius = source_radius_;
                handler.launch(
                    sclx::md_range_t<1>{source_terms.shape()},
                    source_terms,
                    [=] __device__(
                        const sclx::md_index_t<1>& idx,
                        const sclx::kernel_info<>& info
                    ) mutable {
                        if (idx[0] >= bulk_end) {
                            return;
                        }

                        for (int i = 0; i < dimensions; ++i) {
                            local_points(i, info.local_thread_linear_id())
                                = points(i, idx[0]);
                        }
                        if (source_region.contains(
                                &local_points(0, info.local_thread_linear_id())
                            )) {
                            source_terms(idx[0]) = audio_sample;
                        }
                    }
                );
            });
        }

        const size_t& sample_rate() const { return sample_rate_; }

        value_type audio_length() const {
            return static_cast<value_type>(audio_samples_.elements())
                 / static_cast<value_type>(sample_rate_);
        }

        size_t sample_count() const { return audio_samples_.elements(); }

      private:
        sclx::array<value_type, 1> audio_samples_;
        value_type max_frequency_;
        value_type amplitude_;
        value_type source_radius_;
        value_type time_multiplier_;
        size_t sample_rate_;
        size_t frame_offset_;
        bool loop_audio_;
        bool has_finished_ = false;
        std::shared_ptr<path_t> path_;
        bool odd_sim_frame_ = false;
    };

    class spherical_sine_wave_source : public density_source_t {
      public:
        spherical_sine_wave_source(
            value_type radius,
            value_type frequency,
            value_type amplitude,
            value_type time_multiplier = 1.0,
            std::shared_ptr<path_t> path
            = zero_path_t::create(naga::point_t<value_type, dimensions>{})
        )
            : radius_(radius),
              frequency_(frequency),
              amplitude_(amplitude),
              time_multiplier_(time_multiplier),
              path_(std::move(path)) {}

        auto get_amplitude_at_time(const value_type& time) const {
            return amplitude_
                 * naga::math::sin(
                       2 * naga::math::pi<value_type> * frequency_ * time
                 );
        }

        std::future<void> add_density_source(
            const simulation_domain_t& domain,
            const problem_parameters_t& params,
            const solution_t& solution,
            const value_type& time,
            sclx::array<value_type, 1>& source_terms
        ) final {
            region_t source_region{radius_, (*path_)(time * time_multiplier_)};

            auto scaled_time    = time * time_multiplier_;
            const auto& density = solution.macroscopic_values.fluid_density;
            const auto& nominal_density = params.nominal_density;
            const auto& points          = domain.points;
            auto frame_amplitude        = get_amplitude_at_time(scaled_time);

            auto bulk_end = domain.num_bulk_points + domain.num_layer_points;

            return sclx::execute_kernel([=](sclx::kernel_handler& handler
                                        ) mutable {
                sclx::local_array<value_type, 2> local_points(
                    handler,
                    {dimensions,
                     sclx::cuda::traits::kernel::default_block_shape[0]}
                );

                handler.launch(
                    sclx::md_range_t<1>{source_terms.shape()},
                    source_terms,
                    [=] __device__(
                        const sclx::md_index_t<1>& idx,
                        const sclx::kernel_info<>& info
                    ) mutable {
                        if (idx[0] >= bulk_end) {
                            return;
                        }
                        for (int i = 0; i < dimensions; ++i) {
                            local_points(i, info.local_thread_linear_id())
                                = points(i, idx[0]);
                        }
                        if (source_region.contains(
                                &local_points(0, info.local_thread_linear_id())
                            )) {
                            source_terms(idx[0]) = frame_amplitude;
                        }
                    }
                );
            });
        }

      private:
        value_type radius_;
        value_type frequency_;
        value_type amplitude_;
        value_type time_multiplier_;
        std::shared_ptr<path_t> path_;
    };

    template<class SourceRegion>
    class audio_source : public density_source_t {
      public:
        using region_type = SourceRegion;

        audio_source(
            region_type source_region,
            const sclx::filesystem::path& wav_file,
            const value_type& amplitude,
            const size_t& frame_offset = 0,
            std::shared_ptr<path_t> path
            = zero_path_t::create(naga::point_t<value_type, dimensions>{}),
            bool loop_audio = false
        )
            : amplitude_(amplitude),
              source_region_(std::move(source_region)),
              frame_offset_(frame_offset),
              loop_audio_(loop_audio) {
            AudioFile<value_type> audio_file(wav_file.string());
            if (!audio_file.isMono()) {
                sclx::throw_exception<std::runtime_error>(
                    "Audio file must be mono",
                    "audio_source::"
                );
            }
            sample_rate_ = audio_file.getSampleRate();

            audio_samples_ = sclx::array<value_type, 1>{
                static_cast<const size_t&>(audio_file.getNumSamplesPerChannel())
            };

            std::copy(
                audio_file.samples[0].begin(),
                audio_file.samples[0].end(),
                audio_samples_.begin()
            );

            auto max_source_term = *std::max_element(
                audio_samples_.begin(),
                audio_samples_.end()
            );

            if (max_source_term == 0.0) {
                sclx::throw_exception<std::runtime_error>(
                    "Audio file must not be silent",
                    "audio_source::"
                );
            }

            std::transform(
                audio_samples_.begin(),
                audio_samples_.end(),
                audio_samples_.begin(),
                [=](const auto& sample) { return sample / max_source_term; }
            );

            path_ = std::move(path);

            std::cout << "Audio file has " << audio_samples_.elements()
                      << " samples at " << audio_file.getSampleRate()
                      << " Hz\n\n";
        }

        std::future<void> add_density_source(
            const simulation_domain_t& domain,
            const problem_parameters_t& params,
            const solution_t& solution,
            const value_type& time,
            sclx::array<value_type, 1>& source_terms
        ) final {
            value_type lower_frame_number
                = std::floor(time * static_cast<value_type>(sample_rate_));
            value_type upper_frame_number = lower_frame_number + 1;
            auto fractional_frame_number
                = time * static_cast<value_type>(sample_rate_);
            auto lower_weight
                = 1.0 - (fractional_frame_number - lower_frame_number);
            auto upper_weight
                = 1.0 - (upper_frame_number - fractional_frame_number);

            auto frame_number
                = static_cast<size_t>(lower_frame_number) + frame_offset_;
            const auto& amplitude     = amplitude_;
            const auto& audio_samples = audio_samples_;
            const auto& density = solution.macroscopic_values.fluid_density;
            const auto& nominal_density = params.nominal_density;
            const auto& points          = domain.points;

            auto audio_sample_upper
                = amplitude * audio_samples(frame_number + 1);
            auto audio_sample_lower = amplitude * audio_samples(frame_number);
            auto audio_sample       = upper_weight * audio_sample_upper
                              + lower_weight * audio_sample_lower;
            auto source_region = source_region_;
            source_region.shift_region((*path_)(time));

            if (loop_audio_) {
                frame_number = frame_number % audio_samples.elements();
            }

            if (frame_number >= audio_samples.elements()) {
                if (!has_finished_) {
                    std::cout << "Audio source has finished at frame "
                              << frame_number << "\n\n";
                    has_finished_ = true;
                }
                return std::async(std::launch::deferred, []() {});
            }

            auto bulk_end = domain.num_bulk_points + domain.num_layer_points;

            return sclx::execute_kernel([=](sclx::kernel_handler& handler
                                        ) mutable {
                sclx::local_array<value_type, 2> local_points(
                    handler,
                    {dimensions,
                     sclx::cuda::traits::kernel::default_block_shape[0]}
                );

                handler.launch(
                    sclx::md_range_t<1>{source_terms.shape()},
                    source_terms,
                    [=] __device__(
                        const sclx::md_index_t<1>& idx,
                        const sclx::kernel_info<>& info
                    ) mutable {
                        if (idx[0] >= bulk_end) {
                            return;
                        }
                        for (int i = 0; i < dimensions; ++i) {
                            local_points(i, info.local_thread_linear_id())
                                = points(i, idx[0]);
                        }
                        if (source_region.contains(
                                &local_points(0, info.local_thread_linear_id())
                            )) {
                            source_terms(idx[0]) = audio_sample;
                        }
                    }
                );
            });
        }

        const size_t& sample_rate() const { return sample_rate_; }

        value_type audio_length() const {
            return static_cast<value_type>(audio_samples_.elements())
                 / static_cast<value_type>(sample_rate_);
        }

        size_t sample_count() const { return audio_samples_.elements(); }

      private:
        sclx::array<value_type, 1> audio_samples_;
        value_type amplitude_;
        region_type source_region_;
        size_t sample_rate_;
        size_t frame_offset_;
        bool loop_audio_;
        bool has_finished_ = false;
        std::shared_ptr<path_t> path_;
        bool odd_sim_frame_ = false;
    };

    template<class SourceRegion>
    class sine_wav_density_source : public density_source_t {
      public:
        using point_t     = naga::point_t<value_type, dimensions>;
        using region_type = SourceRegion;

        sine_wav_density_source(
            region_type source_region,
            const value_type& amplitude,
            const value_type& pulse_width,
            const value_type& speed_of_sound,
            std::shared_ptr<path_t> path
            = zero_path_t::create(naga::point_t<value_type, dimensions>{}),
            const value_type& periods         = 0.0,
            const value_type& time_multiplier = 1.0
        )
            : amplitude_(amplitude),
              time_multiplier_(time_multiplier),
              source_region_(std::move(source_region)),
              periods_(periods),
              path_(std::move(path)) {
            frequency_ = speed_of_sound / pulse_width;
        }

        value_type frequency() const { return frequency_; }

        auto get_amplitude_at_time(const value_type& time) const {
            // auto ramp_up_time = 5.0 / frequency_;
            // return amplitude_
            //      * naga::math::sin(
            //            2 * naga::math::pi<value_type> * frequency_ * time
            //      ) * (time < ramp_up_time ? naga::math::loopless::pow<2>(
            //                                     time / ramp_up_time)
            //                                     : 1.0);
            return amplitude_
                 * naga::math::sin(
                       2 * naga::math::pi<value_type> * frequency_ * time
                 );
        }

        std::future<void> add_density_source(
            const simulation_domain_t& domain,
            const problem_parameters_t& params,
            const solution_t& solution,
            const value_type& time,
            sclx::array<value_type, 1>& source_terms
        ) final {
            value_type scaled_time = time * time_multiplier_;
            value_type radians
                = 2 * naga::math::pi<value_type> * frequency_ * scaled_time;

            if (radians >= 2 * naga::math::pi<value_type> * periods_
                && periods_ != 0.0) {
                return std::async(std::launch::deferred, []() {});
            }

            const auto& density = solution.macroscopic_values.fluid_density;
            const auto& nominal_density = params.nominal_density;
            const auto& points          = domain.points;
            auto source_region          = source_region_;
            source_region.shift_region((*path_)(scaled_time));

            auto perturbation = get_amplitude_at_time(scaled_time);

            auto bulk_end = domain.num_bulk_points + domain.num_layer_points;

            return sclx::execute_kernel([=](sclx::kernel_handler& handler
                                        ) mutable {
                sclx::local_array<value_type, 2> local_points(
                    handler,
                    {dimensions,
                     sclx::cuda::traits::kernel::default_block_shape[0]}
                );

                handler.launch(
                    sclx::md_range_t<1>{source_terms.shape()},
                    source_terms,
                    [=] __device__(
                        const sclx::md_index_t<1>& idx,
                        const sclx::kernel_info<>& info
                    ) mutable {
                        if (idx[0] >= bulk_end) {
                            return;
                        }
                        for (int i = 0; i < dimensions; ++i) {
                            local_points(i, info.local_thread_linear_id())
                                = points(i, idx[0]);
                        }
                        if (source_region.contains(
                                &local_points(0, info.local_thread_linear_id())
                            )) {
                            source_terms(idx[0]) = perturbation;
                        }
                    }
                );
            });
        }

      private:
        value_type amplitude_;
        value_type time_multiplier_;
        value_type frequency_;
        value_type periods_;
        region_type source_region_;
        std::shared_ptr<path_t> path_;
    };

    class pulse_density_source : public sine_wav_density_source<region_t> {
      public:
        using point_t = typename sine_wav_density_source<region_t>::point_t;

        pulse_density_source(
            const value_type& amplitude,
            const value_type& pulse_width,
            const value_type& speed_of_sound,
            const value_type& source_radius,
            const point_t& source_center,
            const value_type& time_multiplier = 1.0
        )
            : sine_wav_density_source<region_t>(
                  region_t{source_radius, source_center},
                  amplitude,
                  pulse_width,
                  speed_of_sound,
                  zero_path_t::create(naga::point_t<value_type, dimensions>{}),
                  1.0,
                  time_multiplier
              ) {}
    };
};
