#include <naga/fluids/nonlocal_lattice_boltzmann.cuh>
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
    using simulation_domain_t  = typename sim_engine_t::simulation_domain_t;
    using problem_parameters_t = typename sim_engine_t::problem_parameters_t;
    using solution_t           = typename sim_engine_t::solution_t;
    using region_t
        = naga::regions::nd_rectangular_prism<value_type, dimensions>;

    using node_provider_t = naga::experimental::fluids::nonlocal_lbm::
        conforming_point_cloud_provider<lattice_type>;

    class path_t {
      public:
        using point_type = naga::point_t<value_type, dimensions>;
        virtual const point_type& operator()(const value_type& t) const = 0;
    };

    class zero_path_t : public path_t {
      public:
        using base       = path_t;
        using point_type = typename base::point_type;

        static std::shared_ptr<zero_path_t> create(const point_type& origin) {
            return std::shared_ptr<zero_path_t>(new zero_path_t(origin));
        }

        const point_type& operator()(const value_type&) const final {
            return origin_;
        }

      private:
        zero_path_t(point_type origin) : origin_(std::move(origin)) {}
        naga::point_t<value_type, dimensions> origin_;
    };

    class spherical_audio_source : public density_source_t {
      public:
        spherical_audio_source(
            const sclx::filesystem::path& wav_file,
            const value_type& amplitude,
            const value_type& source_radius,
            const value_type& time_multiplier = 1.0,
            const size_t& frame_offset        = 0,
            std::shared_ptr<path_t> path
            = zero_path_t::create(naga::point_t<value_type, dimensions>{})
        )
            : amplitude_(amplitude),
              source_radius_(source_radius),
              time_multiplier_(time_multiplier),
              frame_offset_(frame_offset) {
            AudioFile<value_type> audio_file(wav_file.string());
            if (!audio_file.isMono()) {
                sclx::throw_exception<std::runtime_error>(
                    "Audio file must be mono",
                    "spherical_audio_source::"
                );
            }
            sample_rate_ = audio_file.getSampleRate();

            audio_samples_ = sclx::array<value_type, 1>{
                static_cast<const size_t&>(audio_file.getNumSamplesPerChannel()
                )};

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
            region_t source_region{source_radius_, (*path_)(time)};

            auto lower_frame_number = std::floor(
                time * time_multiplier_ * static_cast<value_type>(sample_rate_)
            );
            auto upper_frame_number = std::ceil(
                time * time_multiplier_ * static_cast<value_type>(sample_rate_)
            );
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
            const auto& nominal_density = params.nondim_factors.density_scale;
            const auto& points          = domain.points;

            if (frame_number >= audio_samples.elements()) {
                if (!has_finished_) {
                    std::cout << "Audio source has finished at frame "
                              << frame_number << "\n\n";
                    has_finished_ = true;
                }
                return std::async(std::launch::deferred, []() {});
            }

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
                        for (int i = 0; i < dimensions; ++i) {
                            local_points(i, info.local_thread_linear_id())
                                = points(i, idx[0]);
                        }
                        auto audio_sample_upper
                            = amplitude * audio_samples(frame_number + 1);
                        auto audio_sample_lower
                            = amplitude * audio_samples(frame_number);
                        auto audio_sample = upper_weight * audio_sample_upper
                                          + lower_weight * audio_sample_lower;
                        if (source_region.contains(
                                &local_points(0, info.local_thread_linear_id())
                            )) {
                            auto distance
                                = naga::distance_functions::loopless::euclidean<
                                    3>{}(
                                    &local_points(
                                        0,
                                        info.local_thread_linear_id()
                                    ),
                                    source_region.center()
                                );

                            auto perturbation = audio_sample;
                            //                            * (1
                            //                               -
                            //                               naga::math::loopless::pow<1>(
                            //                                     distance /
                            //                                     source_region.radius()
                            //                                 ))
                            //                            * naga::math::exp(-naga::math::loopless::pow<2>(
                            //                                2 * distance /
                            //                                source_region.radius()
                            //                            ));

                            auto current_density = density(idx[0]);
                            perturbation += nominal_density - current_density;
                            source_terms(idx[0]) += perturbation;
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
        value_type source_radius_;
        value_type time_multiplier_;
        size_t sample_rate_;
        size_t frame_offset_;
        bool has_finished_ = false;
        std::shared_ptr<path_t> path_;
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
            const value_type& periods         = 0.0,
            const value_type& time_multiplier = 1.0
        )
            : amplitude_(amplitude),
              time_multiplier_(time_multiplier),
              source_region_(std::move(source_region)),
              periods_(periods) {
            frequency_ = speed_of_sound / pulse_width;
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

            auto perturbation = amplitude_ * naga::math::sin(radians);

            const auto& density = solution.macroscopic_values.fluid_density;
            const auto& nominal_density = params.nondim_factors.density_scale;
            const auto& points          = domain.points;
            const auto& source_region   = source_region_;

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
                        for (int i = 0; i < dimensions; ++i) {
                            local_points(i, info.local_thread_linear_id())
                                = points(i, idx[0]);
                        }
                        if (source_region.contains(
                                &local_points(0, info.local_thread_linear_id())
                            )) {
                            auto current_density = density(idx[0]);
                            perturbation += nominal_density - current_density;
                            source_terms(idx[0]) += perturbation;
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
                1.0,
                time_multiplier
            ) {}
    };
};
