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

namespace naga::fluids::nonlocal_lbm::detail {

template<class Lattice>
class pml_div_Q1_field_map {
  public:
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;

    using point_type = point_t<value_type, dimensions>;
};

template<class Lattice>
class partial_pml_subtask {
  public:
    using value_type = typename lattice_traits<Lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<Lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<Lattice>::size;
};

// ------------------ D2Q9 Lattice Specialization ------------------

template<class T>
class pml_div_Q1_field_map<d2q9_lattice<T>> {
  public:
    using lattice    = d2q9_lattice<T>;
    using value_type = typename lattice_traits<lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<lattice>::size;

    using point_type = point_t<value_type, dimensions>;

    __host__ __device__ pml_div_Q1_field_map(
        const value_type* c0,
        const sclx::array<const value_type, 1>& absorption_coeff,
        const sclx::array<const value_type, 1>& Q1,
        size_t pml_start_index,
        size_t pml_end_index
    )
        : absorption_coeff(absorption_coeff),
          Q1(Q1),
          pml_start_index(pml_start_index),
          pml_end_index(pml_end_index) {

        for (uint d = 0; d < dimensions; d++) {
            this->c0[d] = c0[d];
        }
    }

    __host__ __device__ point_type operator[](sclx::index_t i) const {
        point_type c;
        if (i < pml_start_index || i >= pml_end_index) {
            for (uint d = 0; d < dimensions; d++) {
                c[d] = 0.f;
            }
        } else {
            size_t pml_index = i - pml_start_index;
            value_type sigma = absorption_coeff[pml_index];
            for (uint d = 0; d < dimensions; d++) {
                c[d] = -c0[d] * sigma * Q1[pml_index];
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
    value_type c0[dimensions];
    sclx::array<const value_type, 1> absorption_coeff;
    sclx::array<const value_type, 1> Q1;
    size_t pml_start_index;
    size_t pml_end_index;
};

template<class T>
class partial_pml_subtask<d2q9_lattice<T>> {
  public:
    using lattice    = d2q9_lattice<T>;
    using value_type = typename lattice_traits<lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<lattice>::size;

    partial_pml_subtask(
        const simulation_engine<lattice>& engine,
        sclx::kernel_handler& handler,
        const sclx::array_list<value_type, 1, lattice_size>& lattice_Q1_values
    ) {
        params_local_ = sclx::local_array<parameters, 1>(handler, {1});
        params_       = sclx::detail::make_unified_ptr(parameters{});
        *params_      = parameters(engine, handler, lattice_Q1_values);
    }

    __device__ void operator()(
        const sclx::md_index_t<1>& idx,
        const sclx::kernel_info<>& info
    ) {
        auto& params = params_local_[0];
// the following if/else macro prevents linting errors in IDEs
// since the return type is different for host and device
#ifdef __CUDA_ARCH__
        sclx::kernel_handler& handler = params_->handler;
#else
        sclx::kernel_handler handler;
#endif
        if (info.local_thread_linear_id() == 0) {
            params = *params_;
            for (int i = 0; i < dimensions * lattice_size; ++i) {
                params.lattice_velocities(i % dimensions, i / dimensions)
                    = lattice_interface<lattice>::lattice_velocities()
                          .vals[i / dimensions][i % dimensions];

                if (i % dimensions == 0) {
                    params.lattice_weights(i / dimensions)
                        = lattice_interface<lattice>::lattice_weights()
                              .vals[i / dimensions];
                }
            }
        }
        handler.syncthreads();

        value_type unitless_density
            = params.fluid_density[idx] / params.density_scale;
        value_type unitless_velocity[dimensions];
        for (uint d = 0; d < dimensions; ++d) {
            unitless_velocity[d]
                = params.fluid_velocity(d, idx[0]) / params.velocity_scale;
        }

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            value_type f_tilde_eq = compute_equilibrium_distribution<lattice>(
                                        unitless_density,
                                        unitless_velocity,
                                        &params.lattice_velocities(0, alpha),
                                        params.lattice_weights(alpha)
                                    )
                                  - params.lattice_weights(alpha);

            // note that lattice_Q1_values_ is also used to store the
            // previous value of f_tilde_eq
            const value_type& f_tilde_eq_prev
                = params.lattice_Q1_values[alpha][idx]
                - params.lattice_weights(alpha);

            value_type Q_value
                = (f_tilde_eq + f_tilde_eq_prev) * params.lattice_time_step / 2.f;

            params.lattice_Q1_values[alpha][idx] = Q_value;

            if (idx[0] < params.absorption_layer_start
                || idx[0] >= params.absorption_layer_end) {
                continue;
            }

            const value_type& sigma
                = params.absorption_coefficients
                      [idx[0] - params.absorption_layer_start];

            params.lattice_distributions[alpha][idx[0]]
                -= params.lattice_time_step * sigma
                 * (2.f * f_tilde_eq + sigma * Q_value);
        }
    }

    sclx::array_list<value_type, 1, 2 * lattice_size> result() const {
        sclx::array<value_type, 1> result_arrays_raw[2 * lattice_size]{};

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            result_arrays_raw[alpha] = params_->lattice_distributions[alpha];
            result_arrays_raw[alpha + lattice_size]
                = params_->lattice_Q1_values[alpha];
        }

        sclx::array_list<value_type, 1, 2 * lattice_size> result_arrays(
            result_arrays_raw
        );
        return result_arrays;
    }

    struct parameters {
        parameters() = default;

        parameters(
            const simulation_engine<lattice>& engine,
            sclx::kernel_handler& handler,
            const sclx::array_list<value_type, 1, lattice_size>&
                lattice_Q1_values
        ) {
            absorption_layer_start = engine.domain_.num_bulk_points;
            absorption_layer_end   = engine.domain_.num_bulk_points
                                 + engine.domain_.num_layer_points;

            lattice_distributions
                = sclx::array_list<value_type, 1, lattice_size>(
                    engine.solution_.lattice_distributions
                );

            this->lattice_Q1_values = lattice_Q1_values;

            absorption_coefficients = engine.domain_.layer_absorption;
            fluid_density  = engine.solution_.macroscopic_values.fluid_density;
            fluid_velocity = engine.solution_.macroscopic_values.fluid_velocity;

            lattice_velocities = sclx::local_array<value_type, 2>(
                handler,
                {dimensions, lattice_size}
            );
            lattice_weights
                = sclx::local_array<value_type, 1>(handler, {lattice_size});

            density_scale  = engine.parameters_.nominal_density;
            velocity_scale = engine.parameters_.speed_of_sound / lattice_traits<
                                 lattice>::lattice_speed_of_sound;
            lattice_time_step     = engine.parameters_.lattice_time_step;

            this->handler = handler;
        }

        sclx::kernel_handler handler;

        sclx::index_t absorption_layer_start{};
        sclx::index_t absorption_layer_end{};

        sclx::array_list<value_type, 1, lattice_size> lattice_distributions;
        sclx::array_list<value_type, 1, lattice_size> lattice_Q1_values;

        sclx::array<const value_type, 1> absorption_coefficients;
        sclx::array<const value_type, 1> fluid_density;
        sclx::array<const value_type, 2> fluid_velocity;

        sclx::local_array<value_type, 2> lattice_velocities;
        sclx::local_array<value_type, 1> lattice_weights;
        value_type density_scale;
        value_type velocity_scale;
        value_type lattice_time_step;
    };

  private:
    sclx::detail::unified_ptr<parameters> params_;
    sclx::local_array<parameters, 1> params_local_;
};

template<class T>
struct subtask_factory<partial_pml_subtask<d2q9_lattice<T>>> {
    using lattice = d2q9_lattice<T>;
    static partial_pml_subtask<lattice> create(
        const simulation_engine<lattice>& engine,
        sclx::kernel_handler& handler,
        const sclx::array_list<typename lattice::value_type, 1, lattice::size>&
            lattice_Q1_values
    ) {
        return {engine, handler, lattice_Q1_values};
    }
};

template<class T>
class pml_absorption_operator<d2q9_lattice<T>> {
  public:
    using lattice    = d2q9_lattice<T>;
    using value_type = typename lattice_traits<lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<lattice>::size;

    pml_absorption_operator() = default;

    auto get_Q1_values() const {
        return lattice_Q1_values_;
    }

    explicit pml_absorption_operator(simulation_engine<lattice>* engine)
        : engine_(engine) {
        std::vector<sclx::array<value_type, 1>> lattice_Q1_values_raw;
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            sclx::array<value_type, 1> Q1_alpha{engine_->domain_.points.shape(
            )[1]};
            sclx::fill(
                Q1_alpha,
                lattice_interface<lattice>::lattice_weights().vals[alpha]
            );
            lattice_Q1_values_raw.push_back(Q1_alpha);
        }
        lattice_Q1_values_ = sclx::array_list<value_type, 1, lattice_size>(
            lattice_Q1_values_raw
        );
    }

    void apply() {
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            auto subtask
                = subtask_factory<partial_pml_subtask<lattice>>::create(
                    *engine_,
                    handler,
                    lattice_Q1_values_
                );

            handler.launch(
                sclx::md_range_t<1>{engine_->domain_.points.shape()[1]},
                subtask.result(),
                subtask
            );
        }).get();

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            const auto& Q1 = lattice_Q1_values_[alpha];

            pml_div_Q1_field_map<lattice> field_map{
                lattice_interface<lattice>::lattice_velocities().vals[alpha],
                Q1,
                engine_->domain_.layer_absorption,
                engine_->domain_.num_bulk_points,
                engine_->domain_.num_bulk_points
                    + engine_->domain_.num_layer_points
            };

            engine_->advection_operator_ptr_->divergence_op().apply(
                field_map,
                engine_->temporary_distributions_[alpha]
            );

            auto layer_begin = engine_->domain_.num_bulk_points;
            auto layer_end   = engine_->domain_.num_bulk_points
                           + engine_->domain_.num_layer_points;
            sclx::algorithm::elementwise_reduce(
                nonlocal_calculus::forward_euler<T>(engine_->parameters_.lattice_time_step),
                engine_->solution_
                    .lattice_distributions[alpha],  // .get_range({layer_begin},
                                                    // {layer_end}),
                engine_->solution_
                    .lattice_distributions[alpha],  // .get_range({layer_begin},
                                                    // {layer_end}),
                engine_->temporary_distributions_
                    [alpha]  // .get_range({layer_begin},
                             // {layer_end})
            );
        }

        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            auto subtask
                = subtask_factory<compute_equilibrium_subtask<lattice>>::create(
                    *engine_,
                    handler,
                    lattice_Q1_values_
                );
            handler.launch(
                sclx::md_range_t<1>{engine_->domain_.points.shape()[1]},
                subtask.result(),
                subtask
            );
        }).get();
    }

    template<class Archive>
    void save(Archive& ar) const {
        for (const auto& arr : lattice_Q1_values_) {
            sclx::serialize_array(ar, arr);
        }
    }

    template<class Archive>
    void load(Archive& ar) {
        for (auto& arr : lattice_Q1_values_) {
            sclx::deserialize_array(ar, arr);
        }
    }

  private:
    simulation_engine<lattice>* engine_;
    sclx::array_list<value_type, 1, lattice_size> lattice_Q1_values_;
};

// ------------------------ D3Q27 Specialization ------------------------------

template<class T>
class pml_div_Q1_field_map<d3q27_lattice<T>> {
  public:
    using lattice    = d3q27_lattice<T>;
    using value_type = typename lattice_traits<lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<lattice>::size;

    using point_type = point_t<value_type, dimensions>;

    __host__ __device__ pml_div_Q1_field_map(
        const value_type* c0,
        const sclx::array<const value_type, 1>& absorption_coeff,
        const sclx::array<const value_type, 1>& Q1,
        size_t pml_start_index,
        size_t pml_end_index
    )
        : absorption_coeff(absorption_coeff),
          Q1(Q1),
          pml_start_index(pml_start_index),
          pml_end_index(pml_end_index) {

        for (uint d = 0; d < dimensions; d++) {
            this->c0[d] = c0[d];
        }
    }

    __host__ __device__ point_type operator[](sclx::index_t i) const {
        point_type c{{0.f, 0.f, 0.f}};
        if (i < pml_start_index || i >= pml_end_index) {
            return c;
        } else {
            size_t pml_index        = i - pml_start_index;
            const value_type& sigma = absorption_coeff[pml_index];
            for (uint d = 0; d < dimensions; d++) {
                c[d] = -2.f * c0[d] * sigma * Q1[pml_index];
            }
        }

        return c;
    }

    __host__ __device__ point_type operator[](const sclx::md_index_t<1>& index
    ) const {
        return (*this)[index[0]];
    }

    __host__ __device__ size_t size() const {
        return Q1.elements();
    }

  private:
    value_type c0[dimensions];
    sclx::array<const value_type, 1> absorption_coeff;
    sclx::array<const value_type, 1> Q1;
    size_t pml_start_index;
    size_t pml_end_index;
};

template<class T>
class pml_div_Q2_field_map {
  public:
    using lattice    = d3q27_lattice<T>;
    using value_type = typename lattice_traits<lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<lattice>::size;

    using point_type = point_t<value_type, dimensions>;

    __host__ __device__ pml_div_Q2_field_map(
        const value_type* c0,
        const sclx::array<const value_type, 1>& absorption_coeff,
        const sclx::array<const value_type, 1>& Q2,
        size_t pml_start_index,
        size_t pml_end_index
    )
        : absorption_coeff(absorption_coeff),
          Q2(Q2),
          pml_start_index(pml_start_index),
          pml_end_index(pml_end_index) {

        for (uint d = 0; d < dimensions; d++) {
            this->c0[d] = c0[d];
        }
    }

    __host__ __device__ point_type operator[](sclx::index_t i) const {
        point_type c{{0.f, 0.f, 0.f}};
        if (i < pml_start_index || i >= pml_end_index) {
            return c;
        } else {
            size_t pml_index        = i - pml_start_index;
            const value_type& sigma = absorption_coeff[pml_index];
            for (uint d = 0; d < dimensions; d++) {
                c[d] = c0[d] * sigma * sigma * Q2[pml_index];
            }
        }

        return c;
    }

    __host__ __device__ point_type operator[](const sclx::md_index_t<1>& index
    ) const {
        return (*this)[index[0]];
    }

    __host__ __device__ size_t size() const {
        return Q2.elements();
    }

  private:
    value_type c0[dimensions];
    sclx::array<const value_type, 1> absorption_coeff;
    sclx::array<const value_type, 1> Q2;
    size_t pml_start_index;
    size_t pml_end_index;
};

template<class T>
class partial_pml_subtask<d3q27_lattice<T>> {
  public:
    using lattice    = d3q27_lattice<T>;
    using value_type = typename lattice_traits<lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<lattice>::size;

    partial_pml_subtask(
        const simulation_engine<lattice>& engine,
        sclx::kernel_handler& handler,
        const sclx::array_list<value_type, 1, lattice_size>& lattice_Q1_values,
        const sclx::array_list<value_type, 1, lattice_size>& lattice_Q2_values
    ) {
        params_local_ = sclx::local_array<parameters, 1>(handler, {1});
        params_       = sclx::detail::make_unified_ptr(parameters{});
        *params_
            = parameters(engine, handler, lattice_Q1_values, lattice_Q2_values);
    }

    __device__ void operator()(
        const sclx::md_index_t<1>& idx,
        const sclx::kernel_info<>& info
    ) {
        auto& params = params_local_[0];
// the following if/else macro prevents linting errors in IDEs
// since the return type is different for host and device
#ifdef __CUDA_ARCH__
        sclx::kernel_handler& handler = params_->handler;
#else
        sclx::kernel_handler handler;
#endif
        if (info.local_thread_linear_id() == 0) {
            params = *params_;
            for (int i = 0; i < dimensions * lattice_size; ++i) {
                params.lattice_velocities(i % dimensions, i / dimensions)
                    = lattice_interface<lattice>::lattice_velocities()
                          .vals[i / dimensions][i % dimensions];

                if (i % dimensions == 0) {
                    params.lattice_weights(i / dimensions)
                        = lattice_interface<lattice>::lattice_weights()
                              .vals[i / dimensions];
                }
            }
        }
        handler.syncthreads();

        value_type unitless_density
            = params.fluid_density[idx] / params.density_scale;
        value_type unitless_velocity[dimensions];
        for (uint d = 0; d < dimensions; ++d) {
            unitless_velocity[d]
                = params.fluid_velocity(d, idx[0]) / params.velocity_scale;
        }

        for (uint alpha = 0; alpha < lattice_size; ++alpha) {
            value_type f_tilde_eq = compute_equilibrium_distribution<lattice>(
                                        unitless_density,
                                        unitless_velocity,
                                        &params.lattice_velocities(0, alpha),
                                        params.lattice_weights(alpha)
                                    )
                                  - params.lattice_weights(alpha);

            // note that lattice_Q1_values_ is also used to store the
            // previous value of f_tilde_eq
            const value_type& f_tilde_eq_prev
                = params.lattice_Q1_values[alpha][idx]
                - params.lattice_weights(alpha);

            value_type Q1_value
                = (f_tilde_eq + f_tilde_eq_prev) * params.lattice_time_step / 2.f;

            params.lattice_Q1_values[alpha][idx] = Q1_value;

            const value_type& Q1_value_prev
                = params.lattice_Q2_values[alpha][idx];

            value_type Q2_value
                = (Q1_value + Q1_value_prev) * params.lattice_time_step / 2.f;

            params.lattice_Q2_values[alpha][idx] = Q2_value;

            if (idx[0] < params.absorption_layer_start
                || idx[0] >= params.absorption_layer_end) {
                continue;
            }

            const value_type& sigma
                = params.absorption_coefficients
                      [idx[0] - params.absorption_layer_start];

            using namespace math::loopless;

            params.lattice_distributions[alpha][idx[0]]
                -= sigma * params.lattice_time_step
                 * (3.f * f_tilde_eq + 3.f * sigma * Q1_value
                    + pow<2>(sigma) * Q2_value);
        }
    }

    sclx::array_list<value_type, 1, 3 * lattice_size> result() const {
        sclx::array<value_type, 1> result_arrays_raw[3 * lattice_size]{};

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            result_arrays_raw[alpha] = params_->lattice_distributions[alpha];
            result_arrays_raw[alpha + lattice_size]
                = params_->lattice_Q1_values[alpha];
            result_arrays_raw[alpha + 2 * lattice_size]
                = params_->lattice_Q2_values[alpha];
        }

        sclx::array_list<value_type, 1, 3 * lattice_size> result_arrays(
            result_arrays_raw
        );
        return result_arrays;
    }

    struct parameters {
        parameters() = default;

        parameters(
            const simulation_engine<lattice>& engine,
            sclx::kernel_handler& handler,
            const sclx::array_list<value_type, 1, lattice_size>&
                lattice_Q1_values,
            const sclx::array_list<value_type, 1, lattice_size>&
                lattice_Q2_values
        ) {
            absorption_layer_start = engine.domain_.num_bulk_points;
            absorption_layer_end   = engine.domain_.num_bulk_points
                                 + engine.domain_.num_layer_points;

            lattice_distributions
                = sclx::array_list<value_type, 1, lattice_size>(
                    engine.solution_.lattice_distributions
                );

            this->lattice_Q1_values = lattice_Q1_values;
            this->lattice_Q2_values = lattice_Q2_values;

            absorption_coefficients = engine.domain_.layer_absorption;
            fluid_density  = engine.solution_.macroscopic_values.fluid_density;
            fluid_velocity = engine.solution_.macroscopic_values.fluid_velocity;

            lattice_velocities = sclx::local_array<value_type, 2>(
                handler,
                {dimensions, lattice_size}
            );
            lattice_weights
                = sclx::local_array<value_type, 1>(handler, {lattice_size});

            density_scale  = engine.parameters_.nominal_density;
            velocity_scale = engine.parameters_.speed_of_sound / lattice_traits<
                lattice>::lattice_speed_of_sound;
            lattice_time_step     = engine.parameters_.lattice_time_step;

            this->handler = handler;
        }

        sclx::kernel_handler handler;

        sclx::index_t absorption_layer_start{};
        sclx::index_t absorption_layer_end{};

        sclx::array_list<value_type, 1, lattice_size> lattice_distributions;
        sclx::array_list<value_type, 1, lattice_size> lattice_Q1_values;
        sclx::array_list<value_type, 1, lattice_size> lattice_Q2_values;

        sclx::array<const value_type, 1> absorption_coefficients;
        sclx::array<const value_type, 1> fluid_density;
        sclx::array<const value_type, 2> fluid_velocity;

        sclx::local_array<value_type, 2> lattice_velocities;
        sclx::local_array<value_type, 1> lattice_weights;
        value_type density_scale;
        value_type velocity_scale;
        value_type lattice_time_step;
    };

  private:
    sclx::detail::unified_ptr<parameters> params_;
    sclx::local_array<parameters, 1> params_local_;
};

template<class T>
struct subtask_factory<partial_pml_subtask<d3q27_lattice<T>>> {
    using lattice = d3q27_lattice<T>;
    static partial_pml_subtask<lattice> create(
        const simulation_engine<lattice>& engine,
        sclx::kernel_handler& handler,
        const sclx::array_list<typename lattice::value_type, 1, lattice::size>&
            lattice_Q1_values,
        const sclx::array_list<typename lattice::value_type, 1, lattice::size>&
            lattice_Q2_values
    ) {
        return {engine, handler, lattice_Q1_values, lattice_Q2_values};
    }
};

template<class T>
class pml_absorption_operator<d3q27_lattice<T>> {
  public:
    using lattice    = d3q27_lattice<T>;
    using value_type = typename lattice_traits<lattice>::value_type;
    static constexpr uint dimensions   = lattice_traits<lattice>::dimensions;
    static constexpr uint lattice_size = lattice_traits<lattice>::size;

    pml_absorption_operator() = default;

    explicit pml_absorption_operator(simulation_engine<lattice>* engine)
        : engine_(engine) {
        std::vector<sclx::array<value_type, 1>> lattice_Q1_values_raw;
        std::vector<sclx::array<value_type, 1>> lattice_Q2_values_raw;
        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            sclx::array<value_type, 1> Q1_alpha{engine_->domain_.points.shape(
            )[1]};
            sclx::fill(
                Q1_alpha,
                lattice_interface<lattice>::lattice_weights().vals[alpha]
            );
            lattice_Q1_values_raw.push_back(Q1_alpha);

            sclx::array<value_type, 1> Q2_alpha{engine_->domain_.points.shape(
            )[1]};
            sclx::fill(Q2_alpha, value_type{0});
            lattice_Q2_values_raw.push_back(Q2_alpha);
        }
        lattice_Q1_values_ = sclx::array_list<value_type, 1, lattice_size>(
            lattice_Q1_values_raw
        );
        lattice_Q2_values_ = sclx::array_list<value_type, 1, lattice_size>(
            lattice_Q2_values_raw
        );
    }

    void apply() {
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            auto subtask
                = subtask_factory<partial_pml_subtask<lattice>>::create(
                    *engine_,
                    handler,
                    lattice_Q1_values_,
                    lattice_Q2_values_
                );

            handler.launch(
                sclx::md_range_t<1>{engine_->domain_.points.shape()[1]},
                subtask.result(),
                subtask
            );
        }).get();

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            const auto& Q1 = lattice_Q1_values_[alpha];

            pml_div_Q1_field_map<lattice> field_map_Q1{
                lattice_interface<lattice>::lattice_velocities().vals[alpha],
                engine_->domain_.layer_absorption,
                Q1,
                engine_->domain_.num_bulk_points,
                engine_->domain_.num_bulk_points
                    + engine_->domain_.num_layer_points
            };

            engine_->advection_operator_ptr_->divergence_op().apply(
                field_map_Q1,
                engine_->temporary_distributions_[alpha]
            );

            auto layer_begin = engine_->domain_.num_bulk_points;
            auto layer_end   = engine_->domain_.num_bulk_points
                           + engine_->domain_.num_layer_points;
            sclx::algorithm::elementwise_reduce(
                nonlocal_calculus::forward_euler<T>(engine_->parameters_.lattice_time_step),
                engine_->solution_
                    .lattice_distributions[alpha],  // .get_range({layer_begin},
                                                    // {layer_end}),
                engine_->solution_
                    .lattice_distributions[alpha],  // .get_range({layer_begin},
                                                    // {layer_end}),
                engine_->temporary_distributions_
                    [alpha]  // .get_range({layer_begin},
                             // {layer_end})
            );

            const auto& Q2 = lattice_Q2_values_[alpha];

            pml_div_Q2_field_map<value_type> field_map_Q2{
                lattice_interface<lattice>::lattice_velocities().vals[alpha],
                engine_->domain_.layer_absorption,
                Q2,
                engine_->domain_.num_bulk_points,
                engine_->domain_.num_bulk_points
                    + engine_->domain_.num_layer_points
            };

            engine_->advection_operator_ptr_->divergence_op().apply(
                field_map_Q2,
                engine_->temporary_distributions_[alpha]
            );

            sclx::algorithm::elementwise_reduce(
                nonlocal_calculus::forward_euler<T>(engine_->parameters_.lattice_time_step),
                engine_->solution_
                    .lattice_distributions[alpha],  // .get_range({layer_begin},
                                                    // {layer_end}),
                engine_->solution_
                    .lattice_distributions[alpha],  // .get_range({layer_begin},
                                                    // {layer_end}),
                engine_->temporary_distributions_
                    [alpha]  // .get_range({layer_begin},
                             // {layer_end})
            );
        }

        for (int alpha = 0; alpha < lattice_size; ++alpha) {
            auto& Q1 = lattice_Q1_values_[alpha];

            auto& Q2 = lattice_Q2_values_[alpha];
            sclx::assign_array(Q1, Q2).get();
        }

        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            auto subtask
                = subtask_factory<compute_equilibrium_subtask<lattice>>::create(
                    *engine_,
                    handler,
                    lattice_Q1_values_
                );
            handler.launch(
                sclx::md_range_t<1>{engine_->domain_.points.shape()[1]},
                subtask.result(),
                subtask
            );
        }).get();
    }

    auto get_Q1_values() const {
        return lattice_Q1_values_;
    }

    auto get_Q2_values() const {
        return lattice_Q2_values_;
    }

    template<class Archive>
    void save(Archive& ar) const {
        for (const auto& arr : lattice_Q1_values_) {
            sclx::serialize_array(ar, arr);
        }
        for (auto& arr : lattice_Q2_values_) {
            sclx::serialize_array(ar, arr);
        }
    }

    template<class Archive>
    void load(Archive& ar) {
        for (auto& arr : lattice_Q1_values_) {
            sclx::deserialize_array(ar, arr);
        }
        for (auto& arr : lattice_Q2_values_) {
            sclx::deserialize_array(ar, arr);
        }
    }

  private:
    simulation_engine<lattice>* engine_;
    sclx::array_list<value_type, 1, lattice_size> lattice_Q1_values_;
    sclx::array_list<value_type, 1, lattice_size> lattice_Q2_values_;
};

}  // namespace naga::fluids::nonlocal_lbm::detail