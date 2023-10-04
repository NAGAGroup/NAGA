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

#include "nonlocal_lbm_utils.cuh"
#include "utils.hpp"

using value_type = float;

using lattice_t    = naga::fluids::nonlocal_lbm::d3q27_lattice<value_type>;
using sim_engine_t = problem_traits<lattice_t>::sim_engine_t;
using simulation_domain_t = problem_traits<lattice_t>::simulation_domain_t;
using node_provider_t     = problem_traits<lattice_t>::node_provider_t;
using audio_source_t      = problem_traits<lattice_t>::pulse_density_source;

// The time scaling done after the user-defined simulation parameters
// are to ensure that the desired audio frequency can be resolved.
//
// There are three space-time coordinate systems to be aware of:
//      - Simulation space: The space in which the simulation is performed.
//      Note that the lattice boltzmann method also has its own internal
//      non-dimensionalized coordinate system, but this is not relevant
//      here. All returned macroscopic values, point coordinates and time values
//      by the simulation API are in the simulation space. Same goes for
//      the parameters and domain passed to the simulation API.
//
//      - Audio space: The space in which the audio source is defined.
//      Spatially, this is the same as the simulation space. However, the time
//      coordinate is scaled by the time multiplier, which is less than one.
//      The time scaling effectively changes the speed of sound in this
//      coordinate system such that the wavelength at the maximum desired
//      frequency can be resolved.
//
//      - Physical space: The space in which the resulting simulation would
//      exist given a real-world speed of sound of 300 m/s. Time axis
//      is the same as in the audio space. The spatial coordinates are scaled
//      such that the speed of sound is 300 m/s.
//
//      - Visualization space: The space in which the simulation is visualized.
//      This has the same spatial scaling as the simulation space, but the
//      time axis is scaled to the user's desired visualization speed.
//
// Given the above, the behavior of the user-defined parameters is as follows:
//      Simulation Parameters:
//          - nodal_spacing: The spacing between lattice nodes in simulation
//          space
//          - fluid_viscosity: The viscosity of the fluid in simulation space.
//          Since our application is in acoustic propagation in air, we deal
//          with only near-zero viscosity, the user need not worry about how
//          this scales to the real-world value. Instead, this is provided in
//          case a non-zero value is needed for stability.
//          - fluid_density: The density of the fluid in simulation space. This
//          value is not affected by the transformations between the different
//          coordinate systems.
//          - characteristic_velocity: The characteristic velocity of the fluid
//          in simulation space. This value, along with the
//          lattice_characteristic_velocity, implicitly defines the speed of
//          sound in simulation space. Increasing this value scales the speed
//          linearly. Both characteristic velocities are provided as
//          the stable values for each are not yet known.
//          - lattice_characteristic_velocity: The characteristic velocity of
//          the fluid in the lattice coordinate system. This value, along with
//          the characteristic_velocity, implicitly defines the speed of sound
//          in simulation space. Increasing this value inversely scales the
//          speed of sound. Both characteristic velocities are provided as
//          the stable values for each are not yet known.
//          - characteristic_length: The characteristic length of the simulation
//          in simulation space. This value is determined by the domain size
//          loaded from the mesh files.
//
//     Real-World Parameters:
//          - desired_characteristic_length: The desired characteristic length
//          of the simulation in physical space. This value is used to adjust
//          the scaling of simulation time to audio time. If this value is too
//          high such that the required frequency cannot be resolved, it
//          will be lowered to the maximum possible value. If zero, the
//          maximum possible value will be used.
//
//     Audio Parameters:
//          - node_resolution: The minimum number of nodes that
//          resolved wavelengths should span.
//          - max_wav_frequency: The maximum frequency that the user wishes to
//          be able to resolve in the audio.
//          - audio_amplitude: The amplitude of the audio source, where the
//          amplitude has units of fluid_density.
//          - frame_offset: The number of frames to offset the audio source by.
//
//     Visualization Parameters:
//         - simulation_length: the length of time, in audio space, that the
//         simulation will be run for.
//         - visualization_length: the length of time, in visualization space,
//         that the simulation will be visualized for.
//         - visualization_fps: the frame rate at which the simulation will be
//         visualized.

int main() {
    // user-defined parameters
    // outer domain obj file and absorption parameters
    auto obj_resources_dir
        = get_resources_dir() / "lbm_example_domains" / "ball_in_cube";
    sclx::filesystem::path domain_obj       = obj_resources_dir / "cube.obj";
    double outer_absorption_layer_thickness = 0.1;
    double outer_absorption_coefficient     = 0.1;
    //    double outer_absorption_layer_thickness = 0.0;
    //    double outer_absorption_coefficient     = 0.0;

    // immersed boundary obj files and absorption parameters
    //    std::vector<sclx::filesystem::path> immersed_boundary_objs
    //        = {obj_resources_dir / "ball.obj"};
    //    std::vector<double> immersed_absorption_layer_thicknesses = {0.08};
    //    std::vector<double> immersed_absorption_coefficients      = {0.01};
    std::vector<sclx::filesystem::path> immersed_boundary_objs = {};
    std::vector<double> immersed_absorption_layer_thicknesses  = {};
    std::vector<double> immersed_absorption_coefficients       = {};

    // raw simulation parameters
    value_type nodal_spacing                   = 0.04;
    value_type fluid_viscosity                 = 1e-5f;
    value_type fluid_density                   = 1.0;
    value_type characteristic_velocity         = 0.4;
    value_type lattice_characteristic_velocity = 0.2;
    value_type characteristic_length           = 2.;

    // audio source parameters
    uint node_resolution       = 20;
    value_type audio_amplitude = 5e-4;
    value_type source_radius   = 0.06;

    // simulation-space to real-space conversion, zero maximum allowed
    value_type desired_characteristic_length = 0.;

    // visualization parameters
    value_type simulation_length    = 10.;
    value_type visualization_length = 10.;
    value_type visualization_fps    = 60;

    // results directory
    static auto results_path = get_examples_results_dir() / "pulse_3d";

    // observer types
    bool enable_vtk_observer = true;

    // ----------------- Simulation Code Below, Don't Touch ----------------- //

    value_type speed_of_sound = sim_engine_t::speed_of_sound(
        characteristic_velocity,
        lattice_characteristic_velocity
    );
    value_type pulse_frequency
        = speed_of_sound / nodal_spacing / node_resolution;
    value_type min_allowed_wavelength = nodal_spacing * node_resolution;
    value_type min_wavelength         = speed_of_sound / pulse_frequency;
    value_type time_multiplier        = 1.0;
    if (min_wavelength < min_allowed_wavelength) {
        time_multiplier = min_wavelength / min_allowed_wavelength;
    }
    value_type new_speed_of_sound = speed_of_sound / time_multiplier;

    value_type characteristic_length_at_300ms
        = characteristic_length * 300.f / new_speed_of_sound;

    if (desired_characteristic_length != 0.) {
        value_type requested_speed_of_sound = characteristic_length_at_300ms
                                            / desired_characteristic_length
                                            * new_speed_of_sound;
        value_type requested_time_multiplier
            = new_speed_of_sound / requested_speed_of_sound * time_multiplier;
        characteristic_length_at_300ms
            = characteristic_length * 300.f / requested_speed_of_sound;
        if (requested_speed_of_sound < new_speed_of_sound) {
            characteristic_length_at_300ms
                = characteristic_length * 300.f / new_speed_of_sound;
            std::cout
                << "Warning: Requested characteristic length is too small "
                   "to resolve the requested frequency content. \n"
                << "    The simulation will run with a scaled characteristic "
                   "length of "
                << characteristic_length_at_300ms << " m. \n\n";
        } else {
            time_multiplier = requested_time_multiplier;
        }
    }

    // this ensures our time multiplier is 1 / some_integer
    time_multiplier    = 1.f / std::ceil(1.f / time_multiplier);
    new_speed_of_sound = speed_of_sound / time_multiplier;
    characteristic_length_at_300ms
        = characteristic_length * 300.f / new_speed_of_sound;

    audio_source_t source{
        audio_amplitude,
        pulse_frequency * speed_of_sound,
        speed_of_sound,
        source_radius,
        audio_source_t::point_t{},
        time_multiplier};

    // this ensures that the time step of the simulation will be stable
    // and is an integer multiple of the audio source sample rate
    value_type unscaled_time_step = 0.5f * nodal_spacing * nodal_spacing;
    if (unscaled_time_step * speed_of_sound > 0.1f * nodal_spacing) {
        unscaled_time_step = 0.1f * nodal_spacing / speed_of_sound;
    }

    // need to adjust min wavelength for new parameters
    value_type max_resolved_frequency
        = new_speed_of_sound / min_allowed_wavelength;

    std::cout << "Simulation parameters scaled such that the speed of sound is "
                 "300 and \n"
                 "minimum wavelength is at least ~" << node_resolution << " nodes\n";
    std::cout << "    Max resolved frequency: " << max_resolved_frequency
              << " Hz\n";
    std::cout << "    Speed of sound: " << 300 << " m/s\n";
    std::cout << "    Nodal spacing: "
              << nodal_spacing * 300.f / new_speed_of_sound << " m\n";
    std::cout << "    Time step: " << unscaled_time_step * time_multiplier
              << " s\n";
    std::cout << "    Characteristic length: " << characteristic_length_at_300ms
              << " m\n\n";

    sim_engine_t engine;
    engine.set_problem_parameters(
        fluid_viscosity,
        fluid_density,
        unscaled_time_step,
        characteristic_length,
        characteristic_velocity,
        lattice_characteristic_velocity
    );
    engine.register_density_source(source);

    node_provider_t node_provider(
        nodal_spacing,
        domain_obj,
        immersed_boundary_objs,
        outer_absorption_layer_thickness,
        outer_absorption_coefficient,
        immersed_absorption_layer_thicknesses,
        immersed_absorption_coefficients
    );

    engine.init_domain(node_provider);
    const auto& domain = engine.domain();

    int sim_frame = 0;
    std::chrono::milliseconds total_time{0};
    value_type visualization_time_multiplier
        = visualization_length / simulation_length;
    async_time_printer printer;

    sclx::filesystem::remove_all(results_path);
    sclx::filesystem::create_directories(results_path);

    if (visualization_fps == 0) {
        visualization_fps = 1.f / engine.problem_parameters().time_step
                          / time_multiplier / visualization_time_multiplier;
    }
    naga::fluids::nonlocal_lbm::vtk_observer<lattice_t> viz_observer(
        results_path,
        time_multiplier * visualization_time_multiplier,
        visualization_fps
    );
    if (enable_vtk_observer) {
        engine.attach_observer(viz_observer);
    }

    while (engine.time() < simulation_length / time_multiplier) {
        printer.print_next_value(engine.time() * time_multiplier);

        auto start = std::chrono::high_resolution_clock::now();
        engine.step_forward();
        auto end = std::chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        );

        ++sim_frame;
    }

    std::cout << "Average time per frame: " << total_time.count() / sim_frame
              << "ms\n";
    std::cout << "Problem size: " << domain.points.shape()[1] << "\n";

    return 0;
}