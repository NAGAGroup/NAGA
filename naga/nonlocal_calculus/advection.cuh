
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

#include "divergence.cuh"
#include <scalix/algorithm/elementwise_reduce.cuh>
#include <scalix/algorithm/transform.cuh>
#include <scalix/assign_array.cuh>

#include <Eigen/Eigen>

namespace naga::nonlocal_calculus {

template<class T>
class forward_euler {
  public:
    template<class U, class V>
    __host__ __device__ T operator()(U&& f, V&& df_dt) const {
        return f + dt_ * df_dt;
    }

    explicit forward_euler(const T& dt) : dt_(dt) {}

  private:
    T dt_;
};

template<class T, uint Dimensions>
class constant_velocity_field {
  public:
    using point_type = point_t<T, Dimensions>;

    template<class Velocity>
    static constant_velocity_field create(Velocity&& v) {
        constant_velocity_field map;
        for (uint i = 0; i < Dimensions; ++i) {
            map.velocity_[i] = v[i];
        }
        return map;
    }

    __host__ __device__ point_type operator[](const uint& index
    ) const {
        point_type vector_value;
        for (uint i = 0; i < Dimensions; ++i) {
            vector_value[i] = velocity_[i];
        }
        return vector_value;
    }

  private:
    constant_velocity_field() = default;

    T velocity_[Dimensions];
};

template<class T, uint Dimensions>
class advection_operator {

    using matrix_type
        = naga::linalg::matrix<T, naga::linalg::storage_type::sparse_csr>;
    using vector_type
        = naga::linalg::vector<T, naga::linalg::storage_type::dense>;
    using mat_mult_type
        = naga::linalg::matrix_mult<matrix_type, vector_type, vector_type>;

  public:
    advection_operator() = default;

    friend class operator_builder<T, Dimensions>;

    template<class VelocityFieldMap>
    class divergence_field_map {
      public:
        using point_type = point_t<T, Dimensions>;
        friend class advection_operator;

        divergence_field_map(
            const VelocityFieldMap& velocity_field,
            const sclx::array<T, 1>& scalar_field,
            T centering_offset = T(0)
        )
            : velocity_field_(velocity_field),
              scalar_field_(scalar_field),
              centering_offset_(centering_offset) {}

        __host__ __device__ point_type operator[](const uint& index
        ) const {
            point_type velocity = velocity_field_[index];
            T scalar            = scalar_field_[index];
            for (uint i = 0; i < Dimensions; ++i) {
                velocity[i] *= -(scalar - centering_offset_);
            }

            return velocity;
        }

        __host__ __device__ point_type
        operator[](const sclx::md_index_t<1>& index) const {
            return (*this)[index[0]];
        }

        __host__ __device__ uint size() const {
            return scalar_field_.elements();
        }

      private:
        VelocityFieldMap velocity_field_;
        sclx::array<T, 1> scalar_field_;
        T centering_offset_;
    };

    static advection_operator create(const sclx::array<T, 2>& domain) {
        advection_operator op;
        op.divergence_op_
            = std::make_shared<divergence_operator<T, Dimensions>>(
                divergence_operator<T, Dimensions>::create(domain)
            );
        op.domain_size_ = domain.shape()[1];
        return op;
    }

    class advection_task {
      public:
        template<class FieldMap>
        advection_task(
            std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_,
            FieldMap field_map,
            sclx::array<T, 1> f0,
            sclx::array<T, 1> f,
            T dt,
            T centering_offset,
            sclx::array<uint, 1>* explicit_indices = nullptr,
            matrix_type* implicit_matrix             = nullptr,
            mat_mult_type* mat_mult                  = nullptr
        )
            : field_map_(std::make_shared<FieldMap>(std::move(field_map))),
              f0_(std::move(f0)),
              f_(std::move(f)),
              dt_(dt),
              centering_offset_(centering_offset),
              divergence_op_(divergence_op_),
              impl_(&advection_task::impl<FieldMap>),
              explicit_indices_(explicit_indices),
              implicit_matrix_(implicit_matrix),
              mat_mult(mat_mult) {}

        void operator()(sclx::array<T, 1> (&rk_df_dt_list)[4]) {
            sclx::array<T, 1> sliced_rk_df_dt_list[4];
            for (int i = 0; i < 4; ++i) {
                if (rk_df_dt_list[i].elements() == f0_.elements()) {
                    sliced_rk_df_dt_list[i] = rk_df_dt_list[i];
                } else if (rk_df_dt_list[i].elements() > f0_.elements()) {
                    sliced_rk_df_dt_list[i] = sclx::array<T, 1>{
                        f0_.shape(),
                        rk_df_dt_list[i].data()};
                    sliced_rk_df_dt_list[i].set_primary_devices();
                } else {
                    rk_df_dt_list[i]        = sclx::array<T, 1>{f0_.shape()};
                    sliced_rk_df_dt_list[i] = rk_df_dt_list[i];
                }
            }
            impl_(*this, sliced_rk_df_dt_list);
        }

        class index_generator {
          public:
            static constexpr uint range_rank
                = 1;  ///< Used to generate the thread grid with indices of type
                      ///< md_index_t<Rank> (required by execute_kernel).
            static constexpr uint index_rank
                = 1;  ///< Used to generate the write indices with indices of
                      ///< type md_index_t<Rank> (required by execute_kernel).

            index_generator(
                const sclx::shape_t<1>& target_shape,
                sclx::array<uint, 1>& explicit_indices
            )
                : generator_shape_(explicit_indices.shape()),
                  target_shape_(target_shape),
                  indices_(explicit_indices) {}

            __host__ __device__ sclx::md_index_t<index_rank>
            operator()(sclx::md_index_t<range_rank> index) const {
                return sclx::md_index_t<index_rank>{{indices_[index]}};
            }

            __host__ __device__ const sclx::md_range_t<range_rank>&
            range() const {
                return static_cast<const sclx::md_range_t<range_rank>&>(
                    generator_shape_
                );
            }

            __host__ __device__ const sclx::md_range_t<index_rank>&
            index_range() const {
                return static_cast<const sclx::md_range_t<index_rank>&>(
                    target_shape_
                );
            }

          private:
            sclx::shape_t<range_rank> generator_shape_;
            sclx::shape_t<index_rank> target_shape_;
            sclx::array<uint, range_rank> indices_;
        };

        template<class FieldMap>
        static void
        impl(advection_task& task, sclx::array<T, 1> (&rk_df_dt_list)[4]) {
            impl_explicit<FieldMap>(task, rk_df_dt_list);
            if (task.implicit_matrix_ != nullptr) {
                auto& f_tmp1 = rk_df_dt_list[0];
                sclx::assign_array(task.f0_, f_tmp1).get();
                sclx::algorithm::transform(
                    f_tmp1,
                    f_tmp1,
                    task.centering_offset_,
                    sclx::algorithm::minus<>{}
                );
                auto& explicit_indices = *task.explicit_indices_;
                sclx::execute_kernel([&](sclx::kernel_handler& handler) {
                    const auto& f                = task.f_;
                    const auto& centering_offset = task.centering_offset_;
                    handler.launch(
                        index_generator(f_tmp1.shape(), explicit_indices),
                        f_tmp1,
                        [=] __device__(const sclx::md_index_t<1>& index, const auto&){
                            f_tmp1[index] = f[index] - centering_offset;
                        }
                    );
                }).get();
                auto& implicit_matrix = *task.implicit_matrix_;
                auto& mat_mult        = *task.mat_mult;
                mat_mult(implicit_matrix, f_tmp1, task.f_).get();
                sclx::algorithm::transform(
                    task.f_,
                    task.f_,
                    task.centering_offset_,
                    sclx::algorithm::plus<>{}
                );
            }
        }

        template<class FieldMap>
        static void impl_explicit(
            advection_task& task,
            sclx::array<T, 1> (&rk_df_dt_list)[4]
        ) {
            auto& velocity_field
                = *(static_cast<FieldMap*>(task.field_map_.get()));
            auto& f0               = task.f0_;
            auto& f                = task.f_;
            auto& dt               = task.dt_;
            auto& centering_offset = task.centering_offset_;
            auto& divergence_op_   = task.divergence_op_;

            divergence_field_map<FieldMap> div_input_field{
                velocity_field,
                f0,
                centering_offset};

            divergence_op_->apply(div_input_field, rk_df_dt_list[0]);

            div_input_field.scalar_field_ = f;

            std::vector<std::future<void>> transform_futures;

            for (int i = 0; i < 3; ++i) {
                sclx::algorithm::elementwise_reduce(
                    forward_euler<T>{dt * runge_kutta_4::df_dt_weights[i]},
                    f,
                    f0,
                    rk_df_dt_list[i]
                );

                auto t_fut = sclx::algorithm::transform(
                    rk_df_dt_list[i],
                    rk_df_dt_list[i],
                    runge_kutta_4::summation_weights[i],
                    sclx::algorithm::multiplies<>{}
                );
                transform_futures.push_back(std::move(t_fut));

                divergence_op_->apply(div_input_field, rk_df_dt_list[i + 1]);
            }

            sclx::algorithm::transform(
                rk_df_dt_list[3],
                rk_df_dt_list[3],
                runge_kutta_4::summation_weights[3],
                sclx::algorithm::multiplies<>{}
            )
                .get();

            for (auto& t_fut : transform_futures) {
                t_fut.get();
            }

            sclx::algorithm::elementwise_reduce(
                sclx::algorithm::plus<>{},
                rk_df_dt_list[0],
                rk_df_dt_list[0],
                rk_df_dt_list[1],
                rk_df_dt_list[2],
                rk_df_dt_list[3]
            );

            sclx::algorithm::elementwise_reduce(
                forward_euler<T>{dt},
                f,
                f0,
                rk_df_dt_list[0]
            )
                .get();
        }

      private:
        std::shared_ptr<void> field_map_;
        using impl_t = void (*)(
            advection_task& task,
            sclx::array<T, 1> (&rk_df_dt_list)[4]
        );
        impl_t impl_;

        sclx::array<T, 1> f0_;
        sclx::array<T, 1> f_;
        T dt_;
        T centering_offset_;
        std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_;

        sclx::array<uint, 1>* explicit_indices_;
        matrix_type* implicit_matrix_;
        mat_mult_type* mat_mult;
    };

    class advection_executor {
      public:
        ~advection_executor() {
            if (thread_data_ == nullptr) {
                return;
            }
            thread_data_->stop_thread.store(true);
            thread_->join();
        }

        template<class FieldMap>
        static std::future<void> submit(
            std::shared_ptr<divergence_operator<T, Dimensions>>& divergence_op_,
            FieldMap& velocity_field,
            sclx::array<T, 1>& f0,
            sclx::array<T, 1>& f,
            T dt,
            T centering_offset,
            sclx::array<uint, 1>* explicit_indices = nullptr,
            matrix_type* implicit_matrix             = nullptr,
            mat_mult_type* mat_mult                  = nullptr
        ) {
            std::shared_ptr<advection_executor> executor = get_executor();
            auto task_ptr = std::make_unique<advection_task>(
                divergence_op_,
                velocity_field,
                f0,
                f,
                dt,
                centering_offset,
                explicit_indices,
                implicit_matrix,
                mat_mult
            );
            auto expected = false;
            while (!executor->thread_data_->is_preparing
                        .compare_exchange_weak(expected, true)) {
                expected = false;
                std::this_thread::yield();
            }
            while (executor->thread_data_->is_task_submitted.load()) {
                std::this_thread::yield();
            }
            executor->thread_data_->task          = std::move(task_ptr);
            executor->thread_data_->ready_promise = std::promise<void>();
            auto fut = executor->thread_data_->ready_promise.get_future();
            executor->thread_data_->is_task_submitted.store(true);
            executor->thread_data_->is_preparing.store(false);
            if (executor->thread_data_->stop_thread.load()) {
                sclx::throw_exception<std::runtime_error>(
                    "Executor thread was stopped before task was submitted.",
                    "naga::nonlocal_calculus::advection_operator::"
                );
            }
            return fut;
        }

        advection_executor(const advection_executor&)            = default;
        advection_executor& operator=(const advection_executor&) = default;
        advection_executor(advection_executor&&)                 = default;
        advection_executor& operator=(advection_executor&&)      = default;

        static uint max_concurrent_threads() {
            std::lock_guard<std::mutex> lock(get_executor_pool().executors_mutex
            );
            return static_cast<uint>(get_executor_pool().executors.size());
        }

        static void set_max_concurrent_threads(uint max_concurrent_threads) {
            std::lock_guard<std::mutex> lock(get_executor_pool().executors_mutex
            );
            std::vector<std::shared_ptr<advection_executor>>& old_executors
                = get_executor_pool().executors;
            std::vector<std::shared_ptr<advection_executor>> new_executors;
            for (uint i = 0; i < max_concurrent_threads; ++i) {
                if (i < old_executors.size()) {
                    new_executors.push_back(old_executors[i]);
                } else {
                    new_executors.push_back(
                        std::make_shared<advection_executor>(
                            advection_executor{}
                        )
                    );
                    new_executors.back()->init_thread();
                }
            }
            old_executors = new_executors;
        }

      private:
        advection_executor() = default;

        std::unique_ptr<std::thread> thread_{nullptr};

        void init_thread() {
            thread_data_      = std::make_shared<thread_data_t>();
            auto& thread_data = thread_data_;
            thread_ = std::make_unique<std::thread>([thread_data]() mutable {
                sclx::array<T, 1> rk_df_dt_list[4];
                thread_data->is_running.store(true);
                while (!thread_data->stop_thread.load()) {
                    if (thread_data->is_task_submitted.load()) {
                        (*thread_data->task)(rk_df_dt_list);
                        thread_data->ready_promise.set_value();
                        thread_data->is_task_submitted.store(false);
                    } else {
                        std::this_thread::yield();
                    }
                }
                thread_data->is_running.store(false);
            });
            while (!thread_data->is_running.load()) {
                std::this_thread::yield();
            }
        }

        struct thread_data_t {
            std::atomic<bool> is_task_submitted{false};
            std::atomic<bool> stop_thread{false};
            std::atomic<bool> is_preparing{false};
            std::atomic<bool> is_running{false};
            std::shared_ptr<advection_task> task{nullptr};
            std::promise<void> ready_promise;
        };

        struct executor_pool {
            std::vector<std::shared_ptr<advection_executor>> executors;
            std::mutex executors_mutex;
            uint next_executor{0};
            executor_pool() {
                int max_concurrent_threads_ = 4;
                for (uint i = 0; i < max_concurrent_threads_; ++i) {
                    executors.push_back(std::make_shared<advection_executor>(
                        advection_executor{}
                    ));
                    executors.back()->init_thread();
                }
            }
        };

        static executor_pool& get_executor_pool() {
            static executor_pool executor_pool;
            return executor_pool;
        }

        static std::shared_ptr<advection_executor> get_executor() {
            auto& executor_pool = get_executor_pool();
            std::lock_guard<std::mutex> lock(executor_pool.executors_mutex);
            auto executor
                = executor_pool.executors[executor_pool.next_executor];
            executor_pool.next_executor = (executor_pool.next_executor + 1)
                                        % executor_pool.executors.size();
            return executor;
        }

        std::shared_ptr<thread_data_t> thread_data_{nullptr};
    };

    void set_max_concurrent_threads(uint max_concurrent_threads) {
//        max_concurrent_threads = std::min(std::thread::hardware_concurrency(),
//                                          max_concurrent_threads);
//        advection_executor::set_max_concurrent_threads(max_concurrent_threads);
    }

    [[nodiscard]] uint max_concurrent_threads() const {
        return advection_executor::max_concurrent_threads();
    }

    template<class FieldMap>
    std::future<void> step_forward(
        FieldMap& velocity_field,
        sclx::array<T, 1>& f0,
        sclx::array<T, 1>& f,
        T dt,
        T centering_offset = T(0)
    ) {
        if (f0.elements() != domain_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "f0 must have the same number of elements as the domain for "
                "which this operator was created.",
                "naga::nonlocal_calculus::advection_operator::"
            );
        }
        if (f.elements() != domain_size_) {
            sclx::throw_exception<std::invalid_argument>(
                "f must have the same number of elements as the domain for "
                "which this operator was created.",
                "naga::nonlocal_calculus::advection_operator::"
            );
        }
        return advection_executor::submit(
            divergence_op_,
            velocity_field,
            f0,
            f,
            dt,
            centering_offset
        );
    }

    //        void compute_A_matrix(const T (&constant_velocity)[Dimensions]) {
    //        auto identity_mat = cusparse_desc_->identity_mat;
    //        auto A            = cusparse_desc_->A;
    //        A                 = matrix_type();
    //        cusparse_desc_
    //            ->mat_mult(
    //                -constant_velocity[0],
    //                0.,
    //                identity_mat,
    //                divergence_op_->cusparse_desc_->div_mats[0],
    //                A
    //            )
    //            .get();
    //        for (int d = 1; d < Dimensions; ++d) {
    //            cusparse_desc_
    //                ->mat_mult(
    //                    -constant_velocity[d],
    //                    1.,
    //                    identity_mat,
    //                    divergence_op_->cusparse_desc_->div_mats[d],
    //                    A
    //                )
    //                .get();
    //        }
    //        cusparse_desc_->A = A;
    //    }
    //
    //    std::future<void> fast_step_forward(
    //        sclx::array<T, 1> f0,
    //        sclx::array<T, 1> f,
    //        T dt,
    //        T centering_offset = T{0}
    //    ) {
    //        auto A = cusparse_desc_->A;
    //
    //        std::lock_guard<std::mutex> lock(*mutex_);
    //        sclx::array<T, 1> rk_df_dt_list[4];
    //        for (auto& df_dt : rk_df_dt_list) {
    //            auto [fut, array] = rk_df_dt_list_->front();
    //            if (fut.valid()) {
    //                fut.wait();
    //            }
    //            df_dt = std::move(array);
    //            rk_df_dt_list_->pop_front();
    //        }
    //        std::shared_future<void> ready_future;
    //        auto fut = std::async(std::launch::async, [=, &ready_future]()
    //        mutable {
    //            std::promise<void> ready_promise;
    //            ready_future = std::move(ready_promise.get_future());
    //
    //            sclx::algorithm::transform(
    //                f,
    //                f0,
    //                centering_offset,
    //                sclx::algorithm::minus<>{}
    //            )
    //                .get();
    //
    //            divergence_op_->cusparse_desc_->mat_mult(A, f,
    //            rk_df_dt_list[0])
    //                .get();
    //
    //            std::vector<std::future<void>> transform_futures;
    //
    //            for (int i = 0; i < 3; ++i) {
    //                sclx::algorithm::elementwise_reduce(
    //                    forward_euler<T>{dt *
    //                    runge_kutta_4::df_dt_weights[i]}, f, f0,
    //                    rk_df_dt_list[i]
    //                )
    //                    .get();
    //
    //                auto centering_fut = sclx::algorithm::transform(
    //                    f,
    //                    f,
    //                    centering_offset,
    //                    sclx::algorithm::minus<>{}
    //                );
    //
    //                auto t_fut = sclx::algorithm::transform(
    //                    rk_df_dt_list[i],
    //                    rk_df_dt_list[i],
    //                    runge_kutta_4::summation_weights[i],
    //                    sclx::algorithm::multiplies<>{}
    //                );
    //                transform_futures.push_back(std::move(t_fut));
    //
    //                centering_fut.get();
    //
    //                divergence_op_->cusparse_desc_
    //                    ->mat_mult(A, f, rk_df_dt_list[i + 1])
    //                    .get();
    //            }
    //
    //            sclx::algorithm::transform(
    //                rk_df_dt_list[3],
    //                rk_df_dt_list[3],
    //                runge_kutta_4::summation_weights[3],
    //                sclx::algorithm::multiplies<>{}
    //            )
    //                .get();
    //
    //            for (auto& t_fut : transform_futures) {
    //                t_fut.get();
    //            }
    //
    //            sclx::algorithm::elementwise_reduce(
    //                sclx::algorithm::plus<>{},
    //                rk_df_dt_list[0],
    //                rk_df_dt_list[0],
    //                rk_df_dt_list[1],
    //                rk_df_dt_list[2],
    //                rk_df_dt_list[3]
    //            );
    //
    //            sclx::algorithm::elementwise_reduce(
    //                forward_euler<T>{dt},
    //                f,
    //                f0,
    //                rk_df_dt_list[0]
    //            );
    //
    //            ready_promise.set_value();
    //        });
    //        for (auto& arr : rk_df_dt_list) {
    //            while (!ready_future.valid()) {
    //                std::this_thread::yield();
    //            }
    //            rk_df_dt_list_->emplace_back(ready_future, arr);
    //        }
    //
    //        return fut;
    //    }
    //
    //    std::future<void> step_forward_v2(
    //        const T (&constant_velocity)[Dimensions],
    //        sclx::array<T, 1> f0,
    //        sclx::array<T, 1> f,
    //        T dt,
    //        T centering_offset = T{0}
    //    ) {
    //        compute_A_matrix(constant_velocity);
    //        return fast_step_forward(f0, f, dt, centering_offset);
    //    }

    const divergence_operator<T, Dimensions>& divergence_op() const {
        return *divergence_op_;
    }

    void enable_implicit(
        const T (&velocity)[Dimensions],
        T dt,
        uint boundary_start_idx,
        sclx::array<T, 2> boundary_normals
    ) {
        sclx::array<const T, 3> weights = divergence_op_->weights();
        sclx::array<const uint, 2> indices
            = divergence_op_->support_indices();
        std::vector<int> implicit_points(
            indices.shape()[1],
            0
        );  // 0 is undetermined, 1 is implicit, -1 is explicit
        std::vector<std::vector<uint>> parent_nodes;

        uint processed_points = 0;
        uint num_matrices     = 0;
        for (uint p = boundary_start_idx; p < implicit_points.size(); ++p) {
            if (implicit_points[p] != 0) {
                continue;
            }

            bool valid_parent = true;
            for (uint j = 0; j < detail::num_interp_support; ++j) {
                const uint& ij = indices(j, p);
                if (implicit_points[ij] != 0) {
                    valid_parent = false;
                    break;
                }

                for (uint k = 0; k < detail::num_interp_support; ++k) {
                    const uint& ijk = indices(k, ij);
                    if (implicit_points[ijk] == 1) {
                        valid_parent = false;
                        break;
                    }

                    for (uint l = 0; l < detail::num_interp_support; ++l) {
                        const uint& ijkl = indices(l, ijk);
                        if (implicit_points[ijkl] == 1) {
                            valid_parent = false;
                            break;
                        }
                    }
                }

                if (!valid_parent) {
                    break;
                }
            }

            if (!valid_parent) {
                continue;
            }

            ++num_matrices;

            parent_nodes.emplace_back();
            auto& dependent_explicit_nodes = parent_nodes.back();

            implicit_points[p] = 1;
            ++processed_points;

            for (uint dependent_p = 0; dependent_p < detail::num_interp_support;
                 ++dependent_p) {
                const uint& i = indices(dependent_p, p);
                implicit_points[i]     = 1;
                dependent_explicit_nodes.push_back(i);
                ++processed_points;
            }

            for (uint dependent_p = 0; dependent_p < detail::num_interp_support;
                 ++dependent_p) {
                const uint& i = indices(dependent_p, p);
                for (uint support_p = 0; support_p < detail::num_interp_support;
                     ++support_p) {
                    const uint& k = indices(support_p, i);
                    if (implicit_points[k] == 0) {
                        implicit_points[k] = -1;
                        ++processed_points;
                    }
                }
            }
        }

        std::vector<uint> explicit_indices;
        for (uint p = 0; p < implicit_points.size(); ++p) {
            if (implicit_points[p] == 0) {
                implicit_points[p] = -1;
                if (p < boundary_start_idx) {
                    explicit_indices.push_back(p);
                }
                ++processed_points;
            }
        }

        std::vector<int> has_been_processed(processed_points, 0);

        struct matrix_meta {
            std::vector<uint> indices;
            std::vector<std::unordered_map<uint, T>> weights;
        };

        struct implicit_matrix {
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
            std::vector<uint> global_indices;
        };

        std::vector<implicit_matrix> implicit_matrices_(num_matrices);

        std::atomic<uint> total_weights{0};
#pragma omp parallel for
        for (uint m = 0; m < num_matrices; ++m) {
            auto& dependent_explicit_nodes = parent_nodes[m];
            matrix_meta matrix;

            std::unordered_map<uint, uint> indices_map;
            for (auto& dependent_p : dependent_explicit_nodes) {
                auto i = matrix.indices.size();
                matrix.indices.push_back(dependent_p);
                matrix.weights.push_back(std::unordered_map<uint, T>{});
                matrix.weights.back()[i]        = T{1.};
                indices_map[dependent_p]        = static_cast<uint>(i);
                has_been_processed[dependent_p] = 1;
            }
            for (auto& dependent_p : dependent_explicit_nodes) {
                auto i = indices_map[dependent_p];
                for (uint s = 0; s < detail::num_interp_support; ++s) {
                    const uint& support_node = indices(s, dependent_p);
                    if (indices_map.find(support_node) == indices_map.end()) {
                        if (implicit_points[support_node] != -1) {
                            sclx::throw_exception<std::runtime_error>(
                                "Implicit point is not -1.",
                                "naga::nonlocal_calculus::advection_operator::"
                            );
                        }
                        auto j = matrix.indices.size();
                        matrix.indices.push_back(support_node);
                        matrix.weights.push_back(std::unordered_map<uint, T>{}
                        );
                        matrix.weights.back()[j]  = T{1.};
                        indices_map[support_node] = j;
                        if (has_been_processed[support_node] == 0) {
                            if (support_node < boundary_start_idx) {
                                explicit_indices.push_back(support_node);
                            }
                            has_been_processed[support_node] = 1;
                        }
                    }
                    if (dependent_p >= boundary_start_idx) {
                        continue;
                    }
                    auto j      = indices_map[support_node];
                    auto weight = matrix.weights[i][j];
                    for (uint d = 0; d < Dimensions; ++d) {
                        weight += weights(d, s, dependent_p) * velocity[d] * dt;
                    }
                    matrix.weights[i][j] = weight;
                }
            }
            auto matrix_size = matrix.indices.size();

            auto& implicit_mat = implicit_matrices_[m];
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat(
                matrix_size,
                matrix_size
            );
            mat.setZero();
            for (uint i = 0; i < matrix_size; ++i) {
                for (auto [j, weight] : matrix.weights[i]) {
                    mat(i, j) = weight;
                }
            }
            implicit_mat.mat
                = mat.inverse()
                      .block(0, 0, detail::num_interp_support, matrix_size);
            for (auto& index : matrix.indices) {
                implicit_mat.global_indices.push_back(index);
            }

            for (uint i = detail::num_interp_support; i < matrix_size; ++i) {
                auto& index = matrix.indices[i];
                explicit_indices.push_back(index);
            }
            total_weights.fetch_add(matrix_size * matrix_size);
        }

        std::sort(explicit_indices.begin(), explicit_indices.end());
        auto end
            = std::unique(explicit_indices.begin(), explicit_indices.end());
        explicit_indices.erase(end, explicit_indices.end());

        sclx::array<T, 1> sparse_weights;
        sclx::array<int, 1> sparse_row_offsets;
        sclx::array<int, 1> sparse_column_indices;

        std::vector<std::pair<int, implicit_matrix*>> implicit_matrix_ptrs(
            implicit_points.size(),
            {0, nullptr}
        );
        for (auto& matrix : implicit_matrices_) {
            for (int i = 0; i < matrix.mat.rows(); ++i) {
                auto row_index                  = matrix.global_indices[i];
                implicit_matrix_ptrs[row_index] = std::make_pair(i, &matrix);
            }
        }

        auto num_implicit = std::count_if(
            implicit_points.begin(),
            implicit_points.end(),
            [](int val) { return val == 1; });
        sparse_weights
            = sclx::array<T, 1>{total_weights + implicit_points.size() - num_implicit};
        sparse_column_indices
            = sclx::array<int, 1>{sparse_weights.elements()};
        sparse_row_offsets = sclx::array<int, 1>{implicit_points.size() + 1};
        sclx::fill(sparse_row_offsets, 0);
        int nonzero_value_count = 0;
        for (int row_index = 0; row_index < implicit_points.size();
             ++row_index) {
            if (implicit_matrix_ptrs[row_index].second == nullptr) {
                sparse_row_offsets[row_index]       = nonzero_value_count;
                sparse_weights[nonzero_value_count] = T{1.};
                sparse_column_indices[nonzero_value_count] = row_index;
                ++nonzero_value_count;
                continue;
            }
            auto i             = implicit_matrix_ptrs[row_index].first;
            const auto& matrix = *implicit_matrix_ptrs[row_index].second;
            sparse_row_offsets[row_index] = nonzero_value_count;
            for (int j = 0; j < matrix.mat.cols(); ++j) {
                auto col_index = matrix.global_indices[j];
                if (col_index > std::numeric_limits<int>::max()) {
                    sclx::throw_exception<std::runtime_error>(
                        "Column index is too large.",
                        "naga::nonlocal_calculus::advection_operator::"
                    );
                }
                sparse_weights[nonzero_value_count] = matrix.mat(i, j);
                sparse_column_indices[nonzero_value_count]
                    = static_cast<int>(col_index);
                ++nonzero_value_count;
            }
        }
        if (nonzero_value_count > sparse_weights.elements()) {
            sclx::throw_exception<std::runtime_error>(
                "Nonzero value count is greater than the number of elements in "
                "the sparse weights/row indices arrays.",
                "naga::nonlocal_calculus::advection_operator::"
            );
        }
        sparse_row_offsets[implicit_points.size()] = nonzero_value_count;
        implicit_matrix_                           = matrix_type(
            implicit_points.size(),
            sparse_weights,
            sparse_row_offsets,
            sparse_column_indices
        );

        explicit_indices_ = sclx::array<uint, 1>{explicit_indices.size()};
        std::copy(
            explicit_indices.begin(),
            explicit_indices.end(),
            explicit_indices_.begin()
        );

        is_implicit_ = true;
        implicit_dt_ = dt;
        std::copy(velocity, velocity + Dimensions, implicit_velocity_);
    }

    std::future<void> apply_implicit(
        sclx::array<T, 1> f0,
        sclx::array<T, 1> f,
        T centering_offset = T{0}
    ) {
        if (!is_implicit_) {
            sclx::throw_exception<std::runtime_error>(
                "Implicit advection must be enabled before calling "
                "apply_implicit.",
                "naga::nonlocal_calculus::advection_operator::"
            );
        }

        auto velocity_map
            = constant_velocity_field<T, Dimensions>::create(implicit_velocity_
            );

        return advection_executor::submit(
            divergence_op_,
            velocity_map,
            f0,
            f,
            implicit_dt_,
            centering_offset,
            &explicit_indices_,
            &implicit_matrix_,
            &mat_mult
        );
    }

    template<class Archive>
    void save(Archive& ar) const {
        ar(*divergence_op_);
        ar(domain_size_);
        auto max_concurrent_threads_ = max_concurrent_threads();
        ar(max_concurrent_threads_);
    }

    template<class Archive>
    void load(Archive& ar) {
        divergence_op_ = std::make_shared<divergence_operator<T, Dimensions>>();
        ar(*divergence_op_);
        ar(domain_size_);
        auto max_concurrent_threads_ = max_concurrent_threads();
        ar(max_concurrent_threads_);
        set_max_concurrent_threads(max_concurrent_threads_);
    }

    //    void enable_cusparse_algorithm() {
    //        divergence_op_->enable_cusparse_algorithm();
    //
    //        if (cusparse_desc_ == nullptr) {
    //            cusparse_desc_ = std::make_shared<cusparse_algo_desc>();
    //
    //            auto& identity_mat1 = cusparse_desc_->identity_mat1;
    //            sclx::array<T, 2> ones{1, domain_size_};
    //            sclx::fill(ones, T(1));
    //            sclx::array<uint, 2> indices{1, domain_size_};
    //            std::iota(indices.begin(), indices.end(), 0);
    //            identity_mat1
    //                = matrix_type::create_from_index_stencil(indices, ones);
    //            cusparse_desc_->A = matrix_type();
    //        }
    //    }

    //    advection_operator
    //    make_fast_operator_for_velocity(const T
    //    (&constant_velocity)[Dimensions]) {
    //        bool was_cusparse_enabled  = cusparse_algorithm_enabled();
    //        advection_operator fast_op = *this;
    //        fast_op.cusparse_desc_     = nullptr;
    //        fast_op.enable_cusparse_algorithm();
    //        fast_op.compute_A_matrix(constant_velocity);
    //        return fast_op;
    //    }
    //
    //    void disable_cusparse_algorithm() {
    //        divergence_op_->disable_cusparse_algorithm();
    //        cusparse_desc_ = nullptr;
    //    }
    //
    //    bool cusparse_algorithm_enabled() const {
    //        return divergence_op_->cusparse_algorithm_enabled();
    //    }

  public:
    std::shared_ptr<divergence_operator<T, Dimensions>> divergence_op_;
    uint domain_size_;
    bool is_implicit_{false};
    T implicit_velocity_[Dimensions];
    T implicit_dt_;
    sclx::array<uint, 1> explicit_indices_;
    matrix_type implicit_matrix_;
    mat_mult_type mat_mult;

    struct runge_kutta_4 {
        static constexpr T df_dt_weights[3] = {1. / 2., 1. / 2., 1.};
        static constexpr T summation_weights[4]
            = {1. / 6., 2. / 6., 2. / 6., 1. / 6.};
    };

    //    using matrix_type =
    //        typename divergence_operator<T, Dimensions>::matrix_type;
    //    using vector_type =
    //        typename divergence_operator<T, Dimensions>::vector_type;
    //    struct cusparse_algo_desc {
    //        matrix_type identity_mat1;
    //        matrix_type identity_mat2;
    //        matrix_type A;
    //        naga::linalg::matrix_mult<matrix_type, matrix_type, matrix_type>
    //            mat_mult;
    //    };
    //    std::shared_ptr<cusparse_algo_desc> cusparse_desc_{nullptr};
};

}  // namespace naga::nonlocal_calculus
