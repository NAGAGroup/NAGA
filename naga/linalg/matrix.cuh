
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
#include "../detail/cusparse.cuh"
#include <numeric>
#include <scalix/algorithm/reduce.cuh>
#include <scalix/array.cuh>
#include <scalix/execute_kernel.cuh>
#include <type_traits>
#include <utility>

namespace naga::linalg {

namespace detail {
struct sparse_mat_desc_deleter {
    void operator()(cusparseSpMatDescr_t* desc) {
        if (*desc != nullptr) {
            cusparseDestroySpMat(*desc);
        }
        delete desc;
    }
};

struct dense_vec_desc_deleter {
    void operator()(cusparseDnVecDescr_t* desc) {
        if (*desc != nullptr) {
            cusparseDestroyDnVec(*desc);
        }
        delete desc;
    }
};

template<class T>
struct cusparse_data {};

template<>
struct cusparse_data<float> {
    static constexpr auto type = CUDA_R_32F;
};

template<>
struct cusparse_data<double> {
    static constexpr auto type = CUDA_R_64F;
};

}  // namespace detail

enum class storage_type {
    dense,
    sparse_csr,
    sparse_csc,
};

template<class T, storage_type StorageType>
class vector {
    static_assert(std::is_same_v<T, T>, "matrix type not implemented");
};

template<class T, storage_type StorageType>
class matrix {
    static_assert(std::is_same_v<T, T>, "matrix type not implemented");
};

template<
    class LinAlgTypeT = void,
    class LinAlgTypeU = void,
    class LinAlgTypeR = void>
class matrix_mult {
  public:
    template<class LinAlgTypeTO, class LinAlgTypeUO, class LinAlgTypeRO>
    std::future<void>
    operator()(LinAlgTypeTO A, LinAlgTypeUO B, LinAlgTypeRO result) {
        return matrix_mult<LinAlgTypeTO, LinAlgTypeUO, LinAlgTypeRO>{}(
            A,
            B,
            result
        );
    }

    template<class LinAlgTypeTO, class LinAlgTypeUO, class LinAlgTypeRO>
    std::future<void> operator()(
        int device,
        LinAlgTypeTO A,
        LinAlgTypeUO B,
        LinAlgTypeRO result
    ) {
        return matrix_mult<LinAlgTypeTO, LinAlgTypeUO, LinAlgTypeRO>{}(
            device,
            A,
            B,
            result
        );
    }

    template<
        class Alpha,
        class Beta,
        class LinAlgTypeTO,
        class LinAlgTypeUO,
        class LinAlgTypeRO>
    std::future<void> operator()(
        Alpha alpha,
        Beta beta,
        LinAlgTypeTO A,
        LinAlgTypeUO B,
        LinAlgTypeRO result
    ) {
        return matrix_mult<LinAlgTypeTO, LinAlgTypeUO, LinAlgTypeRO>{}(
            alpha,
            beta,
            A,
            B,
            result
        );
    }

    template<
        class Alpha,
        class Beta,
        class LinAlgTypeTO,
        class LinAlgTypeUO,
        class LinAlgTypeRO>
    std::future<void> operator()(
        int device,
        Alpha alpha,
        Beta beta,
        LinAlgTypeTO A,
        LinAlgTypeUO B,
        LinAlgTypeRO result
    ) {
        return matrix_mult<LinAlgTypeTO, LinAlgTypeUO, LinAlgTypeRO>{}(
            device,
            alpha,
            beta,
            A,
            B,
            result
        );
    }
};

#define NAGA_LINALG_DECLARE_FRIENDS()                                          \
    template<class LinAlgTypeT, class LinAlgTypeU, class LinAlgTypeR>          \
    friend class matrix_mult;

template<class T>
class vector<T, storage_type::dense> {
  public:
    vector() = default;

    __host__ vector(uint size) : vector(std::move(sclx::array<T, 1>{size})) {}

    __host__ vector(sclx::array<T, 1> values)
        : desc_(std::make_shared<description_info>(description_info{
            std::move(values)})) {
        cusparseDnVecDescr_t desc = nullptr;
        if (desc_->values_.elements()
            > static_cast<uint>(std::numeric_limits<int>::max())) {
            sclx::throw_exception<std::invalid_argument>(
                "values.elements() > std::numeric_limits<int>::max()",
                "naga::linalg::vector::"
            );
        }
        auto error = cusparseCreateDnVec(
            &desc,
            this->size(),
            desc_->values_.data().get(),
            detail::cusparse_data<T>::type
        );
        if (error != CUSPARSE_STATUS_SUCCESS) {
            sclx::throw_exception<std::runtime_error>(
                "cusparseCreateDnVec failed with error code "
                    + std::to_string(error),
                "naga::linalg::vector::"
            );
        }
        desc_->cusparse_desc_
            = std::move(typename description_info::cusparse_desc_t(
                new cusparseDnVecDescr_t(desc)
            ));
    }

    int size() const { return static_cast<int>(desc_->values_.elements()); }

    sclx::array<T, 1> values() const { return desc_->values_; }

    NAGA_LINALG_DECLARE_FRIENDS()

  private:
    struct description_info {
        using cusparse_desc_t = std::
            unique_ptr<cusparseDnVecDescr_t, detail::dense_vec_desc_deleter>;
        sclx::array<T, 1> values_;
        cusparse_desc_t cusparse_desc_;
    };
    std::shared_ptr<description_info> desc_
        = std::make_shared<description_info>();
};

template<class T>
class matrix<T, storage_type::sparse_csr> {
  public:
    struct description_info {
        using cusparse_desc_t = std::
            unique_ptr<cusparseSpMatDescr_t, detail::sparse_mat_desc_deleter>;
        int rows{};
        int columns{};
        sclx::array<T, 1> values_;
        sclx::array<int, 1> row_offsets_;
        sclx::array<int, 1> column_indices_;
        cusparse_desc_t cusparse_desc_;
    };

    matrix() = default;

    __host__ matrix(
        int matrix_size,  // square matrix
        sclx::array<T, 1> values,
        sclx::array<int, 1> row_offsets,
        sclx::array<int, 1> column_indices
    )
        : matrix(
            matrix_size,
            matrix_size,
            std::move(values),
            std::move(row_offsets),
            std::move(column_indices)
        ) {}

    __host__ matrix(
        int m,
        int n,
        sclx::array<T, 1> values,
        sclx::array<int, 1> row_offsets,
        sclx::array<int, 1> column_indices
    )
        : desc_(std::make_shared<description_info>(description_info{
            m,
            n,
            std::move(values),
            std::move(row_offsets),
            std::move(column_indices)})) {
        cusparseSpMatDescr_t desc = nullptr;
        auto error                = cusparseCreateCsr(
            &desc,
            m,
            n,
            nnz(),
            static_cast<void*>(desc_->row_offsets_.data().get()),
            static_cast<void*>(desc_->column_indices_.data().get()),
            static_cast<void*>(desc_->values_.data().get()),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            detail::cusparse_data<T>::type
        );
        if (error != CUSPARSE_STATUS_SUCCESS) {
            sclx::throw_exception<std::runtime_error>(
                "cusparseCreateCsr failed with error code "
                    + std::to_string(error),
                "naga::linalg::square_matrix::"
            );
        }
        desc_->cusparse_desc_
            = std::move(typename description_info::cusparse_desc_t(
                new cusparseSpMatDescr_t(desc)
            ));
    }

    template<class ColumnCount, class IndexType>
    __host__ static matrix create_from_index_stencil(
        ColumnCount columns,
        sclx::array<IndexType, 2> index_stencil,
        sclx::array<T, 2> values
    ) {
        if (!check_cast_to_int(columns)) {
            sclx::throw_exception<std::invalid_argument>(
                "columns not convertible to int",
                "naga::linalg::square_matrix::"
            );
        }
        check_index_stencil(index_stencil);
        sclx::array<int, 1> row_offsets{index_stencil.shape()[1] + 1};
        if (index_stencil.shape()[0]
                > static_cast<uint>(std::numeric_limits<int>::max())
            || index_stencil.shape()[1]
                   > static_cast<uint>(std::numeric_limits<int>::max())) {
            sclx::throw_exception<std::invalid_argument>(
                "index_stencil.shape()[0] > std::numeric_limits<int>::max() || "
                "index_stencil.shape()[1] > std::numeric_limits<int>::max()",
                "naga::linalg::square_matrix::"
            );
        }
        auto elem_per_row = static_cast<int>(index_stencil.shape()[0]);
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<1>{row_offsets.shape()},
                row_offsets,
                [=] __device__(sclx::md_index_t<1> index, const auto&) {
                    row_offsets[index]
                        = elem_per_row * static_cast<int>(index[0]);
                }
            );
        });
        sclx::array<int, 1> column_indices{index_stencil.elements()};
        sclx::array<T, 1> flat_values{values.elements()};
        auto sort_pattern = std::vector<int>(index_stencil.shape()[0]);
        for (int r = 0; r < index_stencil.shape()[1]; ++r) {
            std::iota(sort_pattern.begin(), sort_pattern.end(), 0);
            std::sort(
                sort_pattern.begin(),
                sort_pattern.end(),
                [&](int a, int b) {
                    return index_stencil(a, r) < index_stencil(b, r);
                }
            );
            for (const auto& i : sort_pattern) {
                column_indices[row_offsets[r] + i] = index_stencil(i, r);
                flat_values[row_offsets[r] + i]    = values(i, r);
            }
        }

        if (index_stencil.shape()[1]
            > static_cast<uint>(std::numeric_limits<int>::max())) {
            sclx::throw_exception<std::invalid_argument>(
                "index_stencil.shape()[1] > std::numeric_limits<int>::max()",
                "naga::linalg::square_matrix::"
            );
        }

        return matrix(
            static_cast<int>(index_stencil.shape()[1]),
            columns,
            std::move(flat_values),
            std::move(row_offsets),
            std::move(column_indices)
        );
    }

    template<class IndexType>
    __host__ static matrix create_from_index_stencil(  // square matrix
        sclx::array<IndexType, 2> index_stencil,
        sclx::array<T, 2> values
    ) {
        return create_from_index_stencil(
            index_stencil.shape()[1],
            std::move(index_stencil),
            std::move(values)
        );
    }

    int rows() const { return desc_->rows; }

    int columns() const { return desc_->columns; }

    int nnz() const {
        return static_cast<int>(desc_->column_indices_.elements());
    }

    NAGA_LINALG_DECLARE_FRIENDS()

    template<class IndexType>
    __host__ __device__ static constexpr bool check_cast_to_int(IndexType value
    ) {
        if constexpr (std::is_signed_v<IndexType> && value < 0) {
            return false;
        }
        return value < std::numeric_limits<int>::max();
    }

    template<class IndexType>
    __host__ static void
    check_index_stencil(sclx::array<IndexType, 2> index_stencil) {

        sclx::array<int, 2> invalid_index_stencils{index_stencil.shape()};
        sclx::execute_kernel([&](sclx::kernel_handler& handler) {
            handler.launch(
                sclx::md_range_t<2>(invalid_index_stencils.shape()),
                invalid_index_stencils,
                [=] __device__(sclx::md_index_t<2> index, const auto&) {
                    invalid_index_stencils[index]
                        = check_cast_to_int(index_stencil[index]) ? 0 : 1;
                }
            );
        }).get();
        auto sum_invalid_flags = sclx::algorithm::reduce(
            invalid_index_stencils,
            0,
            std::plus<int>{}
        );
        if (sum_invalid_flags > 0) {
            sclx::throw_exception<std::invalid_argument>(
                "index_stencil contains invalid indices",
                "naga::linalg::square_matrix::"
            );
        }
    }

    __host__ const description_info& get_description_info() const {
        return *desc_;
    }

  private:
    std::shared_ptr<description_info> desc_
        = std::make_shared<description_info>();
};

template<class T>
class matrix_mult<
    matrix<T, storage_type::sparse_csr>,
    vector<T, storage_type::dense>,
    vector<T, storage_type::dense>> {
  public:
    using matrix_type = matrix<T, storage_type::sparse_csr>;
    using vector_type = vector<T, storage_type::dense>;

    std::future<void> operator()(
        int device,
        T alpha,
        T beta,
        matrix_type A,
        vector_type b,
        vector_type y
    ) {
        return get_next_thread(device)->submit(alpha, beta, A, b, y);
    }

    std::future<void>
    operator()(T alpha, T beta, matrix_type A, vector_type b, vector_type y) {
        return
        operator()(sclx::cuda::traits::current_device(), alpha, beta, A, b, y);
    }

    std::future<void> operator()(matrix_type A, vector_type b, vector_type y) {
        return
        operator()(sclx::cuda::traits::current_device(), 1.0, 0.0, A, b, y);
    }

    std::future<void>
    operator()(int device, matrix_type A, vector_type b, vector_type y) {
        return operator()(device, 1.0, 0.0, A, b, y);
    }

  private:
    struct op_thread {
        op_thread(op_thread&&) = default;

        op_thread(int device) {
            problem_ = std::make_shared<problem_definition>();
            thread_  = std::thread([&, device]() {
                sclx::cuda::set_device(device);
                const auto& handle
                    = naga::detail::cusparse::handle_t::create_for_device(device
                    );
                sclx::array<std::byte, 1> buffer;
                is_running_ = true;
                while (is_running_) {
                    if (is_task_submitted_) {
                        size_t buffer_size;
                        auto matA  = *(problem_->A.desc_->cusparse_desc_);
                        auto vecX  = *(problem_->x.desc_->cusparse_desc_);
                        auto vecY  = *(problem_->y.desc_->cusparse_desc_);
                        auto error = cusparseSpMV_bufferSize(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &problem_->alpha,
                            matA,
                            vecX,
                            &problem_->beta,
                            vecY,
                            detail::cusparse_data<T>::type,
                            CUSPARSE_SPMV_ALG_DEFAULT,
                            &buffer_size
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpMV_bufferSize failed with error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }
                        if (buffer_size > buffer.elements()) {
                            buffer = sclx::array<std::byte, 1>{buffer_size};
                        }
                        auto error2 = cusparseSpMV(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &problem_->alpha,
                            matA,
                            vecX,
                            &problem_->beta,
                            vecY,
                            detail::cusparse_data<T>::type,
                            CUSPARSE_SPMV_ALG_DEFAULT,
                            static_cast<void*>(buffer.data().get())
                        );
                        if (error2 != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpMV failed with error code "
                                    + std::to_string(error2),
                                "naga::linalg::dot::"
                            );
                        }
                        sclx::cuda::stream_synchronize();
                        promise_.set_value();
                        is_task_submitted_.store(false);
                    } else {
                        std::this_thread::yield();
                    }
                }
            });

            while (!is_running_) {
                std::this_thread::yield();
            }
        }

        std::future<void>
        submit(T alpha, T beta, matrix_type A, vector_type x, vector_type y) {
            auto expected = false;
            while (!is_preparing_.compare_exchange_weak(expected, true)) {
                expected = false;
            }
            while (is_task_submitted_) {
                std::this_thread::yield();
            }

            problem_->A     = A;
            problem_->x     = x;
            problem_->y     = y;
            problem_->alpha = alpha;
            problem_->beta  = beta;
            promise_        = std::move(std::promise<void>{});

            auto future        = promise_.get_future();
            is_task_submitted_ = true;
            is_preparing_      = false;
            return future;
        }

        ~op_thread() {
            is_running_ = false;
            thread_.join();
        }
        struct problem_definition {
            matrix_type A;
            vector_type x;
            vector_type y;
            T alpha;
            T beta;
        };
        std::promise<void> promise_;
        std::shared_ptr<problem_definition> problem_;
        std::thread thread_;
        std::atomic<bool> is_running_;
        std::atomic<bool> is_task_submitted_{false};
        std::atomic<bool> is_preparing_{false};
    };

    using thread_pool_t = std::deque<std::unique_ptr<op_thread>>;

    static std::vector<thread_pool_t> make_thread_pool() {
        std::vector<thread_pool_t> threads;
        for (int i = 0; i < sclx::cuda::traits::device_count(); ++i) {
            threads.emplace_back(thread_pool_t{});
            for (int p = 0; p < 4; ++p) {
                threads.back().emplace_back(std::make_unique<op_thread>(i));
            }
        }

        return threads;
    }

    static std::vector<std::atomic<uint>> make_thread_counters() {
        std::vector<std::atomic<uint>> counters(
            sclx::cuda::traits::device_count()
        );
        for (auto& counter : counters) {
            counter.store(0);
        }

        return counters;
    }

    static op_thread* get_next_thread(int device) {
        static auto threads  = make_thread_pool();
        static auto counters = make_thread_counters();
        auto& thread_counter = counters[device];
        auto expected        = thread_counter.load();
        auto new_value       = (expected + 1) % threads[device].size();
        while (!thread_counter.compare_exchange_weak(expected, new_value)) {
            new_value = (expected + 1) % threads[device].size();
        }
        return threads[device][expected].get();
    }
};

template<class T>
class matrix_mult<
    matrix<T, storage_type::sparse_csr>,
    matrix<T, storage_type::sparse_csr>,
    matrix<T, storage_type::sparse_csr>> {
  public:
    using matrix_type = matrix<T, storage_type::sparse_csr>;

    std::future<void> operator()(
        int device,
        T alpha,
        T beta,
        matrix_type A,
        matrix_type B,
        matrix_type C
    ) {
        return get_next_thread(device)->submit(alpha, beta, A, B, C);
    }

    std::future<void>
    operator()(T alpha, T beta, matrix_type A, matrix_type B, matrix_type C) {
        return
        operator()(sclx::cuda::traits::current_device(), alpha, beta, A, B, C);
    }

    std::future<void> operator()(matrix_type A, matrix_type B, matrix_type C) {
        return
        operator()(sclx::cuda::traits::current_device(), 1.0, 0.0, A, B, C);
    }

    std::future<void>
    operator()(int device, matrix_type A, matrix_type B, matrix_type C) {
        return operator()(device, 1.0, 0.0, A, B, C);
    }

  private:
    struct op_thread {
        op_thread(op_thread&&) = default;

        op_thread(int device) {
            problem_ = std::make_shared<problem_definition>();
            thread_  = std::thread([&, device]() {
                sclx::cuda::set_device(device);
                const auto& handle
                    = naga::detail::cusparse::handle_t::create_for_device(device
                    );
                size_t buffer_size1{0};
                size_t buffer_size2{0};
                sclx::array<std::byte, 1> buffer1;
                sclx::array<std::byte, 1> buffer2;
                is_running_ = true;
                while (is_running_) {
                    if (is_task_submitted_) {
                        // init gemm desc
                        const auto& gemm_desc = naga::detail::cusparse::
                            gemm_desc_t::create_for_device(device);
                        const auto& matA = *(problem_->A.desc_->cusparse_desc_);
                        const auto& matB = *(problem_->B.desc_->cusparse_desc_);

                        // init C
                        cusparseSpMatDescr_t matC       = nullptr;
                        problem_->C.desc_->row_offsets_ = sclx::array<int, 1>{
                            static_cast<uint>(problem_->A.rows() + 1)};
                        problem_->C.desc_->rows    = problem_->A.rows();
                        problem_->C.desc_->columns = problem_->B.columns();
                        auto error                 = cusparseCreateCsr(
                            &matC,
                            4,
                            4,
                            0,
                            nullptr,
                            nullptr,
                            nullptr,
                            CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO,
                            detail::cusparse_data<T>::type
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpMV_bufferSize failed with error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }
                        problem_->C.desc_->cusparse_desc_ = std::move(
                            typename matrix_type::description_info::
                                cusparse_desc_t(new cusparseSpMatDescr_t(matC))
                        );

                        // get buffer1 size
                        error = cusparseSpGEMM_workEstimation(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &problem_->alpha,
                            matA,
                            matB,
                            &problem_->beta,
                            matC,
                            detail::cusparse_data<T>::type,
                            CUSPARSE_SPGEMM_DEFAULT,
                            gemm_desc,
                            &buffer_size1,
                            nullptr
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpGEMM_workEstimation failed with "
                                 "error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }

                        // allocate and assign buffer1 to gemm desc
                        if (buffer_size1 > buffer1.elements()) {
                            buffer1 = sclx::array<std::byte, 1>{buffer_size1};
                        }
                        error = cusparseSpGEMM_workEstimation(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &problem_->alpha,
                            matA,
                            matB,
                            &problem_->beta,
                            matC,
                            detail::cusparse_data<T>::type,
                            CUSPARSE_SPGEMM_DEFAULT,
                            gemm_desc,
                            &buffer_size1,
                            buffer1.data().get()
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpGEMM_workEstimation failed with "
                                 "error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }

                        // get buffer2 size
                        error = cusparseSpGEMM_compute(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &problem_->alpha,
                            matA,
                            matB,
                            &problem_->beta,
                            matC,
                            detail::cusparse_data<T>::type,
                            CUSPARSE_SPGEMM_DEFAULT,
                            gemm_desc,
                            &buffer_size2,
                            nullptr
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpGEMM_compute failed with "
                                 "error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }

                        // allocate buffer2 and use it to compute partial A*B
                        if (buffer_size2 > buffer2.elements()) {
                            buffer2 = sclx::array<std::byte, 1>{buffer_size2};
                        }
                        error = cusparseSpGEMM_compute(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &problem_->alpha,
                            matA,
                            matB,
                            &problem_->beta,
                            matC,
                            detail::cusparse_data<T>::type,
                            CUSPARSE_SPGEMM_DEFAULT,
                            gemm_desc,
                            &buffer_size2,
                            buffer2.data().get()
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpGEMM_compute failed with "
                                 "error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }

                        // get matC properties so we can assign values
                        // to the output data
                        int64_t C_num_rows1, C_num_cols1, C_nnz1;
                        error = cusparseSpMatGetSize(
                            matC,
                            &C_num_rows1,
                            &C_num_cols1,
                            &C_nnz1
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpMatGetSize failed with "
                                 "error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }

                        problem_->C.desc_->values_
                            = sclx::array<T, 1>{static_cast<uint>(C_nnz1)};
                        problem_->C.desc_->column_indices_
                            = sclx::array<int, 1>{static_cast<uint>(C_nnz1)};

                        // update matC with the new pointers
                        error = cusparseCsrSetPointers(
                            matC,
                            problem_->C.desc_->row_offsets_.data().get(),
                            problem_->C.desc_->column_indices_.data().get(),
                            problem_->C.desc_->values_.data().get()
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseCsrSetPointers failed with "
                                 "error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }

                        // copy A*B to C
                        error = cusparseSpGEMM_copy(
                            handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &problem_->alpha,
                            matA,
                            matB,
                            &problem_->beta,
                            matC,
                            detail::cusparse_data<T>::type,
                            CUSPARSE_SPGEMM_DEFAULT,
                            gemm_desc
                        );
                        if (error != CUSPARSE_STATUS_SUCCESS) {
                            sclx::throw_exception<std::runtime_error>(
                                "cusparseSpGEMM_copy failed with "
                                 "error "
                                 "code "
                                    + std::to_string(error),
                                "naga::linalg::dot::"
                            );
                        }
                        sclx::cuda::stream_synchronize();
                        promise_.set_value();
                        is_task_submitted_.store(false);
                    } else {
                        std::this_thread::yield();
                    }
                }
            });

            while (!is_running_) {
                std::this_thread::yield();
            }
        }

        std::future<void>
        submit(T alpha, T beta, matrix_type A, matrix_type B, matrix_type C) {
            auto expected = false;
            while (!is_preparing_.compare_exchange_weak(expected, true)) {
                expected = false;
            }
            while (is_task_submitted_) {
                std::this_thread::yield();
            }

            problem_->A     = A;
            problem_->B     = B;
            problem_->C     = C;
            problem_->alpha = alpha;
            problem_->beta  = beta;
            promise_        = std::move(std::promise<void>{});

            auto future        = promise_.get_future();
            is_task_submitted_ = true;
            is_preparing_      = false;
            return future;
        }

        ~op_thread() {
            is_running_ = false;
            thread_.join();
        }
        struct problem_definition {
            matrix_type A;
            matrix_type B;
            matrix_type C;
            T alpha;
            T beta;
        };
        std::promise<void> promise_;
        std::shared_ptr<problem_definition> problem_;
        std::thread thread_;
        std::atomic<bool> is_running_;
        std::atomic<bool> is_task_submitted_{false};
        std::atomic<bool> is_preparing_{false};
    };

    using thread_pool_t = std::deque<std::unique_ptr<op_thread>>;

    static std::vector<thread_pool_t> make_thread_pool() {
        std::vector<thread_pool_t> threads;
        for (int i = 0; i < sclx::cuda::traits::device_count(); ++i) {
            threads.emplace_back(thread_pool_t{});
            for (int p = 0; p < 4; ++p) {
                threads.back().emplace_back(std::make_unique<op_thread>(i));
            }
        }

        return threads;
    }

    static std::vector<std::atomic<uint>> make_thread_counters() {
        std::vector<std::atomic<uint>> counters(
            sclx::cuda::traits::device_count()
        );
        for (auto& counter : counters) {
            counter.store(0);
        }

        return counters;
    }

    static op_thread* get_next_thread(int device) {
        static auto threads  = make_thread_pool();
        static auto counters = make_thread_counters();
        auto& thread_counter = counters[device];
        auto expected        = thread_counter.load();
        auto new_value       = (expected + 1) % threads[device].size();
        while (!thread_counter.compare_exchange_weak(expected, new_value)) {
            new_value = (expected + 1) % threads[device].size();
        }
        return threads[device][expected].get();
    }
};

template class matrix_mult<
    matrix<float, storage_type::sparse_csr>,
    vector<float, storage_type::dense>,
    vector<float, storage_type::dense>>;

template class matrix_mult<
    matrix<float, storage_type::sparse_csr>,
    matrix<float, storage_type::sparse_csr>,
    matrix<float, storage_type::sparse_csr>>;

template class matrix<float, storage_type::sparse_csr>;

template class vector<float, storage_type::dense>;

}  // namespace naga::linalg
