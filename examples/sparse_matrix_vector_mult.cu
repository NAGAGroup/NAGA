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

#include <naga/linalg/matrix.cuh>

int main() {
    const int A_num_rows_cols = 4;
    const int A_nnz           = 9;
    int hA_csrOffsets[A_num_rows_cols + 1]       = {0, 3, 4, 7, 9};
    int hA_columns[A_nnz]          = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    float hA_values[A_nnz] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float hX[A_num_rows_cols]        = {1.0f, 2.0f, 3.0f, 4.0f};
    float hY[A_num_rows_cols]        = {0.0f, 0.0f, 0.0f, 0.0f};
    float hY_result[A_num_rows_cols] = {19.0f, 8.0f, 51.0f, 52.0f};

    sclx::array<int, 1> csr_offsets(sclx::shape_t<1>{A_num_rows_cols + 1}, hA_csrOffsets);
    sclx::array<int, 1> columns(sclx::shape_t<1>{A_nnz}, hA_columns);
    sclx::array<float, 1> values(sclx::shape_t<1>{A_nnz}, hA_values);
    naga::linalg::matrix<float, naga::linalg::storage_type::sparse_csr>
        A(A_num_rows_cols, values, csr_offsets, columns);

    sclx::array<float, 1> X_array(sclx::shape_t<1>{A_num_rows_cols}, hX);
    sclx::array<float, 1> Y_array(sclx::shape_t<1>{A_num_rows_cols}, hY);
    sclx::array<float, 1> Y_expected(sclx::shape_t<1>{A_num_rows_cols}, hY_result);
    naga::linalg::vector<float, naga::linalg::storage_type::dense> X(
        X_array
    );
    naga::linalg::vector<float, naga::linalg::storage_type::dense> Y(
        Y_array
    );

    naga::linalg::matrix_mult dot_op;

    dot_op(A, X, Y);

    for (size_t i = 0; i < Y.values().elements(); ++i) {
        if (Y.values()[i] != Y_expected[i]) {
            std::cout << "Y[" << i << "] = " << Y.values()[i] << " != "
                      << Y_expected[i] << " = Y_expected[" << i
                      << "]\n";
        }
    }

    return 0;
}
