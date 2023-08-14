
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

#include <fstream>
#include <scalix/filesystem.hpp>
#include <thread>
#include <mutex>
#include <iostream>
#include <atomic>

sclx::filesystem::path get_examples_dir() {
    return sclx::filesystem::path(__FILE__).parent_path();
}

sclx::filesystem::path get_examples_results_dir() {
#ifndef NAGA_EXAMPLES_RESULTS_DIR
    return get_examples_dir() / "results";
#else
    return {NAGA_EXAMPLES_RESULTS_DIR};
#endif
}

sclx::filesystem::path get_resources_dir() {
    return sclx::filesystem::path(NAGA_RESOURCES_DIR);
}

class async_time_printer {
  public:
    using value_type = double;
    async_time_printer() {
        thread_ = std::thread([&]() {
            is_running_ = true;
            while (!is_finished_) {
                std::lock_guard<std::mutex> lock(mutex_);
                if (is_next_value_) {
                    std::cout << "\rTime: " << next_value_ << " s"
                              << std::flush;
                    is_next_value_ = false;
                } else {
                    std::this_thread::yield();
                }
            }
            is_running_ = false;
        });
    }

    void print_next_value(const value_type& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        next_value_    = value;
        is_next_value_ = true;
    }

    ~async_time_printer() {
        is_finished_ = true;
        while (is_running_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        std::cout << "\n";
        thread_.join();
    }

  private:
    std::thread thread_;
    bool is_next_value_ = false;
    std::mutex mutex_;
    std::atomic<bool> is_running_  = false;
    std::atomic<bool> is_finished_ = false;
    value_type next_value_{};
};