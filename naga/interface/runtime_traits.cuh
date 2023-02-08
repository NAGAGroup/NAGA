// Copyright 2023 Jack Myers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//
// Created by jack on 2/7/23.
//

#pragma once

namespace naga {

namespace placeholders {
struct incomplete_type;
}
template <class Runtime>
struct runtime_traits {
    template<class T>
    using device_accessor_type = placeholders::incomplete_type;

    template<class T>
    using buffer_type = placeholders::incomplete_type;
};
}
