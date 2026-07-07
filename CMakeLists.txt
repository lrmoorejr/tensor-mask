# Copyright 2026 L. Richard Moore Jr.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)
project(tensor-mask VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

find_package(Catch2 3 REQUIRED)

# TensorMask describes its parameter space with a Flattener<>, so this repo depends on
# it as a git submodule pinned to a specific commit (see flattener/,
# https://github.com/lrmoorejr/flattener). Only its include path is needed here --
# add_subdirectory(flattener) would also pull in its own Flattener-test target/
# enable_testing(), which would collide with a parent build (e.g. commons) that
# already provides one.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/flattener)

add_executable(TensorMask-test TensorMask-test.cpp TensorMask.hpp)
target_link_libraries(TensorMask-test PRIVATE Catch2::Catch2WithMain)

enable_testing()
include(Catch)
catch_discover_tests(TensorMask-test)
