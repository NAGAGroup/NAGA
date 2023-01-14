#!/bin/sh
find . -regex './include/.*\.\(cuh\|cu\)' -exec clang-format -style=file -i {} \;
find . -regex './tests/.*\.\(cuh\|cu\)' -exec clang-format -style=file -i {} \;
cmake-format CMakeLists.txt -o CMakeLists.txt
cmake-format tests/CMakeLists.txt -o tests/CMakeLists.txt