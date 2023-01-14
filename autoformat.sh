#!/bin/sh
find . -regex './include/.*\.\(cuh\|cu\)' -exec clang-format -style=file -i {} \;
find . -regex './tests/.*\.\(cuh\|cu\)' -exec clang-format -style=file -i {} \;