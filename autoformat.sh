#!/bin/sh
find . -regex './packages/.*\.\(cuh\|cu\)' -exec clang-format -style=file -i {} \;