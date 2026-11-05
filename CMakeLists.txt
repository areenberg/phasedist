# The compiled kernels of the uniformization class (src/phasedist/unif.py),
# built by scikit-build-core with nanobind. On Python 3.12 and later the module
# uses the stable ABI, so one wheel per platform serves every later version.
cmake_minimum_required(VERSION 3.18...3.31)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})
find_package(nanobind CONFIG REQUIRED)

# NOMINSIZE keeps the -O3 of a release build (nanobind would otherwise
# optimize the module for size, about a quarter slower here). No fast-math:
# the summation order of the kernels is part of their accuracy
nanobind_add_module(_unifkernel STABLE_ABI NB_STATIC NOMINSIZE src/phasedist/_unifkernel.cpp)

install(TARGETS _unifkernel LIBRARY DESTINATION phasedist)
