cmake_minimum_required(VERSION 3.24)

# Must be set before project()/enable_language(CUDA): CMake populates
# CMAKE_CUDA_ARCHITECTURES with its own compiler-chosen default as part of
# CUDA language enablement, so a guard placed after project() never fires
# -- the variable is already non-empty by then. Verified on this machine:
# CMake 4.2.3 + CUDA 13.3 defaulted to sm_75 (Turing) despite the actual
# dev GPU being an sm_89 RTX 3050 Laptop (Ada), producing a fatbin the
# driver rejects at kernel-launch time ("PTX compiled with an unsupported
# toolchain") instead of a build-time mismatch, so it went undetected by
# compilation alone. CC 8.9 measured directly via a device-query smoke
# test -- see docs/architecture/CPU_GPU.md.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(sihps_solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SIHPS_BUILD_TESTS "Build unit tests" ON)
option(SIHPS_BUILD_BENCHMARKS "Build benchmarks" ON)

find_package(CUDAToolkit REQUIRED)

# Optional by design. Every parallel loop in this codebase is written so
# that dropping the pragmas changes nothing but the wall clock
# (src/parallel/Parallel.hpp), so a toolchain without OpenMP still builds
# and still produces bit-identical results.
find_package(OpenMP)

# Resolved once at configure time. A benchmark record that cannot name the
# commit that produced it is not reproducible, and this project has already
# published three sets of numbers it could not later attribute.
find_package(Git QUIET)
set(SIHPS_GIT_COMMIT "unknown")
set(SIHPS_GIT_DIRTY "unknown")
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SIHPS_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} status --porcelain
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE SIHPS_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(SIHPS_GIT_STATUS STREQUAL "")
    set(SIHPS_GIT_DIRTY "clean")
  else()
    set(SIHPS_GIT_DIRTY "dirty")
  endif()
  if(SIHPS_GIT_COMMIT STREQUAL "")
    set(SIHPS_GIT_COMMIT "unknown")
  endif()
endif()
message(STATUS "SIHPS build stamp: ${SIHPS_GIT_COMMIT} (${SIHPS_GIT_DIRTY})")

include(cmake/CompilerWarnings.cmake)

add_subdirectory(src)

if(SIHPS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(SIHPS_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()
