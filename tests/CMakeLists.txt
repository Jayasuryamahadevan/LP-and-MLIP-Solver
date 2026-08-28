# tests/sparse/test_parallel.cpp explicitly calls omp_set_num_threads(N>1)
# itself (to compare N=1 vs N>1 results bit-for-bit) -- the ONE file in this
# suite whose OpenMP thread count an outer OMP_NUM_THREADS cannot control.
# Every other file's use of OpenMP goes through the ambient default thread
# count, so OMP_NUM_THREADS=1 (set for the TSan test run below) genuinely
# prevents GOMP worker-thread creation for all of them. This file is the
# one exception, excluded from the TSan build for exactly that reason (see
# the SIHPS_ENABLE_TSAN block below and docs/architecture/MILP.md section
# 6.4): GCC's stock libgomp is not itself built with -fsanitize=thread, so
# its worker-thread-pool synchronization is invisible to TSan's
# happens-before tracker, producing false-positive races on every one of
# this file's genuinely-correct (and independently, bit-identical-result-
# verified) multi-threaded OpenMP calls. Its own assertions still run and
# must still pass in every NORMAL (non-TSan) build and CI run.
set(SIHPS_TEST_SOURCES
  test_main.cpp
  bench/test_resource_snapshot.cpp
  memory/test_memory_arena.cpp
  sparse/test_csr_csc.cpp
  sparse/test_parallel.cpp
  cuda/test_cuda_device.cpp
  cuda/test_gpu_spmv.cpp
  cuda/test_gpu_pricing.cpp
  io/test_mps_reader.cpp
  lp/test_basis_factorization.cpp
  lp/test_pdlp.cpp
  lp/test_first_order_pipeline.cpp
  lp/test_presolve.cpp
  lp/test_scaling.cpp
  lp/test_simplex.cpp
  lp/test_adversarial.cpp
  milp/test_milp.cpp
  milp/test_milp_parallel.cpp
)
if(SIHPS_ENABLE_TSAN)
  list(REMOVE_ITEM SIHPS_TEST_SOURCES sparse/test_parallel.cpp)
endif()

add_executable(sihps_tests ${SIHPS_TEST_SOURCES})

target_link_libraries(sihps_tests PRIVATE sihps_core)
target_include_directories(sihps_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(sihps_tests PRIVATE SIHPS_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
sihps_set_warnings(sihps_tests)

add_test(NAME sihps_unit_tests COMMAND sihps_tests)

if(SIHPS_ENABLE_TSAN)
  # OMP_NUM_THREADS=1 for the TSan run specifically (never the normal
  # build): GCC's stock libgomp is not itself built with -fsanitize=thread,
  # so its internal worker-thread-pool barrier/join synchronization is
  # invisible to TSan's happens-before tracker -- every `#pragma omp
  # parallel` region this codebase's PRE-EXISTING OpenMP hot paths use
  # (CSRMatrix::multiply, Simplex::price_cpu, Presolve, etc.) therefore
  # reads as a false-positive race between a GOMP worker thread's write
  # and the main thread's post-join read, confirmed directly by full
  # stack-trace inspection (docs/architecture/MILP.md section 6.4). Two
  # suppression-file approaches (called_from_lib:libgomp.so, then
  # ignore_noninstrumented_modules=1) each failed to fully clear these.
  # OMP_NUM_THREADS=1 alone is NOT sufficient either -- it only prevents
  # GOMP worker-thread creation for code that reads the ambient thread
  # count; sparse/test_parallel.cpp explicitly calls
  # omp_set_num_threads(N>1) itself and is therefore excluded from this
  # binary above, which is the piece that actually makes this combination
  # complete. Neither exclusion nor OMP_NUM_THREADS=1 weakens correctness
  # coverage: the excluded file's own bit-identical-across-thread-count
  # assertions still run in every normal (non-TSan) build, and this
  # section exists to validate the NEW std::thread-based parallel-B&B code
  # (src/milp/MilpSolver.cpp, src/milp/ParallelSearch.hpp), which never
  # touches OpenMP at all.
  set_tests_properties(sihps_unit_tests PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1;TSAN_OPTIONS=halt_on_error=0:history_size=7")
endif()
