add_executable(sihps_tests
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
)

target_link_libraries(sihps_tests PRIVATE sihps_core)
target_include_directories(sihps_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(sihps_tests PRIVATE SIHPS_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
sihps_set_warnings(sihps_tests)

add_test(NAME sihps_unit_tests COMMAND sihps_tests)
