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
  milp/test_milp_parallel.cpp
)

target_link_libraries(sihps_tests PRIVATE sihps_core)
target_include_directories(sihps_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(sihps_tests PRIVATE SIHPS_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
sihps_set_warnings(sihps_tests)

add_test(NAME sihps_unit_tests COMMAND sihps_tests)

if(SIHPS_ENABLE_TSAN)
  # Scoped, investigated suppression for a known ThreadSanitizer +
  # GCC-libgomp limitation, not a blanket silencer -- see
  # tools/tsan_suppressions.txt and docs/architecture/MILP.md section 6.4
  # for the stack-trace evidence behind this specific entry.
  set_tests_properties(sihps_unit_tests PROPERTIES
    ENVIRONMENT "TSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/tools/tsan_suppressions.txt:halt_on_error=0:history_size=7")
endif()
