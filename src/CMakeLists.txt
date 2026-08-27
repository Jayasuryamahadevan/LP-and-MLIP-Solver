add_library(sihps_core STATIC
  bench/RunMetadata.cpp
  bench/ResourceSnapshot.cpp
  memory/MemoryArena.cpp
  sparse/CSRMatrix.cpp
  sparse/CSCMatrix.cpp
  sparse/Convert.cpp
  cuda/CudaStream.cpp
  cuda/CudaEvent.cpp
  cuda/CudaDevice.cpp
  cuda/CusparseHandle.cpp
  cuda/GpuSpMV.cpp
  cuda/GpuPricer.cpp
  cuda/PricingKernels.cu
  cuda/PdlpKernels.cu
  cuda/GpuPdlp.cpp
  io/MpsReader.cpp
  io/NetlibReference.cpp
  lp/BasisFactorization.cpp
  lp/LpProblem.cpp
  lp/LpSolver.cpp
  lp/Presolve.cpp
  lp/Scaling.cpp
  lp/Simplex.cpp
  milp/MilpProblem.cpp
  milp/MilpSolver.cpp
)

target_include_directories(sihps_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(sihps_core PUBLIC cxx_std_17)
target_link_libraries(sihps_core PUBLIC CUDA::cudart CUDA::cusparse)
if(OpenMP_CXX_FOUND)
  target_link_libraries(sihps_core PUBLIC OpenMP::OpenMP_CXX)
endif()
# Reproducibility stamps for benchmark records
# (CLAUDE_OPUS_SOLVER_ROADMAP.md Phase 0). Configured at CMake time rather
# than compiled in from a script so that a record can always be traced back
# to a commit; "unknown" is emitted honestly when git is unavailable.
target_compile_definitions(sihps_core PRIVATE
  SIHPS_GIT_COMMIT="${SIHPS_GIT_COMMIT}"
  SIHPS_GIT_DIRTY="${SIHPS_GIT_DIRTY}"
  SIHPS_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  SIHPS_ENABLE_CUDA=1
)

sihps_set_warnings(sihps_core)
