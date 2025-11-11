# DSP Core - Unit Tests

# Enable testing
enable_testing()

# Create test executable
add_executable(layered_transfer_function_tests
    LayeredTransferFunctionTest.cpp
)

# Sets the necessary include directories
target_include_directories(layered_transfer_function_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Link against dsp_core module and gtest_main
target_link_libraries(layered_transfer_function_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

# Adds googletest-specific CMake commands
include(GoogleTest)

# Add all tests defined with googletest to the CMake metadata
# Note: gtest_discover_tests may not work in all build configurations (e.g., CPM external projects)
# so we also add a simple test target that can be run directly
if(CMAKE_GENERATOR STREQUAL Xcode)
  # On macOS arm64, all binaries have to be signed before running
  gtest_discover_tests(layered_transfer_function_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(layered_transfer_function_tests)
endif()

# Also add a simple test that can always be run
add_test(NAME LayeredTransferFunctionTests COMMAND layered_transfer_function_tests)

# DryWetMixStage tests
add_executable(dry_wet_mix_stage_tests
    DryWetMixStageTest.cpp
)

target_include_directories(dry_wet_mix_stage_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dry_wet_mix_stage_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(dry_wet_mix_stage_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(dry_wet_mix_stage_tests)
endif()

add_test(NAME DryWetMixStageTests COMMAND dry_wet_mix_stage_tests)

# SplineFitter tests
add_executable(spline_fitter_tests
    SplineFitterTest.cpp
)

target_include_directories(spline_fitter_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_fitter_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_fitter_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_fitter_tests)
endif()

add_test(NAME SplineFitterTests COMMAND spline_fitter_tests)

# SplineEvaluator tests
add_executable(spline_evaluator_tests
    SplineEvaluatorTest.cpp
)

target_include_directories(spline_evaluator_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_evaluator_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_evaluator_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_evaluator_tests)
endif()

add_test(NAME SplineEvaluatorTests COMMAND spline_evaluator_tests)

# DCBlockingFilter tests
add_executable(dc_blocking_filter_tests
    DCBlockingFilterTest.cpp
)

target_include_directories(dc_blocking_filter_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dc_blocking_filter_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(dc_blocking_filter_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(dc_blocking_filter_tests)
endif()

add_test(NAME DCBlockingFilterTests COMMAND dc_blocking_filter_tests)

# DynamicOutputBiasing tests
add_executable(dynamic_output_biasing_tests
    DynamicOutputBiasingTest.cpp
)

target_include_directories(dynamic_output_biasing_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dynamic_output_biasing_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(dynamic_output_biasing_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(dynamic_output_biasing_tests)
endif()

add_test(NAME DynamicOutputBiasingTests COMMAND dynamic_output_biasing_tests)

# ZeroCrossingSolver tests
add_executable(zero_crossing_solver_tests
    ZeroCrossingSolverTests.cpp
)

target_include_directories(zero_crossing_solver_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(zero_crossing_solver_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(zero_crossing_solver_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(zero_crossing_solver_tests)
endif()

add_test(NAME ZeroCrossingSolverTests COMMAND zero_crossing_solver_tests)

# DCOffsetCompensator tests
add_executable(dc_offset_compensator_tests
    DCOffsetCompensatorTests.cpp
)

target_include_directories(dc_offset_compensator_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(dc_offset_compensator_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(dc_offset_compensator_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(dc_offset_compensator_tests)
endif()

add_test(NAME DCOffsetCompensatorTests COMMAND dc_offset_compensator_tests)

# CurveFeatureDetector tests
add_executable(curve_feature_detector_tests
    CurveFeatureDetectorTest.cpp
)

target_include_directories(curve_feature_detector_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(curve_feature_detector_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(curve_feature_detector_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(curve_feature_detector_tests)
endif()

add_test(NAME CurveFeatureDetectorTests COMMAND curve_feature_detector_tests)

# SplineLayer tests
add_executable(spline_layer_tests
    SplineLayerTest.cpp
)

target_include_directories(spline_layer_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_layer_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_layer_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_layer_tests)
endif()

add_test(NAME SplineLayerTests COMMAND spline_layer_tests)

# LayeredTransferFunctionSpline tests
add_executable(layered_transfer_function_spline_tests
    LayeredTransferFunctionSplineTest.cpp
)

target_include_directories(layered_transfer_function_spline_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(layered_transfer_function_spline_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(layered_transfer_function_spline_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(layered_transfer_function_spline_tests)
endif()

add_test(NAME LayeredTransferFunctionSplineTests COMMAND layered_transfer_function_spline_tests)

# Algorithm Benchmark tests
add_executable(algorithm_benchmark_tests
    AlgorithmBenchmarkTest.cpp
)

target_include_directories(algorithm_benchmark_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(algorithm_benchmark_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(algorithm_benchmark_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(algorithm_benchmark_tests)
endif()

add_test(NAME AlgorithmBenchmarkTests COMMAND algorithm_benchmark_tests)

# CoordinateSnapper tests
add_executable(coordinate_snapper_tests
    CoordinateSnapperTest.cpp
)

target_include_directories(coordinate_snapper_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(coordinate_snapper_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(coordinate_snapper_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(coordinate_snapper_tests)
endif()

add_test(NAME CoordinateSnapperTests COMMAND coordinate_snapper_tests)

# AdaptiveTolerance tests
add_executable(adaptive_tolerance_tests
    AdaptiveToleranceTest.cpp
)

target_include_directories(adaptive_tolerance_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(adaptive_tolerance_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(adaptive_tolerance_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(adaptive_tolerance_tests)
endif()

add_test(NAME AdaptiveToleranceTests COMMAND adaptive_tolerance_tests)

# SymmetryAnalyzer tests
add_executable(symmetry_analyzer_tests
    SymmetryAnalyzerTest.cpp
)

target_include_directories(symmetry_analyzer_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(symmetry_analyzer_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(symmetry_analyzer_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(symmetry_analyzer_tests)
endif()

add_test(NAME SymmetryAnalyzerTests COMMAND symmetry_analyzer_tests)

# SplineFitterIntegration tests
add_executable(spline_fitter_integration_tests
    SplineFitterIntegrationTest.cpp
)

target_include_directories(spline_fitter_integration_tests
    PRIVATE
    ${GOOGLETEST_SOURCE_DIR}/googletest/include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(spline_fitter_integration_tests
    PRIVATE
    dsp_core
    GTest::gtest_main
)

if(CMAKE_GENERATOR STREQUAL Xcode)
  gtest_discover_tests(spline_fitter_integration_tests DISCOVERY_MODE PRE_TEST)
else()
  gtest_discover_tests(spline_fitter_integration_tests)
endif()

add_test(NAME SplineFitterIntegrationTests COMMAND spline_fitter_integration_tests)
