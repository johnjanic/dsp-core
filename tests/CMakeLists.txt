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
