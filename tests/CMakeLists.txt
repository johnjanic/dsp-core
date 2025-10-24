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
