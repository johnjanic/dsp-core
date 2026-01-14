# DSP-Core Module
# Platform-independent DSP primitives and audio pipeline.
# As of Phase 2 completion, this module has ZERO JUCE dependencies.

add_library(dsp_core STATIC
    # Unity build - dsp_core.cpp includes all implementation files
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.cpp"
)
add_library(dsp::core ALIAS dsp_core)

# Platform module dependency (provides AudioBuffer, PropertyTree, Timer, Geometry)
target_link_libraries(dsp_core
    PUBLIC
        platform::platform
)

# Include directories
target_include_directories(dsp_core
    PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}"
        "${CMAKE_SOURCE_DIR}/modules/plugin-platform"
        "${CMAKE_SOURCE_DIR}/libs/exprtk"
)

# C++20 required
target_compile_features(dsp_core PUBLIC cxx_std_20)

# Add tests subdirectory if testing is enabled
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
