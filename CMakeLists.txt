# DSP-Core JUCE module

# Explicitly tell JUCE where the main header is
juce_add_module(dsp_core
    HEADER_FILE "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.h"
    DEPENDS juce::juce_core
            juce::juce_data_structures
            juce::juce_audio_processors
)

target_sources(dsp_core PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/ExpressionEvaluator.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/ExpressionEvaluator.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/HarmonicLayer.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/HarmonicLayer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/LayeredTransferFunction.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/LayeredTransferFunction.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/TransferFunction.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/TransferFunction.cpp"
)