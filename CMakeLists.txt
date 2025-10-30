# DSP-Core JUCE module

# Explicitly tell JUCE where the main header is
juce_add_module(dsp_core
    HEADER_FILE "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.h"
    DEPENDS juce::juce_core
            juce::juce_data_structures
            juce::juce_audio_processors
            juce::juce_dsp
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
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/SplineTypes.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/TransferFunction.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/TransferFunction.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/CurveFeatureDetector.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/CurveFeatureDetector.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/SplineFitter.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/SplineFitter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/SplineEvaluator.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/SplineEvaluator.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/ZeroCrossingSolver.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/Services/ZeroCrossingSolver.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/AudioProcessingStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/AudioPipeline.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/AudioPipeline.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/GainStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/GainStage.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DryWetMixStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DryWetMixStage.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/WaveshapingStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/WaveshapingStage.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/OversamplingWrapper.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/OversamplingWrapper.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DCBlockingFilter.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DCBlockingFilter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DCOffsetCompensator.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DCOffsetCompensator.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/PeakEnvelopeDetector.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/BiasFadeController.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/SilenceDetector.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/SilenceDetector.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DynamicOutputBiasing.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/audio_pipeline/DynamicOutputBiasing.cpp"
)

# Add tests subdirectory if testing is enabled
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()