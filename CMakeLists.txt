# DSP-Core JUCE module

# Explicitly tell JUCE where the main header is
juce_add_module(dsp_core
    HEADER_FILE "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.h"
    DEPENDS juce::juce_core
            juce::juce_data_structures
            juce::juce_audio_processors
            juce::juce_dsp
)

# Add exprtk include path (header-only library managed by CPM in root CMakeLists.txt)
target_include_directories(dsp_core INTERFACE "${CMAKE_SOURCE_DIR}/libs/exprtk")

target_sources(dsp_core PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/dsp_core.h"
    # Structures
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/structures/AudioHistoryBuffer.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/structures/AudioHistoryBuffer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/structures/AudioInputBuffer.h"
    # Pipeline
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/AudioProcessingStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/AudioPipeline.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/AudioPipeline.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/AudioPipelineBuilder.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/AudioPipelineBuilder.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/GainStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/GainStage.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/DryWetMixStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/DryWetMixStage.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/WaveshapingStage.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/WaveshapingStage.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/OversamplingWrapper.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/OversamplingWrapper.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/DCBlockingFilter.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/DCBlockingFilter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/AudioInputWriter.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/InputPeakTracker.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/pipeline/StageHandles.h"
    # Model
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/SplineTypes.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/TransferFunction.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/TransferFunction.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/HarmonicLayer.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/HarmonicLayer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/SplineLayer.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/SplineLayer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/LayeredTransferFunction.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/model/LayeredTransferFunction.cpp"
    # Engine
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/engine/SeamlessTransferFunction.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/engine/SeamlessTransferFunction.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/engine/SeamlessTransferFunctionImpl.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/engine/SeamlessTransferFunctionImpl.cpp"
    # Services
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/AdaptiveToleranceCalculator.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/AdaptiveToleranceCalculator.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/CoordinateSnapper.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/CoordinateSnapper.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/CurveFeatureDetector.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/CurveFeatureDetector.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/SplineFitter.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/SplineFitter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/SplineEvaluator.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/SplineEvaluator.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/SymmetryAnalyzer.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/SymmetryAnalyzer.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/TransferFunctionOperations.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/services/TransferFunctionOperations.cpp"
    # Utilities
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/ExpressionEvaluator.h"
    "${CMAKE_CURRENT_LIST_DIR}/dsp_core/Source/ExpressionEvaluator.cpp"
)

# Add tests subdirectory if testing is enabled
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()