#ifdef DSP_CORE_H_INCLUDED
/* When you add this cpp file to your project, you mustn't include it in a file where you've
   already included any other headers - just put it inside a file on its own, possibly with your config
   flags preceding it, but don't include anything else. That also includes avoiding any automatic prefix
   header files that the compiler may be using.
*/
#error "Incorrect use of JUCE cpp file"
#endif

#include "dsp_core.h"

// Structures
#include "Source/structures/AudioHistoryBuffer.cpp"

// Pipeline
#include "Source/pipeline/AudioPipeline.cpp"
#include "Source/pipeline/AudioPipelineBuilder.cpp"
#include "Source/pipeline/GainStage.cpp"
#include "Source/pipeline/DryWetMixStage.cpp"
#include "Source/pipeline/WaveshapingStage.cpp"
#include "Source/pipeline/OversamplingWrapper.cpp"
#include "Source/pipeline/DCBlockingFilter.cpp"

// Model
#include "Source/model/TransferFunction.cpp"
#include "Source/model/HarmonicLayer.cpp"
#include "Source/model/SplineLayer.cpp"
#include "Source/model/LayeredTransferFunction.cpp"

// Engine
#include "Source/engine/SeamlessTransferFunction.cpp"
#include "Source/engine/SeamlessTransferFunctionImpl.cpp"

// Services
#include "Source/services/AdaptiveToleranceCalculator.cpp"
#include "Source/services/CoordinateSnapper.cpp"
#include "Source/services/CurveFeatureDetector.cpp"
#include "Source/services/SplineFitter.cpp"
#include "Source/services/SplineEvaluator.cpp"
#include "Source/services/SymmetryAnalyzer.cpp"
#include "Source/services/TransferFunctionOperations.cpp"

// Utilities
#include "Source/ExpressionEvaluator.cpp"
