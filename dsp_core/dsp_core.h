/*******************************************************************************
 The block below describes the properties of this module, and is read by
 the Projucer to automatically generate project code that uses it.
 For details about the syntax and how to create or use a module, see the
 JUCE Module Format.md file.


 BEGIN_JUCE_MODULE_DECLARATION

  ID:                 dsp_core
  vendor:             tbd
  version:            1.0.0
  name:               DSP Core
  description:        A module for creating and editing transfer functions in audio applications.
  website:            tbd
  license:            proprietary/commercial
  minimumCppStandard: 17

  dependencies:       juce_core, juce_data_structures, juce_audio_processors, juce_dsp

 END_JUCE_MODULE_DECLARATION

*******************************************************************************/

#pragma once
#define DSP_CORE_H_INCLUDED

#include <juce_data_structures/juce_data_structures.h>
#include <juce_dsp/juce_dsp.h>

// Primitives
#include "Source/primitives/MathUtils.h"
#include "Source/primitives/Decibels.h"
#include "Source/primitives/SmoothedValue.h"
#include "Source/primitives/IIRFilter.h"
#include "Source/primitives/Gain.h"
#include "Source/primitives/Oversampling.h"

// Structures
#include "Source/structures/AudioHistoryBuffer.h"
#include "Source/structures/AudioInputBuffer.h"

// Pipeline
#include "Source/pipeline/AudioProcessingStage.h"
#include "Source/pipeline/AudioPipeline.h"
#include "Source/pipeline/AudioPipelineBuilder.h"
#include "Source/pipeline/GainStage.h"
#include "Source/pipeline/DryWetMixStage.h"
#include "Source/pipeline/WaveshapingStage.h"
#include "Source/pipeline/OversamplingWrapper.h"
#include "Source/pipeline/DCBlockingFilter.h"
#include "Source/pipeline/AudioInputWriter.h"
#include "Source/pipeline/InputPeakTracker.h"
#include "Source/pipeline/StageHandles.h"

// Model
#include "Source/model/SplineTypes.h"
#include "Source/model/TransferFunction.h"
#include "Source/model/HarmonicLayer.h"
#include "Source/model/SplineLayer.h"
#include "Source/model/LayeredTransferFunction.h"

// Engine
#include "Source/engine/SeamlessTransferFunction.h"
#include "Source/engine/SeamlessTransferFunctionImpl.h"

// Services
#include "Source/services/AdaptiveToleranceCalculator.h"
#include "Source/services/CoordinateSnapper.h"
#include "Source/services/CurveFeatureDetector.h"
#include "Source/services/SplineFitter.h"
#include "Source/services/SplineEvaluator.h"
#include "Source/services/SymmetryAnalyzer.h"
#include "Source/services/TransferFunctionOperations.h"

// Utilities (remaining in Source/)
#include "Source/ExpressionEvaluator.h"
