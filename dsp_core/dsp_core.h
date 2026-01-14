/*******************************************************************************
 dsp-core Module

 Platform-independent DSP primitives and audio pipeline for TotalHarmonicControl.

 Dependencies:
 - audio::primitives - AudioBuffer
 - plugin::core - PropertyTree
 - platform::os - Timer
 - ui::primitives - Geometry (Point, Rectangle)
 - Standard C++20 - threads, atomics, containers

 As of Phase 2 completion, this module has ZERO JUCE dependencies.
*******************************************************************************/

#pragma once
#define DSP_CORE_H_INCLUDED

// Standard library
#include <cmath>
#include <vector>
#include <memory>
#include <atomic>
#include <string>

// Modular platform abstractions
#include <audio-primitives/AudioBuffer.h>
#include <plugin-core/PropertyTree.h>
#include <platform-os/Timer.h>
#include <ui-primitives/Geometry.h>

// Primitives
#include "Source/primitives/MathUtils.h"
#include "Source/primitives/Decibels.h"
#include "Source/primitives/SmoothedValue.h"
#include "Source/primitives/IIRFilter.h"
#include "Source/primitives/Gain.h"
#include "Source/primitives/Oversampling.h"
#include "Source/primitives/LockFreeFIFO.h"

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
#include "Source/services/CurveFeatureDetector.h"
#include "Source/services/SplineFitter.h"
#include "Source/services/SplineEvaluator.h"
#include "Source/services/SymmetryAnalyzer.h"
#include "Source/services/TransferFunctionOperations.h"

// Utilities (remaining in Source/)
#include "Source/ExpressionEvaluator.h"
