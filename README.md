# dsp-core Module

Platform-independent DSP primitives and audio pipeline for TotalHarmonicControl.

## Overview

**Purpose**: Audio processing core - layered transfer functions, spline fitting, seamless audio updates

**As of Phase 2 completion, this module has ZERO JUCE dependencies.**

## Dependencies

- **platform::platform** - AudioBuffer, PropertyTree, Timer, Geometry
- **Standard C++20** - threads, atomics, containers

## Components

### Primitives (Source/primitives/)
- `Decibels.h` - dB conversion utilities
- `SmoothedValue.h` - Linear parameter smoothing
- `IIRFilter.h` - Biquad IIR filter
- `Gain.h` - Gain processor with smoothing
- `Oversampling.h` - Half-band polyphase oversampling
- `MathUtils.h` - Math constants and utilities
- `LockFreeFIFO.h` - Lock-free SPSC queue, triple buffer, and AbstractFIFO

### Pipeline (Source/pipeline/)
- Audio processing stage interface and implementations
- Gain, dry/wet mix, DC blocking, waveshaping stages
- Oversampling wrapper

### Model (Source/model/)
- **LayeredTransferFunction** - Four-layer waveshaping architecture (base + harmonic + spline + composite)
- Transfer function representation
- Spline and harmonic layers
- PropertyTree serialization

### Engine (Source/engine/)
- **SeamlessTransferFunctionImpl** - Triple-buffered LUT with glitch-free crossfade updates
  - AudioEngine - Triple-buffered audio thread component
  - LUTRendererThread - Background LUT rendering worker
  - TransferFunctionDirtyPoller - 25Hz version change detector
- Thread-safe triple-buffered rendering

### Services (Source/services/)
- **SplineFitter, SplineEvaluator** - Curve fitting algorithms (Fritsch-Carlson PCHIP)
- CoordinateSnapper - Coordinate snapping for UI
- CurveFeatureDetector - Extrema and inflection point detection
- SymmetryAnalyzer - Curve symmetry analysis

## Documentation

- [layered-transfer-function.md](docs/layered-transfer-function.md) - Four-layer architecture, normalization, table-based evaluation
- [seamless-transfer-function.md](docs/seamless-transfer-function.md) - Glitch-free updates, triple buffering, DAW lifecycle
- [spline-curve-fitting.md](docs/spline-curve-fitting.md) - Complete fitting pipeline (feature detection → greedy fitting → PCHIP tangents)
- [spline-algorithm-decision.md](docs/spline-algorithm-decision.md) - Fritsch-Carlson vs Akima comparison, no-overshoot rationale
- [spline-layer-refactoring-notes.md](docs/spline-layer-refactoring-notes.md) - SplineLayer implementation details
- [curve-fitting-enhancements-summary.md](docs/curve-fitting-enhancements-summary.md) - Optimization history and performance benchmarks

## Cross-Cutting Patterns

See plugin-level docs for patterns that apply across modules:
- [Audio Thread Safety](../../docs/architecture/dsp-processing.md) - Lock-free algorithms, memory ordering, pre-allocation
- [Error Handling](../../docs/architecture/error-handling-patterns.md) - Result types, jassert vs graceful degradation
- [Service Extraction](../../docs/architecture/services.md) - Service criteria, pure static class pattern

## Related Guides

- [Curve Fitting Tuning](../../docs/guides/curve-fitting-tuning.md) - Parameter tuning workflow
- [Testing](../../docs/guides/testing.md) - Running module tests
