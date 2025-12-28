# dsp-core Module

**Purpose**: Audio processing core - layered transfer functions, spline fitting, seamless audio updates

## Components

- **LayeredTransferFunction** - Four-layer waveshaping architecture (base + harmonic + spline + composite)
- **SeamlessTransferFunctionImpl** - Triple-buffered LUT with glitch-free crossfade updates
  - AudioEngine - Triple-buffered audio thread component
  - LUTRendererThread - Background LUT rendering worker
  - TransferFunctionDirtyPoller - 25Hz version change detector
- **SplineFitter, SplineEvaluator** - Curve fitting algorithms (Fritsch-Carlson PCHIP)
- **Services**:
  - CoordinateMapper - Screen ↔ transfer function coordinate conversions
  - BaseLayerSolver - Harmonic coefficient solver from target curve
  - CurveFeatureDetector - Extrema and inflection point detection

## Documentation

- [layered-transfer-function.md](docs/layered-transfer-function.md) - Four-layer architecture, normalization, table-based evaluation
- [seamless-transfer-function.md](docs/seamless-transfer-function.md) - Glitch-free updates, triple buffering, DAW lifecycle
- [spline-curve-fitting.md](docs/spline-curve-fitting.md) - Complete fitting pipeline (feature detection → greedy fitting → PCHIP tangents)
- [spline-algorithm-decision.md](docs/spline-algorithm-decision.md) - Fritsch-Carlson vs Akima comparison, no-overshoot rationale
- [spline-layer-refactoring-notes.md](docs/spline-layer-refactoring-notes.md) - SplineLayer implementation details
- [curve-fitting-enhancements-summary.md](docs/curve-fitting-enhancements-summary.md) - Optimization history and performance benchmarks

## Dependencies

- JUCE 8.x
- C++20

## Cross-Cutting Patterns

See plugin-level docs for patterns that apply across modules:
- [Audio Thread Safety](../../docs/architecture/dsp-processing.md) - Lock-free algorithms, memory ordering, pre-allocation
- [Error Handling](../../docs/architecture/error-handling-patterns.md) - Result types, jassert vs graceful degradation
- [Service Extraction](../../docs/architecture/services.md) - Service criteria, pure static class pattern

## Related Guides

- [Curve Fitting Tuning](../../docs/guides/curve-fitting-tuning.md) - Parameter tuning workflow
- [Testing](../../docs/guides/testing.md) - Running module tests
