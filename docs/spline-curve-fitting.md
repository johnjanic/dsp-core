# Spline Curve Fitting Pipeline

**Status**: ✅ Current and canonical
**Last Updated**: 2025-12-19

**Purpose**: Overview of the curve fitting algorithm - converts 16384-point LUT into compact spline anchors (10-30 points).

**For detailed implementation**: See [spline-algorithm-implementation.md](spline-algorithm-implementation.md)
**For optimization history**: See [spline-optimization-history.md](spline-optimization-history.md)

---

## Overview

The spline curve fitting system converts a rasterized transfer function (16384-point lookup table) into a compact set of spline anchors (typically 10-30 points) that can be interactively edited. The fitted spline preserves geometric features (extrema, inflection points) while minimizing approximation error.

**Key Design Goals:**
1. **Geometric Correctness** - Preserve peaks, valleys, and inflection points exactly
2. **Error Minimization** - Keep fitted curve within perceptual tolerance (< 1%)
3. **Compactness** - Minimize anchor count for interactive editing performance
4. **Monotonicity** - Prevent spurious oscillations and overshoots
5. **Real-time Performance** - Complete fitting in < 300ms for interactive workflows

---

## Pipeline Stages

```
Input: LayeredTransferFunction (16384 samples)
   ↓
Stage 1: Feature Detection (geometric analysis)
   ↓
Stage 2: Sampling & Densification (raster → polyline)
   ↓
Stage 3: Sanitization (deduplication, monotonicity)
   ↓
Stage 4: Greedy Spline Fitting (iterative refinement)
   ↓
Stage 5: Tangent Computation (PCHIP/Fritsch-Carlson)
   ↓
Output: SplineFitResult (10-30 anchors)
```

---

## Stage 1: Feature Detection

**Purpose:** Identify geometric features (extrema, inflection points) that MUST be preserved in the fitted spline.

**Implementation:** [CurveFeatureDetector.cpp:8-109](../../modules/dsp-core/dsp_core/Source/Services/CurveFeatureDetector.cpp#L8-L109)

### Algorithm (Pseudocode)

```python
def detectFeatures(ltf: LayeredTransferFunction, config: FeatureDetectionConfig) -> FeatureResult:
    """
    Detect mandatory anchor points using geometric analysis.

    Thresholds prevent detecting numerical noise as features:
    - derivativeThreshold = 1e-6 (dy/dx must be significant)
    - secondDerivativeThreshold = 0.002 (Phase 4 v3: 20× higher to filter artifact inflections)
    - enableInflectionDetection = false (Phase 4 v5: disabled by default for CPU savings)
    """
    features = []

    # 1. Find Local Extrema (dy/dx sign changes)
    for i in range(1, tableSize - 1):
        deriv_prev = estimateDerivative(ltf, i - 1)  # Central difference
        deriv = estimateDerivative(ltf, i)

        if |deriv_prev| > config.derivativeThreshold and |deriv| > config.derivativeThreshold:
            if deriv_prev * deriv < 0:  # Sign change
                y = ltf.getCompositeValue(i)
                features.append(Feature{
                    index: i,
                    significance: |y|,  # Amplitude
                    isExtremum: true
                })

    # 2. Find Inflection Points (d²y/dx² sign changes)
    # NOTE: Disabled by default (Phase 4 v5) - saves ~30-40% CPU with no quality impact
    # With secondDerivativeThreshold=0.002, effectively zero inflections detected anyway
    if config.enableInflectionDetection:
        for i in range(2, tableSize - 2):
            d2y_prev = estimateSecondDerivative(ltf, i - 1)
            d2y = estimateSecondDerivative(ltf, i)

            if |d2y_prev| > config.secondDerivativeThreshold and |d2y| > config.secondDerivativeThreshold:
                if d2y_prev * d2y < 0:  # Sign change
                    features.append(Feature{
                        index: i,
                        significance: |d2y|,  # Curvature
                        isExtremum: false
                    })

    # 3. Prioritize Features (if too many)
    mandatoryAnchors = [0, tableSize - 1]  # Always include endpoints

    if len(features) + 2 > maxAnchors:
        # Budget: 80% extrema, 20% inflections
        # (Peaks/valleys are more perceptually important than inflection points)
        maxExtrema = int((maxAnchors - 2) * 0.8)
        maxInflections = (maxAnchors - 2) - maxExtrema

        extrema = [f for f in features if f.isExtremum]
        inflections = [f for f in features if not f.isExtremum]

        # Sort by significance (descending)
        extrema.sort(key=lambda f: f.significance, reverse=True)
        inflections.sort(key=lambda f: f.significance, reverse=True)

        # Take top N most significant
        mandatoryAnchors.extend([f.index for f in extrema[:maxExtrema]])
        mandatoryAnchors.extend([f.index for f in inflections[:maxInflections]])
    else:
        # All features fit within budget
        mandatoryAnchors.extend([f.index for f in features])

    mandatoryAnchors.sort()
    return FeatureResult{mandatoryAnchors, localExtrema, inflectionPoints}
```

### Derivative Estimation

```python
def estimateDerivative(ltf: LayeredTransferFunction, idx: int) -> float:
    """Central difference (2nd order accurate)"""
    x0 = ltf.normalizeIndex(idx - 1)  # Maps to [-1, 1]
    x1 = ltf.normalizeIndex(idx + 1)
    y0 = ltf.getCompositeValue(idx - 1)
    y1 = ltf.getCompositeValue(idx + 1)
    return (y1 - y0) / (x1 - x0)

def estimateSecondDerivative(ltf: LayeredTransferFunction, idx: int) -> float:
    """3-point finite difference"""
    h = ltf.normalizeIndex(1) - ltf.normalizeIndex(0)  # Uniform spacing
    y_prev = ltf.getCompositeValue(idx - 1)
    y = ltf.getCompositeValue(idx)
    y_next = ltf.getCompositeValue(idx + 1)
    return (y_next - 2*y + y_prev) / (h * h)
```

**Key Insights:**
- Thresholds prevent noise from being treated as features
- Significance scoring prioritizes visually/sonically important features
- 70-80% of anchor budget reserved for mandatory features, 20-30% for error-driven refinement
- **Phase 4 v5 (2025-01)**: Inflection detection disabled by default - `secondDerivativeThreshold=0.002` already filters all inflections, so skipping the loop saves ~30-40% CPU with no quality impact
- **Phase 4 v4 (2025-01)**: `relativeErrorTarget` has negligible impact on anchor count (~95% from features, ~5% from greedy refinement)

---

## Unified Sampling Path

**CRITICAL DESIGN PRINCIPLE**: All curve evaluation (fitting, rendering, visualization) uses the same code path: `LayeredTransferFunction::evaluateForRendering(x, normScalar)`.

This unified path ensures:
- Fitting samples what user sees (avoids base-layer-only bugs)
- Audio processing renders exactly what fitting was based on
- Visualizer displays accurate curve preview

**Single Source of Truth**:
```cpp
// Mode-specific rendering via single entry point
double LayeredTransferFunction::evaluateForRendering(double x, double normScalar) const {
    // Routes through RenderingMode (Paint/Harmonic/Spline)
    // Each mode reads base + harmonics + normalization consistently
}
```

**Three Callers, One Path**:
- Spline fitting: `SplineFitter::sampleAndSanitize()` → `ltf.getCompositeValue(i)` → `evaluateForRendering()`
- DSP rendering: `LUTRendererThread::renderDSPLUT()` → `evaluateForRendering()`
- Visualizer: `VisualizerUpdateTimer::timerCallback()` → `evaluateForRendering()`

---

## Stage 2: Sampling & Densification

**Purpose:** Convert rasterized lookup table into a densified polyline suitable for spline fitting.

**Implementation:** [SplineFitter.cpp:82-146](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L82-L146)

### Algorithm (Pseudocode)

```python
def sampleAndSanitize(ltf: LayeredTransferFunction, config: SplineFitConfig) -> List[Sample]:
    """
    Sample composite curve and densify for better error detection.

    CRITICAL: Must sample actual curve, NOT linear interpolation!
    Linear midpoints create false errors for high-frequency curves (e.g., Harmonic 15).
    """
    tableSize = ltf.getTableSize()
    samples = []

    # 1. Sample entire composite curve via unified evaluation path
    # CRITICAL: Read from evaluateForRendering(), not base layer
    # This includes base + harmonics + normalization = what user sees
    for i in range(tableSize):
        x = ltf.normalizeIndex(i)  # Maps table index to [-1, 1]
        y = ltf.evaluateForRendering(x, normScalar)  # Unified rendering path
        samples.append(Sample{x, y})

    # 2. Densify: Add midpoint samples from actual curve
    densified = []
    for i in range(len(samples)):
        densified.append(samples[i])

        if i < len(samples) - 1:
            midX = (samples[i].x + samples[i+1].x) / 2.0

            # Find table index closest to midX
            # CRITICAL: Sample ACTUAL curve, not linear interpolation!
            tableIdx = argmin([|ltf.normalizeIndex(j) - midX| for j in range(tableSize)])
            midY = ltf.getCompositeValue(tableIdx)

            densified.append(Sample{midX, midY})

    return densified
```

**Why Actual Sampling Matters:**

```
Example: Harmonic 15 (high-frequency sine wave)

Linear Interpolation (WRONG):
   A-------M-------B    M = (A.y + B.y) / 2  ← Chord, not curve!
   |       |       |
   |       X       |    ← False error: spline matches curve, not chord
   +--------------+

Actual Sampling (CORRECT):
   A-------M-------B    M = ltf.getCompositeValue(closestIdx(midX))
   |       |       |
   +-------+-------+    ← True curve point, no false error
```

**Result:** ~32,000 samples (doubled from 16,384), capturing fine details for error analysis.

---

## Stage 3: Sanitization

**Purpose:** Clean up samples to ensure valid input for spline fitting.

**Implementation:** [SplineFitter.cpp:148-206](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L148-L206)

### Algorithm (Pseudocode)

```python
def sanitize(samples: List[Sample], config: SplineFitConfig) -> List[Sample]:
    """
    Sanitize samples for spline fitting:
    1. Sort by x (monotonic x-axis required for cubic Hermite)
    2. Deduplicate near-verticals (avoid undefined slopes)
    3. Enforce monotonicity (optional, prevents y-backtracking)
    4. Clamp to range (ensure [-1, 1] bounds)
    """

    # 1. Sort by x (should already be sorted, but guarantee it)
    samples.sort(key=lambda s: s.x)

    # 2. Deduplicate Near-Verticals
    # Problem: Multiple samples at same x → undefined derivative
    # Solution: Average y values for samples with |Δx| < 1e-6
    deduped = []
    i = 0
    while i < len(samples):
        xSum = samples[i].x
        ySum = samples[i].y
        count = 1

        # Group samples with similar x
        while i + count < len(samples) and |samples[i + count].x - samples[i].x| < 1e-6:
            xSum += samples[i + count].x
            ySum += samples[i + count].y
            count += 1

        # Average the group
        deduped.append(Sample{xSum / count, ySum / count})
        i += count

    # 3. Enforce Monotonicity (if enabled)
    # Problem: y can decrease as x increases → non-monotonic curve
    # Solution: Pool Adjacent Violators Algorithm (PAVA) - pairwise averaging
    if config.enforceMonotonicity:
        for i in range(1, len(deduped)):
            if deduped[i].y < deduped[i-1].y:
                # Violates monotonicity - average violating pairs
                avgY = (deduped[i].y + deduped[i-1].y) / 2.0
                deduped[i].y = avgY
                deduped[i-1].y = avgY

    # 4. Clamp to Range
    for s in deduped:
        s.x = clamp(s.x, -1.0, 1.0)
        s.y = clamp(s.y, -1.0, 1.0)

    return deduped
```

**Why Monotonicity?**
- Waveshaping transfer functions should be monotonic (no "folding back")
- Non-monotonic curves cause phase inversion artifacts in audio
- User can disable for creative effects (e.g., folding distortion)

---

## Stage 4: Greedy Spline Fitting with Adaptive Tolerance

**Purpose:** Iteratively refine anchor placement to minimize approximation error while preventing over-fitting.

**Implementation:** [SplineFitter.cpp:570-667](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L570-L667)

### Adaptive Tolerance (Anti-Creeping)

**Problem:** Fixed tolerance causes "anchor creeping" - refitting curves produces exponentially more anchors on each cycle.

**Solution:** Dynamic tolerance that increases with anchor density, preventing over-fitting as anchor count approaches maximum.

```python
# Compute vertical range for scale-relative tolerance
minY = min([s.y for s in samples])
maxY = max([s.y for s in samples])
verticalRange = maxY - minY

# Configure adaptive tolerance
adaptiveConfig = AdaptiveToleranceCalculator.Config()
if verticalRange > 1e-9:
    # Derive relative error target from absolute position tolerance
    adaptiveConfig.relativeErrorTarget = config.positionTolerance / verticalRange
else:
    adaptiveConfig.relativeErrorTarget = 0.01  # Fallback for flat curves

# During refinement loop:
adaptiveTolerance = AdaptiveToleranceCalculator.computeTolerance(
    verticalRange,
    currentAnchorCount,
    config.maxAnchors,
    adaptiveConfig
)

# Use config.positionTolerance as minimum floor
adaptiveTolerance = max(adaptiveTolerance, config.positionTolerance)

# Stop adding anchors when error drops below adaptive tolerance
if maxError <= adaptiveTolerance:
    break
```

**Formula:** `tolerance = baselineTolerance × (1 + anchorRatio × multiplier)`

Where:
- `baselineTolerance = relativeErrorTarget × verticalRange` (default: 0.01 = 1% of vertical range)
- `anchorRatio = currentAnchors / maxAnchors`
- `multiplier = 10.0` (optimal via comprehensive parameter sweep, see tools/tune_adaptive_tolerance.py)

**Effect:**
- At 0% capacity (0 anchors): tolerance = baseline × 1.0 (tight fitting)
- At 2.3% capacity (3/128): tolerance = baseline × 1.23 (early relaxation)
- At 50% capacity: tolerance = baseline × 6.0 (progressive relaxation)
- At 100% capacity: tolerance = baseline × 11.0 (maximum relaxation)
- Result: Backtranslation stability - refit curves converge to minimal anchor count

**Validation:** Backtranslation tests verify no anchor creeping (stability score: 5):
- UserWorkflow (identity+middle): 3 anchors → refit → 3 anchors ✓
- ArbitraryPositions: 4 anchors → refit → 4 anchors ✓
- TanhCurve: 8 anchors → refit → 7 anchors ✓ (progressive simplification)
- H10 harmonic: 18 anchors → refit → 18 anchors ✓

### Algorithm (Pseudocode)

```python
def greedySplineFit(
    samples: List[Sample],
    config: SplineFitConfig,
    ltf: LayeredTransferFunction,
    mandatoryIndices: List[int]  # From Stage 1: Feature Detection
) -> List[SplineAnchor]:
    """
    Greedy iterative fitting: start with feature anchors, add anchors at worst-fit locations.

    Key Insight: Feature-based initialization (Phase 3) + error-driven refinement (Phase 4)
    - Structural correctness: mandatory features preserved
    - Quality: error-driven placement fills gaps
    - Stability: adaptive tolerance prevents anchor creeping
    """

    # Phase 3: Initialize with feature anchors
    if len(mandatoryIndices) == 0:
        # Fallback: Use endpoints only
        anchors = [
            SplineAnchor{samples[0].x, samples[0].y, false, 0.0},
            SplineAnchor{samples[-1].x, samples[-1].y, false, 0.0}
        ]
    else:
        # Convert table indices to sample anchors
        anchors = []
        for tableIdx in mandatoryIndices:
            targetX = ltf.normalizeIndex(tableIdx)

            # Find closest sample to this x coordinate
            closestIdx = argmin([|samples[i].x - targetX| for i in range(len(samples))])
            anchors.append(SplineAnchor{
                samples[closestIdx].x,
                samples[closestIdx].y,
                false,
                0.0
            })

        anchors.sort(key=lambda a: a.x)  # Ensure sorted by x
        anchors = removeDuplicates(anchors, epsilon=1e-9)  # Remove duplicates

    # Compute vertical range for adaptive tolerance
    minY = min([s.y for s in samples])
    maxY = max([s.y for s in samples])
    verticalRange = maxY - minY

    # Configure adaptive tolerance
    # IMPORTANT: Use fixed 1% error target, NOT derived from positionTolerance
    # This decouples adaptive tolerance from absolute position tolerance
    adaptiveConfig = AdaptiveToleranceCalculator.Config()
    adaptiveConfig.relativeErrorTarget = 0.01  # Fixed 1% of vertical range
    adaptiveConfig.anchorDensityMultiplier = 10.0  # Optimal from parameter sweep

    # Phase 4: Error-driven refinement (iterative greedy algorithm)
    remainingAnchors = max(0, config.maxAnchors - len(anchors))

    for iteration in range(remainingAnchors):
        # Compute tangents for current anchor set
        computeTangents(anchors, config)

        # Find sample with highest error
        worstIdx, maxError = findWorstFitSample(samples, anchors)

        # Compute adaptive tolerance (increases with anchor density)
        # NOTE: We do NOT apply positionTolerance as a floor here.
        # The adaptive tolerance calculator is trusted to compute the right
        # tolerance based on curve complexity and anchor density.
        adaptiveTolerance = AdaptiveToleranceCalculator.computeTolerance(
            verticalRange,
            len(anchors),
            config.maxAnchors,
            adaptiveConfig
        )

        # Converged? (adaptive tolerance prevents over-fitting)
        if maxError <= adaptiveTolerance:
            break  # Good enough

        # Insert anchor at worst-fit location
        worstSample = samples[worstIdx]

        # Maintain sorted order
        insertPos = bisect_left(anchors, worstSample.x, key=lambda a: a.x)

        # Don't insert duplicates
        if any(|a.x - worstSample.x| < 1e-9 for a in anchors):
            break  # No progress possible

        anchors.insert(insertPos, SplineAnchor{
            worstSample.x,
            worstSample.y,
            false,
            0.0
        })

    # Optional: Anchor pruning (disabled by default - see Stage 4b)
    if config.enableAnchorPruning:
        # Use adaptive tolerance with multiplier for pruning threshold
        finalTolerance = AdaptiveToleranceCalculator.computeTolerance(
            verticalRange, len(anchors), config.maxAnchors, adaptiveConfig
        )
        finalTolerance = max(finalTolerance, config.positionTolerance)
        pruningTolerance = finalTolerance * config.pruningToleranceMultiplier

        pruneRedundantAnchors(anchors, samples, pruningTolerance, config)

    # Final tangent computation
    computeTangents(anchors, config)

    return anchors

def findWorstFitSample(samples: List[Sample], anchors: List[SplineAnchor]) -> (int, float):
    """Find sample with maximum absolute error vs fitted spline."""
    maxError = 0.0
    worstIdx = 0

    for i, sample in enumerate(samples):
        # Skip samples that already have anchors
        if any(|a.x - sample.x| < 1e-9 for a in anchors):
            continue

        # Evaluate PCHIP spline at this x position
        splineY = SplineEvaluator.evaluate(anchors, sample.x)

        # Compute absolute error
        error = |sample.y - splineY|

        if error > maxError:
            maxError = error
            worstIdx = i

    return worstIdx, maxError
```

**Algorithm Analysis:**

- **Time Complexity**: O(N × K × log K) where N = samples, K = anchors
  - N iterations (worst case: one anchor per sample)
  - Each iteration: K evaluations × log K search
  - Typical: K = 10-30, N = 32,000 → ~1-10ms per iteration

- **Why Greedy Works:**
  1. Feature anchors provide structural skeleton (extrema, inflections)
  2. Error-driven refinement fills gaps intelligently
  3. PCHIP evaluation = objective function (unlike RDP's line-based error)
  4. Converges quickly: most curves < 10 iterations

**Why Not RDP (Ramer-Douglas-Peucker)?**
- RDP measures error against **straight lines**
- But we fit **cubic curves** (PCHIP)
- Objective function mismatch → bowing artifacts in straight regions
- See [SplineFitter.cpp:209-312](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L209-L312) for deprecated RDP implementation

---

## Stage 4b: Anchor Pruning (Optional, Experimental)

**Purpose:** Remove redundant anchors after fitting without degrading quality.

**Status:** ⚠️ **Disabled by default** - experimental feature with known limitations.

**Implementation:** [SplineFitter.cpp:673-709](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L673-L709)

### Algorithm (Pseudocode)

```python
def pruneRedundantAnchors(
    anchors: List[SplineAnchor],
    samples: List[Sample],
    pruningTolerance: float,
    config: SplineFitConfig
):
    """
    Iteratively remove non-endpoint anchors if removal doesn't increase error.

    Strategy: Greedy pruning - test each anchor individually
    """
    if len(anchors) <= 2:
        return  # Cannot prune endpoints

    i = 1  # Start at first non-endpoint anchor
    while i < len(anchors) - 1:
        # 1. Temporarily remove anchor
        removed = anchors[i]
        anchors.pop(i)

        # 2. Recompute tangents with reduced anchor set
        computeTangents(anchors, config)

        # 3. Measure max error across ALL samples
        # CRITICAL: Check all samples, not just local region
        # (Tangent changes can affect entire curve)
        maxError = 0.0
        for sample in samples:
            splineY = SplineEvaluator.evaluate(anchors, sample.x)
            error = |sample.y - splineY|
            maxError = max(maxError, error)

        # 4. If error exceeds tolerance, restore anchor
        if maxError > pruningTolerance:
            anchors.insert(i, removed)  # Restore
            i += 1  # Move to next anchor
        # else: anchor successfully removed, check same index again
```

### Known Limitations

**⚠️ Pruning is too aggressive and can cause regression:**

1. **Discrete Sampling Problem**
   - Pruning only validates error at discrete sample points (~32,000)
   - Anchors preserving features **between** sample points can be removed
   - Continuous curve behavior is degraded even though sampled error looks good

2. **Feature Preservation**
   - Pruning doesn't know which anchors came from feature detection
   - Can remove extrema/inflection point anchors if discrete samples align well
   - Results in loss of geometric correctness

3. **Backtranslation Regression**
   - With pruning enabled, all curves prune down to 2 anchors (just endpoints)
   - Breaks "no anchor creeping" guarantees from adaptive tolerance
   - Example: H10 harmonic needs 10 anchors, pruning reduces to 2 ❌

**Test Results with Pruning Enabled:**

```
Harmonic | Expected | With Pruning | Status
---------|----------|--------------|-------
H1       | 2-3      | 2            | ⚠️ Minimal (acceptable)
H2       | 2-5      | 2            | ❌ Over-pruned
H3       | 3-7      | 2            | ❌ Over-pruned
H5       | 5-10     | 2            | ❌ Over-pruned
H10      | 10-18    | 2            | ❌ Over-pruned
```

### Why Disabled by Default

**Adaptive tolerance (Stage 4) already achieves minimal anchor counts:**
- Parabola: 3 anchors (peak + endpoints) ✓
- H2: 2 anchors (1 extremum) ✓
- H3: 3 anchors (2 extrema) ✓
- H5: 5 anchors (4 extrema) ✓
- H10: 10 anchors (9 extrema) ✓

**Pruning provides diminishing returns with high risk:**
- Potential savings: 0-2 anchors per curve
- Risk: Loss of geometric features, backtranslation instability

### Configuration

```cpp
// SplineFitConfig
bool enableAnchorPruning = false;              // Disabled by default
double pruningToleranceMultiplier = 1.5;       // Multiplier for pruning threshold
```

**When to enable:**
- Experimental workflows only
- When anchor count is more important than geometric accuracy
- NOT recommended for production use

### Potential Improvements (Future Work)

To make pruning production-ready, it would need:

1. **Continuous Error Validation**
   - Sample spline at many more points (10,000+) during pruning
   - Or: Analytical error bounds using cubic properties

2. **Feature-Aware Pruning**
   - Never prune feature-detected anchors (extrema, inflections)
   - Only prune error-driven anchors

3. **Stricter Tolerance**
   - Use higher multiplier (2.5x instead of 1.5x)
   - More conservative pruning decisions

4. **Anchor Significance Scoring**
   - Compute "importance" metric for each anchor
   - Prune least important first

---

## Stage 5: Tangent Computation

**Purpose:** Compute tangent slopes at each anchor to define cubic Hermite curves between anchors.

**Implementation:** [SplineFitter.cpp:381-488](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L381-L488) (PCHIP), [SplineFitter.cpp:502-558](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp#L502-L558) (Fritsch-Carlson)

### Algorithm: PCHIP with Overshoot Detection (Default)

```python
def computePCHIPTangentsImpl(anchors: List[SplineAnchor], config: SplineFitConfig):
    """
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) with refinements:
    1. Fritsch-Carlson base formula (weighted harmonic mean)
    2. Overshoot detection and correction (Phase 1.1)
    3. Long-segment scaling (Phase 1.2)
    """
    n = len(anchors)

    # Step 1: Compute secant slopes d_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
    secants = []
    for i in range(n - 1):
        dx = anchors[i+1].x - anchors[i].x
        if |dx| < 1e-12:
            secants.append(0.0)  # Degenerate segment
        else:
            secants.append((anchors[i+1].y - anchors[i].y) / dx)

    # Step 2: Compute initial tangents using Fritsch-Carlson rules
    for i in range(n):
        if i == 0:
            # Endpoint: one-sided derivative
            anchors[i].tangent = secants[0]
        elif i == n - 1:
            # Endpoint: one-sided derivative
            anchors[i].tangent = secants[n - 2]
        else:
            # Interior point: apply Fritsch-Carlson formula
            d_prev = secants[i - 1]
            d_next = secants[i]

            if d_prev * d_next <= 0:
                # Secants have opposite signs → local extremum
                anchors[i].tangent = 0.0
            else:
                # Weighted harmonic mean
                dx_prev = anchors[i].x - anchors[i-1].x
                dx_next = anchors[i+1].x - anchors[i].x
                w1 = 2.0 * dx_next + dx_prev
                w2 = dx_next + 2.0 * dx_prev
                anchors[i].tangent = harmonicMean(d_prev, d_next, w1, w2)

        # Enforce slope bounds (anti-aliasing)
        anchors[i].tangent = clamp(anchors[i].tangent, config.minSlope, config.maxSlope)

    # Phase 1.1: Overshoot Detection and Correction
    # Problem: PCHIP can overshoot monotonic range between anchors
    # Solution: Sample cubic at interior points, scale tangents if overshoot detected
    maxIterations = 3  # Iterative refinement
    for iteration in range(maxIterations):
        hadOvershoot = false

        for i in range(n - 1):
            yMin = min(anchors[i].y, anchors[i+1].y)
            yMax = max(anchors[i].y, anchors[i+1].y)
            tolerance = 0.001  # Allow tiny numerical error

            # Sample cubic at 5 interior points (t = 0.2, 0.4, 0.6, 0.8)
            for j in range(1, 5):
                t = j / 5.0
                x = anchors[i].x + t * (anchors[i+1].x - anchors[i].x)
                y = SplineEvaluator.evaluateSegment(anchors[i], anchors[i+1], x)

                # Does cubic overshoot endpoint range?
                if y < yMin - tolerance or y > yMax + tolerance:
                    # Scale tangents down by 30%
                    dampingFactor = 0.7
                    anchors[i].tangent *= dampingFactor
                    anchors[i+1].tangent *= dampingFactor
                    hadOvershoot = true
                    break  # Move to next segment

        if not hadOvershoot:
            break  # Converged

    # Phase 1.2: Long-Segment Scaling
    # Problem: Very long segments (sparse anchors) → oscillation
    # Solution: Scale tangents down for segments > 0.3 (15% of full range)
    longSegmentThreshold = 0.3
    for i in range(n - 1):
        segmentLength = anchors[i+1].x - anchors[i].x

        if segmentLength > longSegmentThreshold:
            # Scale factor: min(1.0, threshold / segmentLength)
            # Example: length=0.4 → factor=0.75, length=0.6 → factor=0.5
            lengthFactor = min(1.0, longSegmentThreshold / segmentLength)

            anchors[i].tangent *= lengthFactor
            if i == n - 2:
                anchors[i+1].tangent *= lengthFactor

def harmonicMean(a: float, b: float, wa: float, wb: float) -> float:
    """Weighted harmonic mean: m = (w1 + w2) / (w1/a + w2/b)"""
    if |a| < 1e-12 or |b| < 1e-12:
        return 0.0
    return (wa + wb) / (wa / a + wb / b)
```

### Algorithm: Fritsch-Carlson (No-Overshoot Guarantee)

```python
def computeFritschCarlsonTangents(anchors: List[SplineAnchor], config: SplineFitConfig):
    """
    Fritsch-Carlson monotone-preserving cubic interpolation.

    Guarantee: α² + β² ≤ 9 constraint ensures NO overshoot beyond anchor points.

    Reference: Fritsch, F. N., & Carlson, R. E. (1980).
    "Monotone Piecewise Cubic Interpolation"
    """
    n = len(anchors)
    tangents = [0.0] * n

    # 1. Compute segment slopes
    deltas = []
    for i in range(n - 1):
        dx = anchors[i+1].x - anchors[i].x
        if |dx| < 1e-12:
            deltas.append(0.0)
        else:
            deltas.append((anchors[i+1].y - anchors[i].y) / dx)

    # 2. Initial tangent estimates (weighted average)
    for i in range(1, n - 1):
        if deltas[i-1] * deltas[i] <= 0:
            # Local extremum - force horizontal tangent
            tangents[i] = 0.0
        else:
            # Weighted average (harmonic mean variant)
            w1 = 2.0 * (anchors[i+1].x - anchors[i].x) + (anchors[i].x - anchors[i-1].x)
            w2 = (anchors[i+1].x - anchors[i].x) + 2.0 * (anchors[i].x - anchors[i-1].x)
            tangents[i] = (w1 * deltas[i-1] + w2 * deltas[i]) / (w1 + w2)

    # 3. Enforce Fritsch-Carlson monotonicity constraints
    # Constraint: α² + β² ≤ 9 where α = m_i / δ_i, β = m_{i+1} / δ_i
    for i in range(n - 1):
        if |deltas[i]| < 1e-9:
            continue  # Skip flat segments

        alpha = tangents[i] / deltas[i]
        beta = tangents[i+1] / deltas[i]

        # If constraint violated, scale tangents to satisfy it
        if alpha * alpha + beta * beta > 9.0:
            tau = 3.0 / sqrt(alpha * alpha + beta * beta)
            tangents[i] = tau * alpha * deltas[i]
            tangents[i+1] = tau * beta * deltas[i]

    # 4. Boundary tangents (one-sided slopes)
    tangents[0] = deltas[0]
    tangents[n-1] = deltas[n-2]

    # 5. Apply tangents and enforce slope bounds
    for i in range(n):
        anchors[i].tangent = clamp(tangents[i], config.minSlope, config.maxSlope)
```

**Algorithm Comparison:**

| Property | PCHIP + Overshoot Detection | Fritsch-Carlson |
|----------|----------------------------|-----------------|
| Overshoot Guarantee | Empirical (iterative correction) | Mathematical (α²+β²≤9) |
| Smoothness | Slightly smoother (gentler constraints) | Slightly stiffer (stricter constraints) |
| Performance | 270ms avg | 269ms avg (identical) |
| Default | ❌ | ✅ (chosen for UI predictability) |

**Why Fritsch-Carlson?**
- No-overshoot guarantee simplifies anchor placement UI
- Users can place anchors knowing curve stays within bounds
- Performance identical to PCHIP
- See [spline-algorithm-decision.md](spline-algorithm-decision.md) for full analysis

---

## Stage 6: Evaluation (Cubic Hermite Interpolation)

**Purpose:** Evaluate fitted spline at arbitrary x positions for rendering and audio processing.

**Implementation:** [SplineEvaluator.cpp:9-194](../../modules/dsp-core/dsp_core/Source/Services/SplineEvaluator.cpp#L9-L194)

### Algorithm: Cubic Hermite Basis Functions

```python
def evaluateSegment(p0: SplineAnchor, p1: SplineAnchor, x: float) -> float:
    """
    Evaluate cubic Hermite curve between two anchors.

    Hermite interpolation formula:
    H(t) = h00(t)·y0 + h10(t)·Δx·m0 + h01(t)·y1 + h11(t)·Δx·m1

    where:
    - h00(t) = (1 + 2t)(1-t)²  [basis for y0]
    - h10(t) = t(1-t)²         [basis for tangent m0]
    - h01(t) = t²(3 - 2t)      [basis for y1]
    - h11(t) = t²(t - 1)       [basis for tangent m1]
    """

    # Normalize x to [0, 1] within segment
    dx = p1.x - p0.x
    if |dx| < 1e-12:
        return p0.y  # Degenerate segment

    t = (x - p0.x) / dx
    t = clamp(t, 0.0, 1.0)  # Safety clamp

    # Precompute powers
    t2 = t * t
    t3 = t2 * t
    omt = 1.0 - t      # (1-t)
    omt2 = omt * omt   # (1-t)²

    # Cubic Hermite basis functions
    h00 = (1.0 + 2.0 * t) * omt2    # 2t³ - 3t² + 1
    h10 = t * omt2                   # t³ - 2t² + t
    h01 = t2 * (3.0 - 2.0 * t)       # -2t³ + 3t²
    h11 = t2 * (t - 1.0)             # t³ - t²

    # Get tangent values
    m0 = p0.tangent
    m1 = p1.tangent

    # Hermite interpolation
    return h00 * p0.y + h10 * dx * m0 + h01 * p1.y + h11 * dx * m1
```

### Batch Evaluation (Optimized for Audio Processing)

```python
def evaluateBatch(
    anchors: List[SplineAnchor],
    xValues: Array[float],  # MUST be sorted
    yValues: Array[float],  # Output buffer
    count: int
):
    """
    Optimized batch evaluation for audio processing.

    Key Optimization: Incremental segment search
    - Assumes xValues are sorted (guaranteed for audio processing)
    - Eliminates 256 binary searches → single linear scan
    - Performance: ~5-10ns per sample (cached), ~40-50ns per sample (direct)
    """

    currentSegment = 0  # Start at first segment

    for i in range(count):
        x = xValues[i]

        # Advance segment while x is beyond current segment's end
        # Works because xValues are sorted (each x >= previous x)
        while currentSegment < len(anchors) - 1 and x > anchors[currentSegment + 1].x:
            currentSegment += 1

        # Handle out-of-bounds
        if x < anchors[0].x:
            yValues[i] = anchors[0].y  # Clamp to first anchor
        elif x > anchors[-1].x:
            yValues[i] = anchors[-1].y  # Clamp to last anchor
        else:
            # Evaluate segment
            yValues[i] = evaluateSegment(
                anchors[currentSegment],
                anchors[currentSegment + 1],
                x
            )
```

**Performance Analysis:**

| Method | Time per Sample | Use Case |
|--------|-----------------|----------|
| Single evaluation (binary search) | ~20-30ns | Interactive editing |
| Batch evaluation (incremental) | ~5-10ns | Audio processing (cached) |
| Direct evaluation (no cache) | ~40-50ns | During drag operations |

---

## Configuration Presets

**Implementation:** [SplineTypes.h:64-90](../../modules/dsp-core/dsp_core/Source/SplineTypes.h#L64-L90)

```python
# Tight Fit (PRODUCTION DEFAULT - Phase 4 v3 optimal)
SplineFitConfig.tight():
    positionTolerance = 0.005          # Tight error tolerance (relaxed from 0.002 for stability)
    derivativeTolerance = 0.05
    maxAnchors = 128                   # Phase 4 v3: Optimal for perfect stability + exceptional quality
    tangentAlgorithm = FritschCarlson
    enableAnchorPruning = false        # Adaptive tolerance already minimal
    pruningToleranceMultiplier = 1.5   # (not used unless enabled)
    # Phase 4 v3 results (203 configs tested):
    # - Perfect backtranslation: 3→3, 4→4
    # - Exceptional quality: 0.01% error
    # - Average anchors: ~44-54 (never hits 128 limit for typical curves)

# Smooth Fit (for simpler curves or faster UI)
SplineFitConfig.smooth():
    positionTolerance = 0.01           # Relaxed error tolerance
    derivativeTolerance = 0.02
    maxAnchors = 24                    # Limit anchor count for performance
    tangentAlgorithm = FritschCarlson
    enableAnchorPruning = false        # Adaptive tolerance already minimal
    pruningToleranceMultiplier = 1.5
    # Use when: Simple curves (H1-H5) or UI performance critical

# Monotone Preserving (legacy, more restrictive)
SplineFitConfig.monotonePreserving():
    positionTolerance = 0.001          # Very tight tolerance
    derivativeTolerance = 0.02
    maxAnchors = 32                    # Limited anchor budget
    tangentAlgorithm = FritschCarlson
    enforceMonotonicity = true         # Force monotonic y
    enableAnchorPruning = false        # Preserve monotonicity features
    # Note: tight() preferred - provides 4× more anchor budget with better stability
```

**Key Changes:**
- **2025-12-03**: Updated production default from smooth() to tight() based on Phase 4 v3 optimization
  - Tested 203 configurations to find optimal balance
  - tight() achieves perfect stability (3→3, 4→4) + exceptional quality (0.01% error)
  - maxAnchors=128 provides sufficient budget without hitting limit on typical curves
- **2025-11-10**: All presets now disable anchor pruning by default (experimental feature)
  - Adaptive tolerance (Stage 4) achieves minimal anchor counts without pruning
  - Typical anchor counts: H1=2, H2=2, H3=3, H5=5, H10=10 (matches Chebyshev extrema)
  - Backtranslation stable: refit curves converge to minimal anchor count

---

## Complete Pipeline Example

```python
# Input: LayeredTransferFunction with 16,384 samples
ltf = LayeredTransferFunction()
ltf.setBaseLayerValue(0, 0.5)  # ... user draws curve ...

# Configure fitting
config = SplineFitConfig.tight()

# Stage 1: Feature Detection
maxFeatures = int(config.maxAnchors * 0.7)  # Reserve 30% for error-driven
features = CurveFeatureDetector.detectFeatures(ltf, maxFeatures)
# Result: [0, 2048, 8192, 12288, 16383]  (endpoints + 3 extrema)

# Stage 2-3: Sampling & Sanitization
samples = SplineFitter.sampleAndSanitize(ltf, config)
# Result: ~32,000 samples (densified)

# Stage 4: Greedy Fitting
anchors = SplineFitter.greedySplineFit(samples, config, ltf, features.mandatoryAnchors)
# Result: 18 anchors (5 feature + 13 error-driven)

# Stage 5: Tangent Computation (done internally during Stage 4)
# Final result: 18 SplineAnchors with x, y, tangent values

# Output: SplineFitResult
result = SplineFitResult{
    success: true,
    anchors: [...],  # 18 anchors
    numAnchors: 18,
    maxError: 0.0018,  # < 0.2% error
    averageError: 0.0006,
    message: "Fitted 18 anchors (including 5 feature anchors), max error: 0.0018"
}

# Stage 6: Evaluation
# Interactive rendering (single point)
y = SplineEvaluator.evaluate(result.anchors, x=0.5)

# Audio processing (batch)
xBuffer = [-1.0, -0.999, -0.998, ...]  # 256 samples
yBuffer = [0.0] * 256
SplineEvaluator.evaluateBatch(result.anchors, xBuffer, yBuffer, 256)
```

---

## Performance Characteristics

**Typical Timings (measured on production curves):**

| Stage | Time | Notes |
|-------|------|-------|
| Feature Detection | ~5-10ms | O(N) scan with derivative computation |
| Sampling | ~10-20ms | O(N) composite evaluation + densification |
| Sanitization | ~5ms | O(N) sorting, deduplication |
| Greedy Fitting | ~200-250ms | O(K × N × log K), K=10-30 typical |
| Tangent Computation | ~1-5ms | O(K), included in each greedy iteration |
| **Total Pipeline** | **~250-300ms** | Acceptable for interactive workflow |

**Memory Usage:**
- Input: 16,384 samples × 8 bytes = 128 KB
- Densified: 32,000 samples × 16 bytes = 512 KB
- Output: 10-30 anchors × 32 bytes = ~1 KB (reduced from previous 30-50 anchors)

**Anchor Count Improvements (with adaptive tolerance):**
- Simple curves (H1-H2): 2 anchors (previously 3-5)
- Medium curves (H3-H5): 3-5 anchors (previously 8-15)
- Complex curves (H10): 10 anchors (previously 20-30)
- Backtranslation stable: refit converges to same count

---

## Optimization History

The current configuration is the result of extensive parameter tuning (Phase 4 v3/v4/v5, 203 configurations tested).

**Key Results**:
- ✅ Perfect backtranslation stability (3→3, 4→4)
- ✅ 0.01% error on all harmonics
- ✅ 30-40% CPU improvement (inflection detection disabled)
- ✅ Anchor counts match geometric complexity

**For complete optimization history**: See [spline-optimization-history.md](spline-optimization-history.md)

---

## Related Documentation

### Algorithm Documentation
- [spline-algorithm-implementation.md](spline-algorithm-implementation.md) - Detailed stage-by-stage implementation
- [spline-optimization-history.md](spline-optimization-history.md) - Phase 4 v3/v4/v5 parameter tuning history
- [spline-algorithm-decision.md](spline-algorithm-decision.md) - Why Fritsch-Carlson was chosen over Akima

### Architecture Integration
- [services.md](../../../docs/architecture/services.md) - Service layer architecture (SplineFitter, SplineEvaluator, CurveFeatureDetector)
- [layered-transfer-function.md](layered-transfer-function.md) - SplineLayer architecture and thread safety
- [mvc-patterns.md](../../../docs/architecture/mvc-patterns.md) - Controller integration and command pattern
- [undo-redo.md](../../transfer_function_editor/docs/undo-redo.md) - Mode transitions with curve fitting integration

### Implementation Reference
- [SplineFitter.cpp](../../modules/dsp-core/dsp_core/Source/Services/SplineFitter.cpp) - Complete implementation
- [SplineEvaluator.cpp](../../modules/dsp-core/dsp_core/Source/Services/SplineEvaluator.cpp) - Cubic Hermite evaluation
- [CurveFeatureDetector.cpp](../../modules/dsp-core/dsp_core/Source/Services/CurveFeatureDetector.cpp) - Geometric feature detection
