#include "SplineFitter.h"
#include "SplineEvaluator.h"
#include "CurveFeatureDetector.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {
namespace Services {

//==============================================================================
// Main API
//==============================================================================

SplineFitResult SplineFitter::fitCurve(
    const LayeredTransferFunction& ltf,
    const SplineFitConfig& config) {

    SplineFitResult result;

    // Step 0: FEATURE-BASED ANCHOR PLACEMENT (Phase 3)
    // Detect and anchor at geometric features (structural correctness)
    // Limit mandatory features to 70% of maxAnchors, reserving 30% for error-driven refinement
    int maxFeatures = static_cast<int>(config.maxAnchors * 0.7);
    auto features = CurveFeatureDetector::detectFeatures(ltf, maxFeatures);
    std::vector<int> mandatoryIndices = features.mandatoryAnchors;

    // Step 1: Sample & sanitize
    auto samples = sampleAndSanitize(ltf, config);

    if (samples.empty()) {
        result.success = false;
        result.message = "No samples to fit";
        return result;
    }

    if (samples.size() < 2) {
        result.success = false;
        result.message = "Need at least 2 samples";
        return result;
    }

    // Step 2: Greedy spline fitting with feature-based initialization
    // Start with mandatory feature anchors, then iteratively refine (error-driven)
    auto anchors = greedySplineFit(samples, config, &ltf, mandatoryIndices);

    if (anchors.empty()) {
        result.success = false;
        result.message = "Greedy fit failed to produce anchors";
        return result;
    }

    // Note: No need for endpoint truncation fix - greedy algorithm
    // naturally respects maxAnchors limit during iteration

    // Step 3: Error analysis
    // Compute error statistics by comparing original samples to fitted spline
    double maxErr = 0.0;
    double sumErr = 0.0;
    for (const auto& s : samples) {
        double fitted = SplineEvaluator::evaluate(anchors, s.x);
        double err = std::abs(s.y - fitted);
        maxErr = std::max(maxErr, err);
        sumErr += err;
    }

    result.success = true;
    result.anchors = std::move(anchors);
    result.numAnchors = static_cast<int>(result.anchors.size());
    result.maxError = maxErr;
    result.averageError = sumErr / static_cast<double>(samples.size());
    result.message = "Fitted " + juce::String(result.numAnchors) + " anchors (including " +
                     juce::String(static_cast<int>(mandatoryIndices.size())) + " feature anchors), max error: " +
                     juce::String(maxErr, 4);

    return result;
}

//==============================================================================
// Step 1: Sample & Sanitize
//==============================================================================

std::vector<SplineFitter::Sample> SplineFitter::sampleAndSanitize(
    const LayeredTransferFunction& ltf,
    const SplineFitConfig& config) {

    const int tableSize = ltf.getTableSize();
    std::vector<Sample> samples;
    samples.reserve(tableSize * 2);  // Reserve extra space for densification

    // Raster-to-polyline: sample entire composite (what user sees)
    // CRITICAL: Read from composite, not base layer
    // This includes base + harmonics + normalization = actual visible curve
    for (int i = 0; i < tableSize; ++i) {
        double x = ltf.normalizeIndex(i);  // Maps to [-1, 1]
        double y = ltf.getCompositeValue(i);
        samples.push_back({x, y});
    }

    // Phase 2.1 (Simplified): Add midpoint samples for better coverage
    // CRITICAL: Must sample actual curve, NOT linear interpolation!
    // Linear interpolation creates false errors for high-frequency curves (e.g., Harmonic 15)
    std::vector<Sample> densified;
    densified.reserve(samples.size() * 2);

    for (size_t i = 0; i < samples.size(); ++i) {
        densified.push_back(samples[i]);

        // Add midpoint sample between this and next point
        if (i < samples.size() - 1) {
            double midX = (samples[i].x + samples[i+1].x) / 2.0;

            // Find the table index corresponding to midX and sample the ACTUAL curve
            // This prevents false errors from linear interpolation on high-frequency curves
            int tableIdx = 0;
            double minDist = std::abs(ltf.normalizeIndex(0) - midX);
            for (int j = 1; j < tableSize; ++j) {
                double dist = std::abs(ltf.normalizeIndex(j) - midX);
                if (dist < minDist) {
                    minDist = dist;
                    tableIdx = j;
                }
            }
            double midY = ltf.getCompositeValue(tableIdx);

            densified.push_back({midX, midY});
        }
    }

    samples = std::move(densified);

    // Sort by x (should already be sorted, but ensure)
    sortByX(samples);

    // Deduplicate near-verticals
    deduplicateNearVerticals(samples);

    // Enforce strict monotonicity (if enabled)
    if (config.enforceMonotonicity) {
        enforceMonotonicity(samples);
    }

    // Clamp to [-1, 1] range
    clampToRange(samples);

    return samples;
}

void SplineFitter::sortByX(std::vector<Sample>& samples) {
    std::sort(samples.begin(), samples.end(),
        [](const Sample& a, const Sample& b) { return a.x < b.x; });
}

void SplineFitter::deduplicateNearVerticals(std::vector<Sample>& samples) {
    // Average y values for samples with nearly identical x
    constexpr double kXEpsilon = 1e-6;

    std::vector<Sample> deduped;
    deduped.reserve(samples.size());

    for (size_t i = 0; i < samples.size(); ) {
        double xSum = samples[i].x;
        double ySum = samples[i].y;
        int count = 1;

        // Group samples with similar x
        while (i + count < samples.size() &&
               std::abs(samples[i + count].x - samples[i].x) < kXEpsilon) {
            xSum += samples[i + count].x;
            ySum += samples[i + count].y;
            ++count;
        }

        // Average the group
        deduped.push_back({xSum / count, ySum / count});
        i += count;
    }

    samples = std::move(deduped);
}

void SplineFitter::enforceMonotonicity(std::vector<Sample>& samples) {
    // Light isotonic regression: ensure y is strictly increasing with x
    // Use Pool Adjacent Violators Algorithm (PAVA) with minimal deviation

    if (samples.size() < 2) return;

    // Forward pass: ensure y[i+1] >= y[i]
    for (size_t i = 1; i < samples.size(); ++i) {
        if (samples[i].y < samples[i-1].y) {
            // Average violating pairs (minimal deviation)
            double avgY = (samples[i].y + samples[i-1].y) / 2.0;
            samples[i].y = avgY;
            samples[i-1].y = avgY;
        }
    }

    // TODO: More sophisticated isotonic regression if needed
    // For now, simple pairwise averaging is sufficient
}

void SplineFitter::clampToRange(std::vector<Sample>& samples) {
    for (auto& s : samples) {
        s.x = juce::jlimit(-1.0, 1.0, s.x);
        s.y = juce::jlimit(-1.0, 1.0, s.y);
    }
}

//==============================================================================
// DEPRECATED RDP IMPLEMENTATION (Kept for reference)
// Issue: RDP measures error against straight lines, but PCHIP uses cubic curves
// This objective function mismatch caused "bowing artifacts" in straight regions
// Replaced by greedy spline-aware fitting (see greedySplineFit)
//==============================================================================

std::vector<SplineAnchor> SplineFitter::ramerDouglasPeucker(
    const std::vector<Sample>& samples,
    const SplineFitConfig& config) {

    if (samples.size() < 3) {
        // Too few points - return all as anchors
        std::vector<SplineAnchor> anchors;
        for (const auto& s : samples) {
            anchors.push_back({s.x, s.y, false, 0.0});
        }
        return anchors;
    }

    // RDP with hybrid error metric
    std::vector<bool> keep(samples.size(), false);
    keep[0] = true;  // Always keep endpoints
    keep[samples.size() - 1] = true;

    rdpRecursive(samples, 0, samples.size() - 1, config, keep);

    // Convert kept samples to anchors
    std::vector<SplineAnchor> anchors;
    for (size_t i = 0; i < samples.size(); ++i) {
        if (keep[i]) {
            anchors.push_back({samples[i].x, samples[i].y, false, 0.0});
        }
    }

    return anchors;
}

void SplineFitter::rdpRecursive(
    const std::vector<Sample>& samples,
    size_t startIdx,
    size_t endIdx,
    const SplineFitConfig& config,
    std::vector<bool>& keep) {

    if (endIdx - startIdx < 2) return;

    // Find point with maximum hybrid error
    double maxError = 0.0;
    size_t maxIdx = startIdx;

    for (size_t i = startIdx + 1; i < endIdx; ++i) {
        double error = computeHybridError(
            samples[i],
            samples[startIdx],
            samples[endIdx],
            config.positionTolerance,
            config.derivativeTolerance
        );

        if (error > maxError) {
            maxError = error;
            maxIdx = i;
        }
    }

    // If max error exceeds tolerance, split at that point
    if (maxError > config.positionTolerance) {
        keep[maxIdx] = true;
        rdpRecursive(samples, startIdx, maxIdx, config, keep);
        rdpRecursive(samples, maxIdx, endIdx, config, keep);
    }
}

double SplineFitter::computeHybridError(
    const Sample& point,
    const Sample& lineStart,
    const Sample& lineEnd,
    double alpha,
    double beta) {

    // Compute perpendicular distance to line (position error)
    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;
    double lineLenSq = dx*dx + dy*dy;

    if (lineLenSq < 1e-12) {
        // Degenerate line - just return distance to start
        return std::abs(point.y - lineStart.y);
    }

    double t = ((point.x - lineStart.x) * dx + (point.y - lineStart.y) * dy) / lineLenSq;
    t = juce::jlimit(0.0, 1.0, t);

    double projX = lineStart.x + t * dx;
    double projY = lineStart.y + t * dy;

    double positionError = std::sqrt((point.x - projX)*(point.x - projX) +
                                      (point.y - projY)*(point.y - projY));

    // TODO: Add derivative error component (beta * |y' - ŷ'|)
    // For now, just use position error

    return positionError;
}

double SplineFitter::estimateDerivative(
    const std::vector<Sample>& samples,
    size_t index) {

    // Central difference for interior points
    if (index > 0 && index < samples.size() - 1) {
        double dx = samples[index + 1].x - samples[index - 1].x;
        if (std::abs(dx) < 1e-12) return 0.0;
        return (samples[index + 1].y - samples[index - 1].y) / dx;
    }

    // Forward difference for first point
    if (index == 0 && samples.size() > 1) {
        double dx = samples[1].x - samples[0].x;
        if (std::abs(dx) < 1e-12) return 0.0;
        return (samples[1].y - samples[0].y) / dx;
    }

    // Backward difference for last point
    if (index == samples.size() - 1 && samples.size() > 1) {
        double dx = samples[index].x - samples[index - 1].x;
        if (std::abs(dx) < 1e-12) return 0.0;
        return (samples[index].y - samples[index - 1].y) / dx;
    }

    return 0.0;
}

//==============================================================================
// Step 3: Tangent Computation (Dispatcher)
//==============================================================================

void SplineFitter::computeTangents(
    std::vector<SplineAnchor>& anchors,
    const SplineFitConfig& config) {

    switch (config.tangentAlgorithm) {
        case TangentAlgorithm::PCHIP:
            computePCHIPTangentsImpl(anchors, config);
            break;
        case TangentAlgorithm::FritschCarlson:
            computeFritschCarlsonTangents(anchors, config);
            break;
        case TangentAlgorithm::Akima:
            computeAkimaTangents(anchors, config);
            break;
        case TangentAlgorithm::FiniteDifference:
            computeFiniteDifferenceTangents(anchors, config);
            break;
        default:
            jassertfalse;
            computePCHIPTangentsImpl(anchors, config);
            break;
    }
}

// Legacy API (deprecated)
void SplineFitter::computePCHIPTangents(
    std::vector<SplineAnchor>& anchors,
    const SplineFitConfig& config) {
    computeTangents(anchors, config);
}

//==============================================================================
// PCHIP Tangent Computation (Implementation)
//==============================================================================

void SplineFitter::computePCHIPTangentsImpl(
    std::vector<SplineAnchor>& anchors,
    const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2) return;

    // Compute secant slopes d_i = (y_{i+1} - y_i) / (x_{i+1} - x_i)
    std::vector<double> secants(n - 1);
    for (int i = 0; i < n - 1; ++i) {
        double dx = anchors[i+1].x - anchors[i].x;
        if (std::abs(dx) < 1e-12) {
            secants[i] = 0.0;  // Degenerate segment
        } else {
            secants[i] = (anchors[i+1].y - anchors[i].y) / dx;
        }
    }

    // Compute tangents m_i using Fritsch-Carlson rules
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            // Endpoint: use one-sided derivative
            anchors[i].tangent = secants[0];
        } else if (i == n - 1) {
            // Endpoint: use one-sided derivative
            anchors[i].tangent = secants[n - 2];
        } else {
            // Interior point: apply Fritsch-Carlson formula
            double d_prev = secants[i - 1];
            double d_next = secants[i];

            // If secants have opposite signs, set tangent to zero (local extremum)
            if (d_prev * d_next <= 0.0) {
                anchors[i].tangent = 0.0;
            } else {
                // Weighted harmonic mean
                double dx_prev = anchors[i].x - anchors[i-1].x;
                double dx_next = anchors[i+1].x - anchors[i].x;

                double w1 = 2.0 * dx_next + dx_prev;
                double w2 = dx_next + 2.0 * dx_prev;

                anchors[i].tangent = harmonicMean(d_prev, d_next, w1, w2);
            }
        }

        // Enforce slope caps (for anti-aliasing)
        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope,
                                           anchors[i].tangent);
    }

    // Phase 1.1: Overshoot detection and correction
    // Check if PCHIP cubic overshoots the monotonic range between anchors
    // This prevents visual "bulges" and sonic artifacts in smooth regions
    const int maxOvershootIterations = 3;  // Iterative refinement
    for (int iter = 0; iter < maxOvershootIterations; ++iter) {
        bool hadOvershoot = false;

        for (int i = 0; i < n - 1; ++i) {
            // Sample cubic at 5 interior points between anchors
            const double yMin = std::min(anchors[i].y, anchors[i+1].y);
            const double yMax = std::max(anchors[i].y, anchors[i+1].y);
            const double overshootTolerance = 0.001;  // Allow tiny numerical error

            for (int j = 1; j < 5; ++j) {
                const double t = j / 5.0;
                const double x = anchors[i].x + t * (anchors[i+1].x - anchors[i].x);
                const double y = SplineEvaluator::evaluateSegment(anchors[i], anchors[i+1], x);

                // If cubic overshoots endpoint range, scale tangents down
                if (y < yMin - overshootTolerance || y > yMax + overshootTolerance) {
                    const double dampingFactor = 0.7;  // Reduce tangents by 30%
                    anchors[i].tangent *= dampingFactor;
                    anchors[i+1].tangent *= dampingFactor;
                    hadOvershoot = true;
                    break;  // Move to next segment
                }
            }
        }

        // If no overshoots detected, we're done
        if (!hadOvershoot) {
            break;
        }
    }

    // Phase 1.2: Length-based tangent scaling
    // Very long segments (sparse anchor distribution) need gentler tangents to avoid oscillation
    // Only apply scaling to segments longer than threshold (0.3 = 15% of full range)
    const double longSegmentThreshold = 0.3;
    for (int i = 0; i < n - 1; ++i) {
        const double segmentLength = anchors[i+1].x - anchors[i].x;

        if (segmentLength > longSegmentThreshold) {
            // Scale factor for long segments: gradually reduce tangents
            // Formula: min(1.0, threshold / segmentLength)
            // Example: length=0.4 → factor=0.75, length=0.6 → factor=0.5
            const double lengthFactor = std::min(1.0, longSegmentThreshold / segmentLength);

            anchors[i].tangent *= lengthFactor;

            // Also scale the next anchor's tangent if it's the last one
            if (i == n - 2) {
                anchors[i+1].tangent *= lengthFactor;
            }
        }
    }
}

double SplineFitter::harmonicMean(double a, double b, double wa, double wb) {
    // Formula: m = (w1 + w2) / (w1/d_prev + w2/d_next)
    if (std::abs(a) < 1e-12 || std::abs(b) < 1e-12) {
        return 0.0;
    }
    return (wa + wb) / (wa / a + wb / b);
}

//==============================================================================
// Fritsch-Carlson Tangent Computation (Monotone-Preserving)
//==============================================================================

void SplineFitter::computeFritschCarlsonTangents(
    std::vector<SplineAnchor>& anchors,
    const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2) return;

    std::vector<double> tangents(n, 0.0);
    std::vector<double> deltas(n - 1);

    // 1. Compute segment slopes
    for (int i = 0; i < n - 1; ++i) {
        double dx = anchors[i+1].x - anchors[i].x;
        if (std::abs(dx) < 1e-12) {
            deltas[i] = 0.0;
        } else {
            deltas[i] = (anchors[i+1].y - anchors[i].y) / dx;
        }
    }

    // 2. Initial tangent estimates
    for (int i = 1; i < n - 1; ++i) {
        if (deltas[i-1] * deltas[i] <= 0.0) {
            tangents[i] = 0.0;  // Local extremum - force horizontal tangent
        } else {
            // Weighted average (harmonic mean variant)
            double w1 = 2.0 * (anchors[i+1].x - anchors[i].x) + (anchors[i].x - anchors[i-1].x);
            double w2 = (anchors[i+1].x - anchors[i].x) + 2.0 * (anchors[i].x - anchors[i-1].x);
            tangents[i] = (w1 * deltas[i-1] + w2 * deltas[i]) / (w1 + w2);
        }
    }

    // 3. Enforce Fritsch-Carlson monotonicity constraints
    for (int i = 0; i < n - 1; ++i) {
        if (std::abs(deltas[i]) < 1e-9)
            continue;  // Skip flat segments

        double alpha = tangents[i] / deltas[i];
        double beta = tangents[i+1] / deltas[i];

        // Constraint: α² + β² ≤ 9 (ensures no overshoot)
        if (alpha * alpha + beta * beta > 9.0) {
            double tau = 3.0 / std::sqrt(alpha * alpha + beta * beta);
            tangents[i] = tau * alpha * deltas[i];
            tangents[i+1] = tau * beta * deltas[i];
        }
    }

    // 4. Boundary tangents (use one-sided slopes)
    tangents[0] = deltas[0];
    tangents[n-1] = deltas[n-2];

    // 5. Apply tangents to anchors and enforce slope bounds
    for (int i = 0; i < n; ++i) {
        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope, tangents[i]);
    }
}

//==============================================================================
// Akima Tangent Computation (Local Weighted Average)
//==============================================================================

void SplineFitter::computeAkimaTangents(
    std::vector<SplineAnchor>& anchors,
    const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2) return;

    std::vector<double> tangents(n);
    std::vector<double> slopes(n + 3);  // Extended for boundary handling

    // 1. Compute slopes with extrapolation for boundaries
    for (int i = 2; i < static_cast<int>(slopes.size()) - 2; ++i) {
        int anchorIdx = i - 2;
        double dx = anchors[anchorIdx + 1].x - anchors[anchorIdx].x;
        if (std::abs(dx) < 1e-12) {
            slopes[i] = 0.0;
        } else {
            slopes[i] = (anchors[anchorIdx + 1].y - anchors[anchorIdx].y) / dx;
        }
    }

    // Extrapolate boundary slopes
    slopes[0] = 2.0 * slopes[2] - slopes[3];
    slopes[1] = 2.0 * slopes[2] - slopes[3];
    slopes[slopes.size() - 1] = 2.0 * slopes[slopes.size() - 3] - slopes[slopes.size() - 4];
    slopes[slopes.size() - 2] = 2.0 * slopes[slopes.size() - 3] - slopes[slopes.size() - 4];

    // 2. Akima weighted formula
    for (int i = 0; i < n; ++i) {
        double m1 = slopes[i];
        double m2 = slopes[i + 1];
        double m3 = slopes[i + 2];
        double m4 = slopes[i + 3];

        double w1 = std::abs(m4 - m3);
        double w2 = std::abs(m2 - m1);

        if (w1 + w2 < 1e-9) {
            tangents[i] = (m2 + m3) * 0.5;  // Average if weights are zero
        } else {
            tangents[i] = (w1 * m2 + w2 * m3) / (w1 + w2);
        }
    }

    // 3. Apply tangents to anchors and enforce slope bounds
    for (int i = 0; i < n; ++i) {
        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope, tangents[i]);
    }
}

//==============================================================================
// Finite Difference Tangent Computation (Simple Baseline)
//==============================================================================

void SplineFitter::computeFiniteDifferenceTangents(
    std::vector<SplineAnchor>& anchors,
    const SplineFitConfig& config) {

    const int n = static_cast<int>(anchors.size());
    if (n < 2) return;

    for (int i = 0; i < n; ++i) {
        double tangent = 0.0;

        if (i == 0) {
            // Forward difference for first point
            double dx = anchors[1].x - anchors[0].x;
            if (std::abs(dx) >= 1e-12) {
                tangent = (anchors[1].y - anchors[0].y) / dx;
            }
        } else if (i == n - 1) {
            // Backward difference for last point
            double dx = anchors[i].x - anchors[i-1].x;
            if (std::abs(dx) >= 1e-12) {
                tangent = (anchors[i].y - anchors[i-1].y) / dx;
            }
        } else {
            // Central difference for interior points
            double dx = anchors[i+1].x - anchors[i-1].x;
            if (std::abs(dx) >= 1e-12) {
                tangent = (anchors[i+1].y - anchors[i-1].y) / dx;
            }
        }

        anchors[i].tangent = juce::jlimit(config.minSlope, config.maxSlope, tangent);
    }
}

//==============================================================================
// Greedy Spline Fitting (Replaces RDP)
//==============================================================================

SplineFitter::WorstFitResult SplineFitter::findWorstFitSample(
    const std::vector<Sample>& samples,
    const std::vector<SplineAnchor>& anchors) {

    WorstFitResult result{0, 0.0};

    if (anchors.size() < 2) {
        return result;
    }

    // Scan all samples, find worst error
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& sample = samples[i];

        // Skip samples that already have anchors at their x position
        bool hasAnchor = std::any_of(anchors.begin(), anchors.end(),
            [&sample](const SplineAnchor& a) {
                return std::abs(a.x - sample.x) < 1e-9;
            });

        if (hasAnchor) {
            continue;  // Skip this sample
        }

        // Evaluate PCHIP spline at this x position
        double splineY = SplineEvaluator::evaluate(anchors, sample.x);

        // Compute absolute error
        double error = std::abs(sample.y - splineY);

        if (error > result.maxError) {
            result.maxError = error;
            result.sampleIndex = i;
        }
    }

    return result;
}

std::vector<SplineAnchor> SplineFitter::greedySplineFit(
    const std::vector<Sample>& samples,
    const SplineFitConfig& config,
    const LayeredTransferFunction* ltf,
    const std::vector<int>& mandatoryAnchorIndices) {

    if (samples.size() < 2)
        return {};

    // Phase 3: FEATURE-BASED ANCHOR PLACEMENT
    // Start with mandatory feature anchors (extrema, inflection points)
    std::vector<SplineAnchor> anchors;

    // DEBUG: Log spatial distribution of mandatory anchors
    if (!mandatoryAnchorIndices.empty() && ltf != nullptr) {
        int leftCount = 0, midCount = 0, rightCount = 0;
        for (int idx : mandatoryAnchorIndices) {
            double x = ltf->normalizeIndex(idx);
            if (x < -0.33) leftCount++;
            else if (x > 0.33) rightCount++;
            else midCount++;
        }
        DBG("========== SPLINE FIT DEBUG ==========");
        DBG("Total mandatory anchors: " + juce::String(static_cast<int>(mandatoryAnchorIndices.size())));
        DBG("Left (x<-0.33): " + juce::String(leftCount) +
            " | Mid: " + juce::String(midCount) +
            " | Right (x>0.33): " + juce::String(rightCount));
        DBG("maxAnchors: " + juce::String(config.maxAnchors));
    }

    if (mandatoryAnchorIndices.empty() || ltf == nullptr) {
        // Fallback: If no mandatory anchors provided, use endpoints only
        anchors.push_back({samples.front().x, samples.front().y, false, 0.0});
        anchors.push_back({samples.back().x, samples.back().y, false, 0.0});
    } else {
        // Convert mandatory table indices to sample anchors
        // mandatoryAnchorIndices are table indices from CurveFeatureDetector
        // We need to convert them to x coordinates, then find the closest samples

        for (int tableIdx : mandatoryAnchorIndices) {
            // Convert table index to x coordinate using LayeredTransferFunction's mapping
            double targetX = ltf->normalizeIndex(tableIdx);

            // Find closest sample to this x coordinate
            // Since samples are sorted by x, we could use binary search,
            // but linear search is fine for small arrays
            size_t closestIdx = 0;
            double minDist = std::abs(samples[0].x - targetX);

            for (size_t i = 1; i < samples.size(); ++i) {
                double dist = std::abs(samples[i].x - targetX);
                if (dist < minDist) {
                    minDist = dist;
                    closestIdx = i;
                }
            }

            anchors.push_back({samples[closestIdx].x, samples[closestIdx].y, false, 0.0});
        }

        // Sort anchors by x (should already be sorted, but ensure)
        std::sort(anchors.begin(), anchors.end(),
            [](const SplineAnchor& a, const SplineAnchor& b) { return a.x < b.x; });

        // Remove any duplicates
        anchors.erase(
            std::unique(anchors.begin(), anchors.end(),
                [](const SplineAnchor& a, const SplineAnchor& b) {
                    return std::abs(a.x - b.x) < 1e-9;
                }),
            anchors.end()
        );
    }

    // Calculate how many additional anchors we can add
    // CRITICAL: Clamp to 0 to prevent negative iteration count if mandatory anchors exceed maxAnchors
    int remainingAnchors = std::max(0, config.maxAnchors - static_cast<int>(anchors.size()));

    // Iteratively refine with error-driven placement (quality)
    for (int iteration = 0; iteration < remainingAnchors; ++iteration) {
        // Compute tangents for current anchor set using configured algorithm
        computeTangents(anchors, config);

        // Find sample with highest error
        auto worst = findWorstFitSample(samples, anchors);

        // Converged?
        if (worst.maxError <= config.positionTolerance)
            break;

        // Insert anchor at worst-fit location
        const auto& worstSample = samples[worst.sampleIndex];

        // Find insertion point (maintain sorted order by x)
        auto insertPos = std::lower_bound(
            anchors.begin(), anchors.end(), worstSample.x,
            [](const SplineAnchor& a, double x) { return a.x < x; }
        );

        // Don't insert duplicate x positions
        bool isDuplicate = std::any_of(anchors.begin(), anchors.end(),
            [&worstSample](const SplineAnchor& a) {
                return std::abs(a.x - worstSample.x) < 1e-9;
            });

        if (isDuplicate)
            break;  // No progress possible

        anchors.insert(insertPos, {worstSample.x, worstSample.y, false, 0.0});
    }

    // Final tangent computation using configured algorithm
    computeTangents(anchors, config);

    return anchors;
}

} // namespace Services
} // namespace dsp_core
