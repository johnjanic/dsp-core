#include "SplineFitter.h"
#include "SplineEvaluator.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dsp_core {
namespace Services {

//==============================================================================
// Main API
//==============================================================================

SplineFitResult SplineFitter::fitCurve(
    const LayeredTransferFunction& ltf,
    const SplineFitConfig& config) {

    SplineFitResult result;

    // Step 1: Sample the curve
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

    // Step 2: Simple uniform anchor placement
    auto anchors = greedySplineFit(samples, config, &ltf, {}, nullptr);

    if (anchors.empty()) {
        result.success = false;
        result.message = "Failed to produce anchors";
        return result;
    }

    // Step 3: Error analysis
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
    result.message = "Fitted " + juce::String(result.numAnchors) + " anchors, max error: " +
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
    samples.reserve(tableSize);

    // Sample entire composite curve (what user sees)
    for (int i = 0; i < tableSize; ++i) {
        double x = ltf.normalizeIndex(i);  // Maps to [-1, 1]
        double y = ltf.getCompositeValue(i);
        samples.push_back({x, y});
    }

    return samples;
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
// Uniform Sampling Fallback (Legacy)
//==============================================================================

std::vector<SplineAnchor> SplineFitter::greedySplineFitUniform(
    const std::vector<Sample>& samples,
    const SplineFitConfig& config) {

    if (samples.size() < 2)
        return {};

    std::vector<SplineAnchor> anchors;

    int numAnchors = juce::jmin(config.maxAnchors, static_cast<int>(samples.size()));

    if (numAnchors <= 2) {
        anchors.push_back({samples.front().x, samples.front().y, false, 0.0});
        anchors.push_back({samples.back().x, samples.back().y, false, 0.0});
    } else {
        // Uniform sampling: spread anchors evenly across the curve
        for (int i = 0; i < numAnchors; ++i) {
            int sampleIdx = (i * static_cast<int>(samples.size())) / numAnchors;
            sampleIdx = juce::jmin(sampleIdx, static_cast<int>(samples.size()) - 1);
            anchors.push_back({samples[sampleIdx].x, samples[sampleIdx].y, false, 0.0});
        }
    }

    // Compute tangents using configured algorithm (PCHIP/Fritsch-Carlson/Akima)
    computeTangents(anchors, config);

    return anchors;
}

//==============================================================================
// Feature-Based Anchor Placement (Hybrid Algorithm)
//==============================================================================

std::vector<SplineAnchor> SplineFitter::greedySplineFit(
    const std::vector<Sample>& samples,
    const SplineFitConfig& config,
    const LayeredTransferFunction* ltf,
    const std::vector<int>& mandatoryAnchorIndices,
    const CurveFeatureDetector::FeatureResult* features) {

    if (samples.size() < 2)
        return {};

    std::vector<SplineAnchor> anchors;

    // STRATEGY: Use feature-based placement if LTF available, otherwise uniform
    if (ltf != nullptr) {
        // === FEATURE-BASED ANCHOR PLACEMENT ===
        // Detect geometric features (extrema, inflection points) and place anchors there
        // This ensures round-trip consistency: same features → same anchor count

        std::cerr << "DEBUG: greedySplineFit using feature-based placement" << std::endl;

        CurveFeatureDetector::FeatureResult detectedFeatures;

        if (features != nullptr) {
            // Features provided externally (for testing or caching)
            detectedFeatures = *features;
        } else {
            // Detect features from LTF
            detectedFeatures = CurveFeatureDetector::detectFeatures(
                *ltf,
                config.maxAnchors,                      // Budget limit
                config.localDensityWindowSize,          // Prevent clustering (coarse)
                config.maxAnchorsPerWindow,
                config.localDensityWindowSizeFine,      // Prevent clustering (fine)
                config.maxAnchorsPerWindowFine
            );
        }

        // Convert feature indices to spline anchors
        for (int idx : detectedFeatures.mandatoryAnchors) {
            // Map table index to sample index
            // Features are in table indices [0, tableSize-1]
            // Samples are in order, so we can map directly
            if (idx >= 0 && idx < static_cast<int>(samples.size())) {
                anchors.push_back({
                    samples[idx].x,
                    samples[idx].y,
                    false,  // No custom tangent
                    0.0
                });
            }
        }

        // Ensure we have at least endpoints (defensive programming)
        if (anchors.empty()) {
            anchors.push_back({samples.front().x, samples.front().y, false, 0.0});
            anchors.push_back({samples.back().x, samples.back().y, false, 0.0});
        } else if (anchors.size() == 1) {
            // Only one anchor - add the other endpoint
            if (anchors.front().x > 0.0) {
                anchors.insert(anchors.begin(), {samples.front().x, samples.front().y, false, 0.0});
            } else {
                anchors.push_back({samples.back().x, samples.back().y, false, 0.0});
            }
        }

    } else {
        // === FALLBACK: UNIFORM SAMPLING ===
        // Used when LTF not available (e.g., direct sample fitting)
        return greedySplineFitUniform(samples, config);
    }

    // Compute tangents using configured algorithm (PCHIP/Fritsch-Carlson/Akima)
    computeTangents(anchors, config);

    return anchors;
}

} // namespace Services
} // namespace dsp_core
