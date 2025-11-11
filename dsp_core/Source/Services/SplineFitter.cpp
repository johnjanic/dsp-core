#include "SplineFitter.h"
#include "SplineEvaluator.h"
#include "CurveFeatureDetector.h"
#include "AdaptiveToleranceCalculator.h"
#include "SymmetryAnalyzer.h"
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
    // CRITICAL: Always read base + harmonics, ignoring spline layer state
    // When re-entering spline mode, we must fit the baked base curve, not the old spline
    for (int i = 0; i < tableSize; ++i) {
        double x = ltf.normalizeIndex(i);  // Maps to [-1, 1]
        double y = ltf.evaluateBaseAndHarmonics(x);  // FIX: Always read base+harmonics
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

            // Sample the ACTUAL base+harmonics curve at midpoint
            // This prevents false errors from linear interpolation on high-frequency curves
            double midY = ltf.evaluateBaseAndHarmonics(midX);  // FIX: Always read base+harmonics

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
// Tangent Computation (Dispatcher)
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

    // Compute vertical range for adaptive tolerance calculation
    double minY = samples.front().y;
    double maxY = samples.front().y;
    for (const auto& sample : samples) {
        minY = std::min(minY, sample.y);
        maxY = std::max(maxY, sample.y);
    }
    double verticalRange = maxY - minY;

    // Configure adaptive tolerance to respect config.positionTolerance
    // Derive relativeErrorTarget from positionTolerance and verticalRange
    AdaptiveToleranceCalculator::Config adaptiveConfig;
    if (verticalRange > 1e-9) {
        adaptiveConfig.relativeErrorTarget = config.positionTolerance / verticalRange;
    } else {
        adaptiveConfig.relativeErrorTarget = 0.01;  // Fallback for flat curves
    }

    // Analyze symmetry if needed
    bool useSymmetricMode = false;
    if (config.symmetryMode == SymmetryMode::Always) {
        useSymmetricMode = true;
    } else if (config.symmetryMode == SymmetryMode::Auto && ltf != nullptr) {
        auto symmetryResult = SymmetryAnalyzer::analyzeOddSymmetry(*ltf);
        useSymmetricMode = (symmetryResult.score >= config.symmetryThreshold);
    }
    // If Never: useSymmetricMode remains false

    // Iteratively refine with error-driven placement (quality)
    for (int iteration = 0; iteration < remainingAnchors; ++iteration) {
        // Compute tangents for current anchor set using configured algorithm
        computeTangents(anchors, config);

        // Find sample with highest error
        auto worst = findWorstFitSample(samples, anchors);

        // Compute adaptive tolerance (increases with anchor density to prevent over-fitting)
        double adaptiveTolerance = AdaptiveToleranceCalculator::computeTolerance(
            verticalRange,
            static_cast<int>(anchors.size()),
            config.maxAnchors,
            adaptiveConfig
        );

        // Use config.positionTolerance as minimum floor
        adaptiveTolerance = std::max(adaptiveTolerance, config.positionTolerance);

        // Converged?
        if (worst.maxError <= adaptiveTolerance)
            break;

        // Insert anchor(s) at worst-fit location
        const auto& worstSample = samples[worst.sampleIndex];

        if (useSymmetricMode) {
            // === SYMMETRIC MODE: Add complementary pair ===

            // Find complementary x position
            double complementaryX = -worstSample.x;

            // Find sample closest to complementary x
            size_t complementaryIdx = 0;
            double minDist = std::abs(samples[0].x - complementaryX);
            for (size_t i = 1; i < samples.size(); ++i) {
                double dist = std::abs(samples[i].x - complementaryX);
                if (dist < minDist) {
                    minDist = dist;
                    complementaryIdx = i;
                }
            }

            const auto& complementarySample = samples[complementaryIdx];

            // Check if complementary position already has anchor
            bool hasComplementaryAnchor = std::any_of(anchors.begin(), anchors.end(),
                [&complementarySample](const SplineAnchor& a) {
                    return std::abs(a.x - complementarySample.x) < 1e-9;
                });

            // Check if original position already has anchor
            bool hasOriginalAnchor = std::any_of(anchors.begin(), anchors.end(),
                [&worstSample](const SplineAnchor& a) {
                    return std::abs(a.x - worstSample.x) < 1e-9;
                });

            // Check if complementary position ALSO has significant error
            // (Don't waste anchor budget on low-error positions)
            double complementaryError = 0.0;
            if (!hasComplementaryAnchor) {
                double splineYAtComplementary = SplineEvaluator::evaluate(anchors, complementarySample.x);
                complementaryError = std::abs(complementarySample.y - splineYAtComplementary);
            }

            // Only add complementary pair if:
            // 1. Original position doesn't have anchor
            // 2. Complementary position doesn't have anchor
            // 3. Complementary position has significant error (>50% of adaptive tolerance)
            // 4. We have room for 2 more anchors
            bool canAddPair = !hasOriginalAnchor && !hasComplementaryAnchor &&
                              complementaryError > adaptiveTolerance * 0.5 &&
                              (iteration + 1 < remainingAnchors);

            if (canAddPair) {
                // Add both anchors with symmetric y values
                // For odd symmetry: if we have f(x) at x and f(-x) at -x,
                // we want to enforce perfect symmetry by using the average
                double yOriginal = worstSample.y;  // f(x)
                double yComplementary = complementarySample.y;  // f(-x)

                // For perfect odd symmetry: f(-x) = -f(x)
                // Average the two to get a symmetric value
                double ySymmetric = (yOriginal - yComplementary) / 2.0;

                // Insert original anchor at (x, ySymmetric)
                auto insertPos1 = std::lower_bound(
                    anchors.begin(), anchors.end(), worstSample.x,
                    [](const SplineAnchor& a, double x) { return a.x < x; }
                );
                anchors.insert(insertPos1, {worstSample.x, ySymmetric, false, 0.0});

                // Insert complementary anchor at (-x, -ySymmetric) for odd symmetry
                auto insertPos2 = std::lower_bound(
                    anchors.begin(), anchors.end(), complementarySample.x,
                    [](const SplineAnchor& a, double x) { return a.x < x; }
                );
                anchors.insert(insertPos2, {complementarySample.x, -ySymmetric, false, 0.0});

                // Consume 2 iterations (we added 2 anchors)
                ++iteration;

            } else {
                // Fallback: add single anchor (asymmetric, but necessary)
                // Better to break symmetry slightly than leave large errors unaddressed
                auto insertPos = std::lower_bound(
                    anchors.begin(), anchors.end(), worstSample.x,
                    [](const SplineAnchor& a, double x) { return a.x < x; }
                );

                bool isDuplicate = std::any_of(anchors.begin(), anchors.end(),
                    [&worstSample](const SplineAnchor& a) {
                        return std::abs(a.x - worstSample.x) < 1e-9;
                    });

                if (isDuplicate)
                    break;  // No progress possible

                anchors.insert(insertPos, {worstSample.x, worstSample.y, false, 0.0});
            }

        } else {
            // === ASYMMETRIC MODE: Original greedy algorithm ===

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
    }

    // Zero-crossing drift verification (defensive DC blocking)
    if (config.enableZeroCrossingCheck && ltf != nullptr) {
        auto zcInfo = analyzeZeroCrossing(*ltf, anchors, config);

        // Only intervene if:
        // 1. Base curve has zero-crossing
        // 2. Fitted spline introduced significant drift
        // 3. No anchor already exists at x≈0
        // 4. Anchor budget allows
        if (zcInfo.baseCurveHasZeroCrossing &&
            zcInfo.drift > config.zeroCrossingTolerance &&
            anchors.size() < static_cast<size_t>(config.maxAnchors)) {

            // Check if anchor already exists at x≈0
            bool hasAnchorAtZero = std::any_of(anchors.begin(), anchors.end(),
                [](const SplineAnchor& a) {
                    return std::abs(a.x) < 1e-6;  // Within 1e-6 of origin
                });

            if (!hasAnchorAtZero) {
                // Add corrective anchor at exactly x=0 with base curve's y value
                // (Anchors have continuous freedom - can be placed at exact 0.0)
                auto insertPos = std::lower_bound(
                    anchors.begin(), anchors.end(), 0.0,
                    [](const SplineAnchor& a, double x) { return a.x < x; }
                );
                anchors.insert(insertPos, {0.0, zcInfo.baseYAtZero, false, 0.0});

                // Recompute tangents with new anchor
                computeTangents(anchors, config);
            }
        }
    }

    // Optional: Prune redundant anchors after fitting
    if (config.enableAnchorPruning && anchors.size() > 2) {
        // Use the final adaptive tolerance with multiplier for pruning
        double finalAdaptiveTolerance = AdaptiveToleranceCalculator::computeTolerance(
            verticalRange,
            static_cast<int>(anchors.size()),
            config.maxAnchors,
            adaptiveConfig
        );
        finalAdaptiveTolerance = std::max(finalAdaptiveTolerance, config.positionTolerance);
        double pruningTolerance = finalAdaptiveTolerance * config.pruningToleranceMultiplier;

        pruneRedundantAnchors(anchors, samples, pruningTolerance, config);
    }

    // Final tangent computation using configured algorithm
    computeTangents(anchors, config);

    return anchors;
}

//==============================================================================
// Anchor Pruning (Optional Post-Processing)
//==============================================================================

void SplineFitter::pruneRedundantAnchors(
    std::vector<SplineAnchor>& anchors,
    const std::vector<Sample>& samples,
    double pruningTolerance,
    const SplineFitConfig& config) {

    if (anchors.size() <= 2) {
        return;  // Cannot prune endpoints
    }

    // Iteratively test removing each non-endpoint anchor
    // If removal doesn't increase error beyond tolerance, keep it removed
    for (int i = 1; i < static_cast<int>(anchors.size()) - 1; ) {
        // 1. Temporarily remove anchor
        auto removed = anchors[i];
        anchors.erase(anchors.begin() + i);

        // 2. Recompute tangents with reduced anchor set
        computeTangents(anchors, config);

        // 3. Measure max error across ALL samples
        // (We need to check all samples because tangent changes can affect the entire curve)
        double maxError = 0.0;
        for (const auto& sample : samples) {
            double splineY = SplineEvaluator::evaluate(anchors, sample.x);
            double error = std::abs(sample.y - splineY);
            maxError = std::max(maxError, error);
        }

        // 4. If error exceeds tolerance, restore anchor
        if (maxError > pruningTolerance) {
            anchors.insert(anchors.begin() + i, removed);
            ++i;  // Move to next anchor
        }
        // else: anchor was successfully removed, check same index again (array shifted)
    }
}

//==============================================================================
// Zero-Crossing Analysis (Defensive DC Drift Detection)
//==============================================================================

SplineFitter::ZeroCrossingInfo SplineFitter::analyzeZeroCrossing(
    const LayeredTransferFunction& ltf,
    const std::vector<SplineAnchor>& anchors,
    const SplineFitConfig& config) {

    ZeroCrossingInfo info;

    // Evaluate base + harmonics at exactly x=0 (matches what we're fitting)
    // CRITICAL: Must use evaluateBaseAndHarmonics, not raw base layer
    // This ensures we're checking zero-crossing of the actual curve being fitted
    info.baseYAtZero = ltf.evaluateBaseAndHarmonics(0.0);

    // Check if base curve crosses zero (within tolerance)
    if (std::abs(info.baseYAtZero) < config.zeroCrossingTolerance) {
        info.baseCurveHasZeroCrossing = true;

        // Evaluate fitted spline at exactly x=0
        info.fittedYAtZero = SplineEvaluator::evaluate(anchors, 0.0);

        // Compute drift
        info.drift = std::abs(info.fittedYAtZero - info.baseYAtZero);
    }

    return info;
}

} // namespace Services
} // namespace dsp_core
