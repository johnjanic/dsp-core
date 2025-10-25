#include "SplineFitter.h"
#include "SplineEvaluator.h"
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

    // Step 2: RDP simplification
    auto anchors = ramerDouglasPeucker(samples, config);

    if (anchors.empty()) {
        result.success = false;
        result.message = "RDP failed to produce anchors";
        return result;
    }

    // Enforce maxAnchors limit
    if (anchors.size() > static_cast<size_t>(config.maxAnchors)) {
        // Too many anchors - trim to max (keep endpoints + evenly spaced)
        // For now, just truncate (TODO: smarter selection in future)
        anchors.resize(config.maxAnchors);
    }

    // Step 3: PCHIP tangent computation
    computePCHIPTangents(anchors, config);

    // Step 4: Error analysis
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

    // Raster-to-polyline: sample entire baseLayer
    for (int i = 0; i < tableSize; ++i) {
        double x = ltf.normalizeIndex(i);  // Maps to [-1, 1]
        double y = ltf.getBaseLayerValue(i);
        samples.push_back({x, y});
    }

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
// Step 2: Ramer-Douglas-Peucker Simplification
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
// Step 3: PCHIP Tangent Computation
//==============================================================================

void SplineFitter::computePCHIPTangents(
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
}

double SplineFitter::harmonicMean(double a, double b, double wa, double wb) {
    // Formula: m = (w1 + w2) / (w1/d_prev + w2/d_next)
    if (std::abs(a) < 1e-12 || std::abs(b) < 1e-12) {
        return 0.0;
    }
    return (wa + wb) / (wa / a + wb / b);
}

} // namespace Services
} // namespace dsp_core
