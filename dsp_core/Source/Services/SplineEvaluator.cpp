#include "SplineEvaluator.h"
#include <algorithm>
#include <cmath>
#include <juce_core/juce_core.h>

namespace dsp_core {
namespace Services {

double SplineEvaluator::evaluate(
    const std::vector<SplineAnchor>& anchors,
    double x) {

    if (anchors.empty()) return 0.0;
    if (anchors.size() == 1) return anchors[0].y;

    // Find segment containing x
    int segIdx = findSegment(anchors, x);
    if (segIdx < 0) return anchors.front().y;  // Before first anchor
    if (segIdx >= static_cast<int>(anchors.size()) - 1) {
        return anchors.back().y;  // After last anchor
    }

    // Evaluate cubic Hermite on segment
    return evaluateSegment(anchors[segIdx], anchors[segIdx + 1], x);
}

double SplineEvaluator::evaluateDerivative(
    const std::vector<SplineAnchor>& anchors,
    double x) {

    if (anchors.empty()) return 0.0;
    if (anchors.size() == 1) return 0.0;

    // Find segment containing x
    int segIdx = findSegment(anchors, x);
    if (segIdx < 0) return anchors.front().tangent;  // Before first anchor
    if (segIdx >= static_cast<int>(anchors.size()) - 1) {
        return anchors.back().tangent;  // After last anchor
    }

    // Evaluate cubic Hermite derivative on segment
    return evaluateSegmentDerivative(anchors[segIdx], anchors[segIdx + 1], x);
}

int SplineEvaluator::findSegment(
    const std::vector<SplineAnchor>& anchors,
    double x) {

    // Binary search for segment containing x
    // Find first anchor with anchor.x >= x
    auto it = std::lower_bound(
        anchors.begin(), anchors.end(), x,
        [](const SplineAnchor& anchor, double val) { return anchor.x < val; }
    );

    if (it == anchors.begin()) return -1;  // Before first anchor
    if (it == anchors.end()) return static_cast<int>(anchors.size()) - 1;  // After last anchor

    // Return index of segment start (the anchor before 'it')
    return static_cast<int>(std::distance(anchors.begin(), it)) - 1;
}

double SplineEvaluator::evaluateSegment(
    const SplineAnchor& p0,
    const SplineAnchor& p1,
    double x) {

    // Normalize to [0, 1] within segment
    double dx = p1.x - p0.x;
    if (std::abs(dx) < 1e-12) return p0.y;  // Degenerate segment

    double t = (x - p0.x) / dx;
    t = juce::jlimit(0.0, 1.0, t);  // Clamp to segment bounds

    // Cubic Hermite basis functions
    // h00(t) = 2t³ - 3t² + 1 = (1 + 2t)(1-t)²
    // h10(t) = t³ - 2t² + t = t(1-t)²
    // h01(t) = -2t³ + 3t² = t²(3-2t)
    // h11(t) = t³ - t² = t²(t-1)

    double t2 = t * t;
    double t3 = t2 * t;
    double omt = 1.0 - t;  // (1-t)
    double omt2 = omt * omt;  // (1-t)²

    double h00 = (1.0 + 2.0 * t) * omt2;  // 2t³ - 3t² + 1
    double h10 = t * omt2;                 // t³ - 2t² + t
    double h01 = t2 * (3.0 - 2.0 * t);     // -2t³ + 3t²
    double h11 = t2 * (t - 1.0);           // t³ - t²

    // Get tangent values (slopes at control points)
    double m0 = p0.tangent;
    double m1 = p1.tangent;

    // Hermite interpolation
    // H(t) = h00·y0 + h10·Δx·m0 + h01·y1 + h11·Δx·m1
    return h00 * p0.y + h10 * dx * m0 + h01 * p1.y + h11 * dx * m1;
}

double SplineEvaluator::evaluateSegmentDerivative(
    const SplineAnchor& p0,
    const SplineAnchor& p1,
    double x) {

    // Normalize to [0, 1] within segment
    double dx = p1.x - p0.x;
    if (std::abs(dx) < 1e-12) return 0.0;  // Degenerate segment

    double t = (x - p0.x) / dx;
    t = juce::jlimit(0.0, 1.0, t);  // Clamp to segment bounds

    // Derivatives of cubic Hermite basis functions (with respect to t)
    // h00'(t) = 6t² - 6t = 6t(t-1)
    // h10'(t) = 3t² - 4t + 1
    // h01'(t) = -6t² + 6t = 6t(1-t)
    // h11'(t) = 3t² - 2t

    double t2 = t * t;

    double h00_dt = 6.0 * t * (t - 1.0);        // 6t² - 6t
    double h10_dt = 3.0 * t2 - 4.0 * t + 1.0;   // 3t² - 4t + 1
    double h01_dt = 6.0 * t * (1.0 - t);        // -6t² + 6t
    double h11_dt = 3.0 * t2 - 2.0 * t;         // 3t² - 2t

    // Get tangent values
    double m0 = p0.tangent;
    double m1 = p1.tangent;

    // H'(t) with respect to t
    double dH_dt = h00_dt * p0.y + h10_dt * dx * m0 + h01_dt * p1.y + h11_dt * dx * m1;

    // Convert to derivative with respect to x: dH/dx = (dH/dt) / (dx/dt) = (dH/dt) / dx
    return dH_dt / dx;
}

} // namespace Services
} // namespace dsp_core
