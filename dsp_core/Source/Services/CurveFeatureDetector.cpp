#include "CurveFeatureDetector.h"
#include <algorithm>
#include <cmath>

namespace dsp_core {
namespace Services {

int CurveFeatureDetector::countAnchorsInWindow(
    int candidateIdx,
    const std::vector<int>& existingAnchors,
    int tableSize,
    double windowSizeFraction) {

    // Window extends ±(windowSize/2) around candidate
    // Example: windowSize=0.10, tableSize=256 → halfWindow = 12.8 indices
    int halfWindowIndices = static_cast<int>(tableSize * windowSizeFraction / 2.0);

    int count = 0;
    for (int existingIdx : existingAnchors) {
        if (std::abs(existingIdx - candidateIdx) <= halfWindowIndices) {
            count++;
        }
    }

    return count;
}

CurveFeatureDetector::FeatureResult CurveFeatureDetector::detectFeatures(
    const LayeredTransferFunction& ltf,
    int maxMandatoryAnchors,
    double localDensityWindowSize,
    int maxAnchorsPerWindow,
    double localDensityWindowSizeFine,
    int maxAnchorsPerWindowFine) {
    FeatureResult result;
    int tableSize = ltf.getTableSize();

    // Tolerance thresholds to avoid detecting numerical noise
    const double derivativeThreshold = 1e-6;
    const double secondDerivativeThreshold = 1e-4;

    // 1. Find local extrema (dy/dx sign changes) with significance scores
    struct Feature {
        int index;
        double significance;  // For extrema: |y|, for inflection: |d²y/dx²|
        bool isExtremum;      // true = extremum, false = inflection point
    };
    std::vector<Feature> features;

    for (int i = 1; i < tableSize - 1; ++i) {
        double deriv_prev = estimateDerivative(ltf, i - 1);
        double deriv = estimateDerivative(ltf, i);

        // Detect sign changes (extrema). At least one derivative must be significant to avoid noise.
        // Note: At the extremum itself, one derivative may be ~0, so we use OR not AND.
        if ((std::abs(deriv_prev) > derivativeThreshold || std::abs(deriv) > derivativeThreshold) &&
            deriv_prev * deriv < 0.0) {  // Sign change = local extremum
            result.localExtrema.push_back(i);

            // Significance = amplitude (how tall the peak/valley is)
            double y = ltf.getCompositeValue(i);
            features.push_back({i, std::abs(y), true});
        }
    }

    // 2. Find inflection points (d²y/dx² sign changes) with curvature scores
    for (int i = 2; i < tableSize - 2; ++i) {
        double d2y_prev = estimateSecondDerivative(ltf, i - 1);
        double d2y = estimateSecondDerivative(ltf, i);

        // Detect sign changes in curvature (inflection points). At least one must be significant.
        // Note: At the inflection point itself, curvature may be ~0, so we use OR not AND.
        if ((std::abs(d2y_prev) > secondDerivativeThreshold || std::abs(d2y) > secondDerivativeThreshold) &&
            d2y_prev * d2y < 0.0) {  // Sign change = inflection point
            result.inflectionPoints.push_back(i);

            // Significance = curvature magnitude
            features.push_back({i, std::abs(d2y), false});
        }
    }

    // 3. Prioritize and limit features if maxMandatoryAnchors is set
    result.mandatoryAnchors.push_back(0);  // Always include endpoints
    result.mandatoryAnchors.push_back(tableSize - 1);

    // Separate extrema and inflections
    std::vector<Feature> extrema, inflections;
    for (const auto& f : features) {
        if (f.isExtremum) extrema.push_back(f);
        else inflections.push_back(f);
    }

    // Sort by significance (descending)
    auto bySignificance = [](const Feature& a, const Feature& b) {
        return a.significance > b.significance;
    };
    std::sort(extrema.begin(), extrema.end(), bySignificance);
    std::sort(inflections.begin(), inflections.end(), bySignificance);

    // Apply local density constraint if enabled (localDensityWindowSize > 0)
    const bool constraintEnabled = (localDensityWindowSize > 0.0 && maxAnchorsPerWindow > 0);

    // Determine limits: if maxMandatoryAnchors is set and we need to limit, use it
    // Otherwise, allow all features (but still apply density constraint if enabled)
    bool needsLimiting = (maxMandatoryAnchors > 0 && static_cast<int>(features.size()) + 2 > maxMandatoryAnchors);
    int maxExtrema = needsLimiting ? static_cast<int>((maxMandatoryAnchors - 2) * 0.8) : static_cast<int>(extrema.size());
    int maxInflections = needsLimiting ? ((maxMandatoryAnchors - 2) - maxExtrema) : static_cast<int>(inflections.size());

    if (needsLimiting) {
        // Greedy selection: pick highest-significance features that satisfy density constraint
        int selectedExtrema = 0;
        for (int i = 0; i < static_cast<int>(extrema.size()) && selectedExtrema < maxExtrema; ++i) {
            int candidateIdx = extrema[i].index;

            // Check local density constraint
            bool satisfiesConstraint = true;
            if (constraintEnabled) {
                int localDensity = countAnchorsInWindow(
                    candidateIdx,
                    result.mandatoryAnchors,
                    tableSize,
                    localDensityWindowSize
                );

                // Use conservative threshold to leave headroom for error-driven refinement phase
                // Reserve ~30% of window budget for refinement (e.g., 8 -> 5-6 for features)
                int conservativeThreshold = static_cast<int>(maxAnchorsPerWindow * 0.7);
                if (localDensity >= conservativeThreshold) {
                    satisfiesConstraint = false;  // Window approaching limit
                }
            }

            if (satisfiesConstraint) {
                result.mandatoryAnchors.push_back(candidateIdx);
                selectedExtrema++;
            }
            // else: skip this feature, try next highest-significance extremum
        }

        // Same for inflections
        int selectedInflections = 0;
        for (int i = 0; i < static_cast<int>(inflections.size()) && selectedInflections < maxInflections; ++i) {
            int candidateIdx = inflections[i].index;

            bool satisfiesConstraint = true;
            if (constraintEnabled) {
                int localDensity = countAnchorsInWindow(
                    candidateIdx,
                    result.mandatoryAnchors,
                    tableSize,
                    localDensityWindowSize
                );

                // Use same conservative threshold for inflections
                int conservativeThreshold = static_cast<int>(maxAnchorsPerWindow * 0.7);
                if (localDensity >= conservativeThreshold) {
                    satisfiesConstraint = false;
                }
            }

            if (satisfiesConstraint) {
                result.mandatoryAnchors.push_back(candidateIdx);
                selectedInflections++;
            }
        }
    } else {
        // Not limited - add all features (no density constraint needed)
        result.mandatoryAnchors.insert(
            result.mandatoryAnchors.end(),
            result.localExtrema.begin(),
            result.localExtrema.end()
        );
        result.mandatoryAnchors.insert(
            result.mandatoryAnchors.end(),
            result.inflectionPoints.begin(),
            result.inflectionPoints.end()
        );
    }

    // 4. Sort and deduplicate
    std::sort(result.mandatoryAnchors.begin(), result.mandatoryAnchors.end());
    result.mandatoryAnchors.erase(
        std::unique(result.mandatoryAnchors.begin(), result.mandatoryAnchors.end()),
        result.mandatoryAnchors.end()
    );

    return result;
}

double CurveFeatureDetector::estimateDerivative(const LayeredTransferFunction& ltf, int idx) {
    // Central difference (more accurate than forward/backward)
    if (idx == 0 || idx == ltf.getTableSize() - 1)
        return 0.0;  // Boundary handling

    double x0 = ltf.normalizeIndex(idx - 1);
    double x1 = ltf.normalizeIndex(idx + 1);
    double y0 = ltf.getCompositeValue(idx - 1);
    double y1 = ltf.getCompositeValue(idx + 1);

    return (y1 - y0) / (x1 - x0);
}

double CurveFeatureDetector::estimateSecondDerivative(const LayeredTransferFunction& ltf, int idx) {
    if (idx < 1 || idx >= ltf.getTableSize() - 1)
        return 0.0;

    double h = ltf.normalizeIndex(1) - ltf.normalizeIndex(0);  // Uniform spacing
    double y_prev = ltf.getCompositeValue(idx - 1);
    double y = ltf.getCompositeValue(idx);
    double y_next = ltf.getCompositeValue(idx + 1);

    return (y_next - 2.0 * y + y_prev) / (h * h);
}

} // namespace Services
} // namespace dsp_core
