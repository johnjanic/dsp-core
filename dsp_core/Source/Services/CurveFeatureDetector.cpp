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

double CurveFeatureDetector::computeProminence(
    const Feature& feature,
    const std::vector<Feature>& allFeatures,
    const LayeredTransferFunction& ltf) {

    if (!feature.isExtremum) {
        // Inflection points use curvature magnitude as prominence
        return feature.significance;
    }

    double featureY = ltf.getCompositeValue(feature.index);
    bool isPeak = true;  // Assume peak initially

    // Determine if this is a peak or valley by checking neighbors
    int tableSize = ltf.getTableSize();
    if (feature.index > 0 && feature.index < tableSize - 1) {
        double prevY = ltf.getCompositeValue(feature.index - 1);
        double nextY = ltf.getCompositeValue(feature.index + 1);
        isPeak = (featureY > prevY && featureY > nextY);
    }

    // Find nearest valley on left side (for peaks) or peak on left (for valleys)
    double leftReferenceY = isPeak ? -1.0 : 1.0;  // Worst case: domain boundary
    for (const auto& other : allFeatures) {
        if (other.index >= feature.index) break;  // Only look left
        if (!other.isExtremum) continue;

        double otherY = ltf.getCompositeValue(other.index);
        bool otherIsPeak = true;
        if (other.index > 0 && other.index < tableSize - 1) {
            double prevY = ltf.getCompositeValue(other.index - 1);
            double nextY = ltf.getCompositeValue(other.index + 1);
            otherIsPeak = (otherY > prevY && otherY > nextY);
        }

        // For peaks, find valleys; for valleys, find peaks
        if (isPeak && !otherIsPeak) {
            leftReferenceY = std::max(leftReferenceY, otherY);
        } else if (!isPeak && otherIsPeak) {
            leftReferenceY = std::min(leftReferenceY, otherY);
        }
    }

    // Find nearest valley on right side (for peaks) or peak on right (for valleys)
    double rightReferenceY = isPeak ? -1.0 : 1.0;  // Worst case: domain boundary
    for (const auto& other : allFeatures) {
        if (other.index <= feature.index) continue;  // Only look right
        if (!other.isExtremum) continue;

        double otherY = ltf.getCompositeValue(other.index);
        bool otherIsPeak = true;
        if (other.index > 0 && other.index < tableSize - 1) {
            double prevY = ltf.getCompositeValue(other.index - 1);
            double nextY = ltf.getCompositeValue(other.index + 1);
            otherIsPeak = (otherY > prevY && otherY > nextY);
        }

        // For peaks, find valleys; for valleys, find peaks
        if (isPeak && !otherIsPeak) {
            rightReferenceY = std::max(rightReferenceY, otherY);
        } else if (!isPeak && otherIsPeak) {
            rightReferenceY = std::min(rightReferenceY, otherY);
        }
    }

    // Prominence = height above higher valley (peaks) or depth below lower peak (valleys)
    double referenceLevel = isPeak ? std::max(leftReferenceY, rightReferenceY)
                                    : std::min(leftReferenceY, rightReferenceY);
    double prominence = std::abs(featureY - referenceLevel);

    return prominence;
}

CurveFeatureDetector::FeatureTier CurveFeatureDetector::classifyFeature(
    const Feature& feature,
    const std::vector<Feature>& allFeatures,
    const LayeredTransferFunction& ltf,
    int tableSize) {

    // Endpoints are always mandatory (handled separately)
    if (feature.index == 0 || feature.index == tableSize - 1) {
        return FeatureTier::Mandatory;
    }

    // Global extrema are mandatory (highest peak, lowest valley)
    if (feature.isExtremum) {
        double featureY = ltf.getCompositeValue(feature.index);

        // Check if this is a global maximum or minimum
        bool isGlobalMax = true;
        bool isGlobalMin = true;
        for (const auto& other : allFeatures) {
            if (!other.isExtremum) continue;
            double otherY = ltf.getCompositeValue(other.index);
            if (otherY > featureY) isGlobalMax = false;
            if (otherY < featureY) isGlobalMin = false;
        }

        if (isGlobalMax || isGlobalMin) {
            return FeatureTier::Mandatory;
        }
    }

    // Sharp inflection points are mandatory (high curvature changes)
    if (!feature.isExtremum && feature.significance > 0.5) {
        return FeatureTier::Mandatory;
    }

    // Features with high prominence are significant
    if (feature.prominence > 0.2) {  // 20% of range
        return FeatureTier::Significant;
    }

    // Features with moderate prominence and isolation are significant
    if (feature.prominence > 0.1) {  // 10% of range
        // Check isolation (distance to nearest feature)
        int minDistance = tableSize;
        for (const auto& other : allFeatures) {
            if (other.index == feature.index) continue;
            minDistance = std::min(minDistance, std::abs(other.index - feature.index));
        }
        double isolation = static_cast<double>(minDistance) / tableSize;

        if (isolation > 0.1) {  // > 10% of domain
            return FeatureTier::Significant;
        }
    }

    // Everything else is minor (small wiggles, scribble noise)
    return FeatureTier::Minor;
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
            features.push_back({i, std::abs(y), 0.0, true});  // prominence computed later
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
            features.push_back({i, std::abs(d2y), 0.0, false});  // prominence computed later
        }
    }

    // 2.5. Compute prominence for all features
    for (auto& feature : features) {
        feature.prominence = computeProminence(feature, features, ltf);
    }

    // 3. Prioritize and limit features using tiered selection
    result.mandatoryAnchors.push_back(0);  // Always include endpoints
    result.mandatoryAnchors.push_back(tableSize - 1);

    // Apply local density constraint if enabled (localDensityWindowSize > 0)
    const bool constraintEnabled = (localDensityWindowSize > 0.0 && maxAnchorsPerWindow > 0);

    // Classify all features by tier
    std::vector<Feature> mandatoryFeatures, significantFeatures, minorFeatures;
    for (auto& feature : features) {
        FeatureTier tier = classifyFeature(feature, features, ltf, tableSize);
        if (tier == FeatureTier::Mandatory) {
            mandatoryFeatures.push_back(feature);
        } else if (tier == FeatureTier::Significant) {
            significantFeatures.push_back(feature);
        } else {
            minorFeatures.push_back(feature);
        }
    }

    // Sort significant features by prominence (descending)
    auto byProminence = [](const Feature& a, const Feature& b) {
        return a.prominence > b.prominence;
    };
    std::sort(significantFeatures.begin(), significantFeatures.end(), byProminence);

    // Step 3a: Add all mandatory features (no limit, always include)
    for (const auto& feature : mandatoryFeatures) {
        result.mandatoryAnchors.push_back(feature.index);
    }

    // Step 3b: Add significant features up to budget
    // Calculate remaining budget after mandatory features
    int remainingBudget = maxMandatoryAnchors > 0
        ? maxMandatoryAnchors - static_cast<int>(result.mandatoryAnchors.size())
        : static_cast<int>(significantFeatures.size());  // No limit

    for (const auto& feature : significantFeatures) {
        if (remainingBudget <= 0) break;

        int candidateIdx = feature.index;

        // Check density constraints
        bool satisfiesConstraint = true;
        if (constraintEnabled) {
            // Coarse window constraint
            int localDensity = countAnchorsInWindow(
                candidateIdx,
                result.mandatoryAnchors,
                tableSize,
                localDensityWindowSize
            );

            // REMOVED: 70% conservative threshold
            // Use full window budget since error-driven refinement is separate phase
            if (localDensity >= maxAnchorsPerWindow) {
                satisfiesConstraint = false;
            }

            // Fine window constraint (pixel-level clustering prevention)
            if (satisfiesConstraint && localDensityWindowSizeFine > 0.0 && maxAnchorsPerWindowFine > 0) {
                int fineDensity = countAnchorsInWindow(
                    candidateIdx,
                    result.mandatoryAnchors,
                    tableSize,
                    localDensityWindowSizeFine
                );
                if (fineDensity >= maxAnchorsPerWindowFine) {
                    satisfiesConstraint = false;
                }
            }
        }

        if (satisfiesConstraint) {
            result.mandatoryAnchors.push_back(candidateIdx);
            remainingBudget--;
        }
    }

    // Step 3c: Minor features are intentionally skipped for sparseness

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
