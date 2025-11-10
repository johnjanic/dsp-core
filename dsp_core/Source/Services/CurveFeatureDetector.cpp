#include "CurveFeatureDetector.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>

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
    // Use actual endpoint value, not domain boundary
    double leftReferenceY = (feature.index > 0) ? ltf.getCompositeValue(0)
                                                 : (isPeak ? -1.0 : 1.0);
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
    // Use actual endpoint value, not domain boundary
    double rightReferenceY = (feature.index < tableSize - 1) ? ltf.getCompositeValue(tableSize - 1)
                                                               : (isPeak ? -1.0 : 1.0);
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

    // DEBUG: Log first few prominence calculations
    static int debugCount = 0;
    if (debugCount < 3 && feature.isExtremum) {
        std::cerr << "  Prominence calc: idx=" << feature.index << " y=" << featureY
                  << " leftRef=" << leftReferenceY << " rightRef=" << rightReferenceY
                  << " prom=" << prominence << std::endl;
        debugCount++;
    }

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
    // Use epsilon to avoid marking many near-equal features as "global"
    if (feature.isExtremum) {
        double featureY = ltf.getCompositeValue(feature.index);
        const double globalEpsilon = 0.005;  // 0.5% of range - only truly global extrema

        // Check if this is THE global maximum or THE global minimum (not just tied)
        bool isGlobalMax = true;
        bool isGlobalMin = true;
        for (const auto& other : allFeatures) {
            if (!other.isExtremum) continue;
            double otherY = ltf.getCompositeValue(other.index);
            if (otherY > featureY + globalEpsilon) isGlobalMax = false;
            if (otherY < featureY - globalEpsilon) isGlobalMin = false;
        }

        if (isGlobalMax || isGlobalMin) {
            // DEBUG: Log global extrema classification
            std::cerr << "  Global extremum at idx=" << feature.index << " y=" << featureY
                      << " (max=" << isGlobalMax << " min=" << isGlobalMin << ")" << std::endl;
            return FeatureTier::Mandatory;
        }
    }

    // Sharp inflection points are mandatory (very high curvature changes)
    // High threshold to prevent noise derivative spikes from being mandatory
    if (!feature.isExtremum && feature.significance > 1000.0) {
        return FeatureTier::Mandatory;
    }

    // Significant inflection points (moderate curvature changes)
    if (!feature.isExtremum && feature.significance > 0.1) {
        return FeatureTier::Significant;
    }

    // Gentle inflection points in smooth curves (round-trip stability)
    // If feature set is sparse, even low-curvature inflections are significant
    // This distinguishes smooth curves from noisy scribbles
    if (!feature.isExtremum && feature.significance > 0.005) {  // Lowered to catch gentle curves
        int nonEndpointFeatures = 0;
        for (const auto& other : allFeatures) {
            if (other.index != 0 && other.index != tableSize - 1) {
                nonEndpointFeatures++;
            }
        }

        // DEBUG: Log inflection point classification
        std::cerr << "  Inflection at idx=" << feature.index
                  << " sig=" << feature.significance
                  << " nonEndpointFeats=" << nonEndpointFeatures << std::endl;

        // Sparse feature set (<10) → likely smooth curve → include gentle inflections
        if (nonEndpointFeatures < 10) {
            std::cerr << "    -> Promoted to Significant (sparse feature set)" << std::endl;
            return FeatureTier::Significant;
        }
    }

    // Features with high prominence are significant
    // Increased threshold aggressively to filter noise on monotonic/near-monotonic curves
    if (feature.prominence > 0.50) {  // 50% of range - very strict for noise filtering
        return FeatureTier::Significant;
    }

    // REMOVED: moderate prominence with isolation exception
    // This was allowing too many noise features through in scribble test

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
    // Very low threshold for detection - prominence/classification filters noise
    const double derivativeThreshold = 1e-5;     // Detect all potential extrema
    const double secondDerivativeThreshold = 1e-3;  // Detect all potential inflection points

    // 1. Find local extrema using TWO methods for robustness:
    //    Method A: Derivative sign changes (catches sharp features)
    //    Method B: Local y-value min/max (catches smooth features)
    std::vector<Feature> features;
    std::set<int> extremaIndices;  // Track unique extrema

    // Method A: Derivative sign changes
    for (int i = 1; i < tableSize - 1; ++i) {
        double deriv_prev = estimateDerivative(ltf, i - 1);
        double deriv = estimateDerivative(ltf, i);

        // Detect sign changes (extrema). At least one derivative must be significant to avoid noise.
        if ((std::abs(deriv_prev) > derivativeThreshold || std::abs(deriv) > derivativeThreshold) &&
            deriv_prev * deriv < 0.0) {  // Sign change = local extremum
            extremaIndices.insert(i);
        }
    }

    // Method B: Local y-value min/max (for smooth curves)
    // Detect if change is above noise floor
    const double minExtremumHeight = 0.0001;  // Very low - let prominence filter noise

    for (int i = 1; i < tableSize - 1; ++i) {
        double y_prev = ltf.getCompositeValue(i - 1);
        double y = ltf.getCompositeValue(i);
        double y_next = ltf.getCompositeValue(i + 1);

        // Local maximum: y > both neighbors by threshold
        // Local minimum: y < both neighbors by threshold
        bool isLocalMax = (y - y_prev > minExtremumHeight && y - y_next > minExtremumHeight);
        bool isLocalMin = (y_prev - y > minExtremumHeight && y_next - y > minExtremumHeight);

        if (isLocalMax || isLocalMin) {
            extremaIndices.insert(i);
        }
    }

    // Convert to features list
    for (int idx : extremaIndices) {
        result.localExtrema.push_back(idx);
        double y = ltf.getCompositeValue(idx);
        features.push_back({idx, std::abs(y), 0.0, true});  // prominence computed later
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

            // Significance = curvature magnitude (use max of prev/current for better metric)
            double curvatureMagnitude = std::max(std::abs(d2y_prev), std::abs(d2y));

            // DEBUG: Log first few inflection points
            if (result.inflectionPoints.size() <= 3) {
                std::cerr << "  Detected inflection at idx=" << i
                          << " d2y_prev=" << d2y_prev
                          << " d2y=" << d2y
                          << " significance=" << curvatureMagnitude << std::endl;
            }

            features.push_back({i, curvatureMagnitude, 0.0, false});  // prominence computed later
        }
    }

    // 2.5. Merge nearby features to eliminate redundant detections (noise clustering)
    // Use adaptive merge radius based on feature density:
    // - High density (>200 features) → aggressive merging for noise
    // - Medium density (50-200) → moderate merging for harmonics
    // - Low density (<50) → minimal merging for smooth curves
    int mergeRadiusSamples;
    int featureCount = static_cast<int>(features.size());
    if (featureCount > 200) {
        mergeRadiusSamples = 48;  // ~19% of domain - scribble/noise
    } else if (featureCount > 50) {
        mergeRadiusSamples = 16;  // ~6% of domain - harmonics
    } else {
        mergeRadiusSamples = 8;   // ~3% of domain - smooth curves
    }

    std::vector<Feature> mergedFeatures;

    // Sort features by index for efficient merging
    std::sort(features.begin(), features.end(),
              [](const Feature& a, const Feature& b) { return a.index < b.index; });

    for (const auto& feature : features) {
        bool merged = false;

        // Try to merge with existing features of the same type
        for (auto& existing : mergedFeatures) {
            if (feature.isExtremum != existing.isExtremum) continue;

            int distance = std::abs(existing.index - feature.index);
            if (distance <= mergeRadiusSamples) {
                // Features are close - keep the more extreme one
                double featureY = ltf.getCompositeValue(feature.index);
                double existingY = ltf.getCompositeValue(existing.index);

                // For extrema: keep the one farther from zero (more extreme)
                // For inflections: keep the one with higher curvature (significance)
                bool replaceExisting = false;
                if (feature.isExtremum) {
                    replaceExisting = (std::abs(featureY) > std::abs(existingY));
                } else {
                    replaceExisting = (feature.significance > existing.significance);
                }

                if (replaceExisting) {
                    existing = feature;
                }
                merged = true;
                break;
            }
        }

        if (!merged) {
            mergedFeatures.push_back(feature);
        }
    }

    // Update features and result lists after merging
    int originalFeatureCount = static_cast<int>(features.size());
    features = mergedFeatures;

    // Rebuild localExtrema and inflectionPoints from merged features
    result.localExtrema.clear();
    result.inflectionPoints.clear();
    for (const auto& feature : features) {
        if (feature.isExtremum) {
            result.localExtrema.push_back(feature.index);
        } else {
            result.inflectionPoints.push_back(feature.index);
        }
    }

    // DEBUG: Log merging results
    std::cerr << "  Feature merging: " << originalFeatureCount
              << " → " << features.size() << " features" << std::endl;

    // 2.6. Compute prominence for all features
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

        // DEBUG: Log first few extrema classifications
        if (feature.isExtremum && mandatoryFeatures.size() + significantFeatures.size() < 5) {
            double y = ltf.getCompositeValue(feature.index);
            std::cerr << "  Extremum at idx=" << feature.index << " y=" << y
                      << " prom=" << feature.prominence << " tier="
                      << (tier == FeatureTier::Mandatory ? "Mandatory" :
                          tier == FeatureTier::Significant ? "Significant" : "Minor") << std::endl;
        }

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

    // DEBUG: Temporary output
    std::cerr << "DEBUG: CurveFeatureDetector detected " << features.size() << " total features, "
        << result.localExtrema.size() << " extrema, "
        << result.mandatoryAnchors.size() << " mandatory anchors" << std::endl;
    std::cerr << "  Mandatory: " << mandatoryFeatures.size() << ", Significant: " << significantFeatures.size()
        << ", Minor: " << minorFeatures.size() << std::endl;

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
