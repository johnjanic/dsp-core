#include "SplineLayer.h"
#include "../services/SplineEvaluator.h"

namespace dsp_core {

SplineLayer::SplineLayer() {
    // Initialize with empty anchor vector (C++17 atomic store)
    std::atomic_store(&anchorsPtr, std::make_shared<const std::vector<SplineAnchor>>());
}

void SplineLayer::setAnchors(const std::vector<SplineAnchor>& anchors) {
    // Create new shared_ptr (heap allocation - acceptable on UI thread)
    auto newPtr = std::make_shared<const std::vector<SplineAnchor>>(anchors);

    // Atomic swap (lock-free, C++17 compatible)
    // Old ptr refcount decremented (if audio thread still using, deallocation deferred)
    std::atomic_store(&anchorsPtr, newPtr);
}

std::vector<SplineAnchor> SplineLayer::getAnchors() const {
    // Atomic load (lock-free, C++17 compatible)
    auto ptr = std::atomic_load(&anchorsPtr);

    // Return copy (safe snapshot)
    if (ptr) {
        return *ptr;
    }

    return {}; // Empty vector if null (defensive)
}

double SplineLayer::evaluate(double x) const {
    // Atomic load (lock-free, C++17 compatible)
    auto ptr = std::atomic_load(&anchorsPtr);

    // Null check (defensive, should never happen)
    if (!ptr || ptr->empty()) {
        return 0.0;
    }

    // Evaluate PCHIP spline (~30-40ns)
    return Services::SplineEvaluator::evaluate(*ptr, x);
}

juce::ValueTree SplineLayer::toValueTree() const {
    juce::ValueTree vt("SplineLayer");

    auto ptr = std::atomic_load(&anchorsPtr);
    if (ptr && !ptr->empty()) {
        juce::ValueTree anchorsVT("Anchors");

        for (const auto& anchor : *ptr) {
            juce::ValueTree anchorVT("Anchor");
            anchorVT.setProperty("x", anchor.x, nullptr);
            anchorVT.setProperty("y", anchor.y, nullptr);
            anchorVT.setProperty("tangent", anchor.tangent, nullptr);
            anchorVT.setProperty("hasCustomTangent", anchor.hasCustomTangent, nullptr);
            anchorsVT.addChild(anchorVT, -1, nullptr);
        }

        vt.addChild(anchorsVT, -1, nullptr);
    }

    return vt;
}

void SplineLayer::fromValueTree(const juce::ValueTree& vt) {
    if (!vt.isValid() || vt.getType().toString() != "SplineLayer") {
        return;
    }

    auto anchorsVT = vt.getChildWithName("Anchors");
    if (!anchorsVT.isValid()) {
        return;
    }

    std::vector<SplineAnchor> anchors;
    for (int i = 0; i < anchorsVT.getNumChildren(); ++i) {
        auto anchorVT = anchorsVT.getChild(i);
        if (anchorVT.getType().toString() == "Anchor") {
            SplineAnchor anchor;
            anchor.x = static_cast<double>(anchorVT.getProperty("x", 0.0));
            anchor.y = static_cast<double>(anchorVT.getProperty("y", 0.0));
            anchor.tangent = static_cast<double>(anchorVT.getProperty("tangent", 0.0));
            anchor.hasCustomTangent = static_cast<bool>(anchorVT.getProperty("hasCustomTangent", false));
            anchors.push_back(anchor);
        }
    }

    if (!anchors.empty()) {
        setAnchors(anchors);
    }
}

} // namespace dsp_core
