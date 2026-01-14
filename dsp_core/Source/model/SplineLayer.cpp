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

platform::PropertyTree SplineLayer::toPropertyTree() const {
    platform::PropertyTree tree("SplineLayer");

    auto ptr = std::atomic_load(&anchorsPtr);
    if (ptr && !ptr->empty()) {
        platform::PropertyTree anchorsTree("Anchors");

        for (const auto& anchor : *ptr) {
            platform::PropertyTree anchorTree("Anchor");
            anchorTree.setProperty("x", anchor.x);
            anchorTree.setProperty("y", anchor.y);
            anchorTree.setProperty("tangent", anchor.tangent);
            anchorTree.setProperty("hasCustomTangent", anchor.hasCustomTangent);
            anchorsTree.addChild(anchorTree);
        }

        tree.addChild(anchorsTree);
    }

    return tree;
}

void SplineLayer::fromPropertyTree(const platform::PropertyTree& tree) {
    if (!tree.isValid() || tree.getType() != "SplineLayer") {
        return;
    }

    const auto* anchorsTree = tree.getChildWithType("Anchors");
    if (anchorsTree == nullptr) {
        return;
    }

    std::vector<SplineAnchor> anchors;
    for (int i = 0; i < anchorsTree->getNumChildren(); ++i) {
        const auto& anchorTree = anchorsTree->getChild(i);
        if (anchorTree.getType() == "Anchor") {
            SplineAnchor anchor;
            anchor.x = anchorTree.getProperty<double>("x", 0.0);
            anchor.y = anchorTree.getProperty<double>("y", 0.0);
            anchor.tangent = anchorTree.getProperty<double>("tangent", 0.0);
            anchor.hasCustomTangent = anchorTree.getProperty<bool>("hasCustomTangent", false);
            anchors.push_back(anchor);
        }
    }

    if (!anchors.empty()) {
        setAnchors(anchors);
    }
}

} // namespace dsp_core
