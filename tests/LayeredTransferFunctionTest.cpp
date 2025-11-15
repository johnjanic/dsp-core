#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <thread>
#include <atomic>

namespace dsp_core_test {

/**
 * Test fixture for LayeredTransferFunction
 * Tests normalization, deferred normalization, harmonic mixing, and thread safety
 */
class LayeredTransferFunctionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ltf = std::make_unique<dsp_core::LayeredTransferFunction>(256, -1.0, 1.0);
    }

    std::unique_ptr<dsp_core::LayeredTransferFunction> ltf;
};

// ============================================================================
// Normalization Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, UpdateComposite_NormalizesToRange) {
    // Set extreme base layer values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 10.0);  // Way out of range
    }

    ltf->updateComposite();

    // Composite should be normalized to [-1, 1]
    for (int i = 0; i < 256; ++i) {
        double value = ltf->getCompositeValue(i);
        EXPECT_GE(value, -1.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST_F(LayeredTransferFunctionTest, UpdateComposite_NormalizesNegativeValues) {
    // Set extreme negative base layer values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, -5.0);  // Way out of range
    }

    ltf->updateComposite();

    // Composite should be normalized to [-1, 1]
    for (int i = 0; i < 256; ++i) {
        double value = ltf->getCompositeValue(i);
        EXPECT_GE(value, -1.0);
        EXPECT_LE(value, 1.0);
    }
}

TEST_F(LayeredTransferFunctionTest, UpdateComposite_MaintainsRelativeProportions) {
    // Set base layer with known proportions
    ltf->setBaseLayerValue(100, 2.0);
    ltf->setBaseLayerValue(150, 4.0);  // Double the value at index 100
    ltf->setCoefficient(0, 1.0);  // 100% WT mix

    ltf->updateComposite();

    // After normalization, proportions should be maintained
    double value100 = ltf->getCompositeValue(100);
    double value150 = ltf->getCompositeValue(150);

    EXPECT_NEAR(value150 / value100, 2.0, 0.01);  // Ratio should be ~2.0
}

// ============================================================================
// Deferred Normalization Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, DeferNormalization_FreezesScalar) {
    // Set initial state
    ltf->setBaseLayerValue(100, 0.5);
    ltf->setCoefficient(0, 1.0);  // 100% WT mix
    ltf->updateComposite();
    double composite1 = ltf->getCompositeValue(100);
    double scalar1 = ltf->getNormalizationScalar();

    // Enable deferred normalization
    ltf->setDeferNormalization(true);
    EXPECT_TRUE(ltf->isNormalizationDeferred());

    // Change base layer significantly
    ltf->setBaseLayerValue(100, 5.0);
    ltf->updateComposite();

    // Normalization scalar should be frozen
    double scalar2 = ltf->getNormalizationScalar();
    EXPECT_NEAR(scalar1, scalar2, 1e-6);

    // Composite should change but use frozen scalar
    double composite2 = ltf->getCompositeValue(100);
    EXPECT_NE(composite1, composite2);  // Value changed
}

TEST_F(LayeredTransferFunctionTest, DeferNormalization_ResumesAfterDisable) {
    // Set initial state
    ltf->setBaseLayerValue(100, 0.5);
    ltf->setCoefficient(0, 1.0);
    ltf->updateComposite();
    double scalar1 = ltf->getNormalizationScalar();

    // Enable deferred normalization
    ltf->setDeferNormalization(true);

    // Change base layer
    ltf->setBaseLayerValue(100, 5.0);
    ltf->updateComposite();

    // Scalar should be frozen
    double scalar2 = ltf->getNormalizationScalar();
    EXPECT_NEAR(scalar1, scalar2, 1e-6);

    // Disable deferred normalization
    ltf->setDeferNormalization(false);
    EXPECT_FALSE(ltf->isNormalizationDeferred());

    // Update composite again - scalar should now update
    ltf->updateComposite();
    double scalar3 = ltf->getNormalizationScalar();
    EXPECT_NE(scalar1, scalar3);  // Scalar should have changed
}

TEST_F(LayeredTransferFunctionTest, DeferNormalization_PreventsVisualJumping) {
    // Simulate paint stroke scenario
    ltf->setCoefficient(0, 1.0);  // 100% WT mix

    // Set initial values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 0.5);
    }
    ltf->updateComposite();

    double initialScalar = ltf->getNormalizationScalar();

    // Begin paint stroke (defer normalization)
    ltf->setDeferNormalization(true);

    // Paint several points with large values
    for (int i = 100; i < 110; ++i) {
        ltf->setBaseLayerValue(i, 2.0);
        ltf->updateComposite();
    }

    // Scalar should remain frozen
    double paintingScalar = ltf->getNormalizationScalar();
    EXPECT_NEAR(initialScalar, paintingScalar, 1e-6);

    // End paint stroke
    ltf->setDeferNormalization(false);
    ltf->updateComposite();

    // Now scalar can update
    double finalScalar = ltf->getNormalizationScalar();
    EXPECT_LT(finalScalar, initialScalar);  // Larger values require smaller scalar
}

// ============================================================================
// Harmonic Mixing Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, HarmonicMixing_WeightedSum) {
    // Set WT coefficient to 0.6, first harmonic to 0.4
    ltf->setCoefficient(0, 0.6);  // WT mix
    ltf->setCoefficient(1, 0.4);  // First harmonic

    // Set known base layer value
    ltf->setBaseLayerValue(100, 1.0);

    ltf->updateComposite();

    // Composite should be weighted sum (after normalization)
    // Exact value depends on harmonic evaluation, but should be in range
    double composite = ltf->getCompositeValue(100);
    EXPECT_GE(composite, -1.0);
    EXPECT_LE(composite, 1.0);
}

TEST_F(LayeredTransferFunctionTest, HarmonicMixing_ZeroWTCoeff) {
    // Set WT coefficient to 0, harmonic to 1.0 (pure harmonic mode)
    ltf->setCoefficient(0, 0.0);  // No WT
    ltf->setCoefficient(1, 1.0);  // 100% first harmonic

    // Set base layer values (should have no effect)
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 1.0);
    }

    ltf->updateComposite();

    // Composite should only reflect harmonic contribution
    // (base layer is ignored when WT coeff = 0)
    for (int i = 0; i < 256; ++i) {
        double composite = ltf->getCompositeValue(i);
        EXPECT_GE(composite, -1.0);
        EXPECT_LE(composite, 1.0);
    }
}

TEST_F(LayeredTransferFunctionTest, HarmonicMixing_MultipleHarmonics) {
    // Set multiple harmonic coefficients
    ltf->setCoefficient(0, 0.5);  // WT mix
    ltf->setCoefficient(1, 0.3);  // First harmonic
    ltf->setCoefficient(2, 0.2);  // Second harmonic

    // Set base layer
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }

    ltf->updateComposite();

    // All composites should be in valid range
    for (int i = 0; i < 256; ++i) {
        double composite = ltf->getCompositeValue(i);
        EXPECT_GE(composite, -1.0);
        EXPECT_LE(composite, 1.0);
    }
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, ThreadSafety_AtomicReads) {
    // Write from main thread
    ltf->setBaseLayerValue(100, 0.5);
    ltf->setCoefficient(0, 1.0);
    ltf->updateComposite();

    // Simulate audio thread read (lock-free)
    std::atomic<bool> readComplete{false};
    std::thread audioThread([&]() {
        for (int i = 0; i < 1000; ++i) {
            double value = ltf->getCompositeValue(100);
            EXPECT_GE(value, -1.0);
            EXPECT_LE(value, 1.0);
        }
        readComplete.store(true);
    });

    // Main thread continues writing
    for (int i = 0; i < 100; ++i) {
        ltf->setBaseLayerValue(100, static_cast<double>(i) / 100.0);
        ltf->updateComposite();
    }

    audioThread.join();
    EXPECT_TRUE(readComplete.load());
}

TEST_F(LayeredTransferFunctionTest, ThreadSafety_MultipleIndices) {
    // Initialize values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }
    ltf->setCoefficient(0, 1.0);
    ltf->updateComposite();

    // Audio thread reads multiple indices
    std::atomic<bool> readComplete{false};
    std::atomic<int> readCount{0};

    std::thread audioThread([&]() {
        for (int iteration = 0; iteration < 100; ++iteration) {
            for (int i = 0; i < 256; ++i) {
                double value = ltf->getCompositeValue(i);
                EXPECT_GE(value, -1.0);
                EXPECT_LE(value, 1.0);
                readCount.fetch_add(1);
            }
        }
        readComplete.store(true);
    });

    // Main thread updates while audio thread reads
    for (int iteration = 0; iteration < 10; ++iteration) {
        for (int i = 0; i < 256; ++i) {
            ltf->setBaseLayerValue(i, static_cast<double>(iteration) / 10.0);
        }
        ltf->updateComposite();
    }

    audioThread.join();
    EXPECT_TRUE(readComplete.load());
    EXPECT_EQ(readCount.load(), 100 * 256);  // All reads completed
}

// ============================================================================
// Base Layer Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, ClearBaseLayer_SetsAllToZero) {
    // Set base layer values
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }

    // Clear base layer
    ltf->clearBaseLayer();

    // Verify all values are zero
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), 0.0, 1e-6);
    }
}

// ============================================================================
// Coefficient Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, Coefficients_GetSetConsistent) {
    // Set coefficients
    for (int i = 0; i < ltf->getNumCoefficients(); ++i) {
        ltf->setCoefficient(i, static_cast<double>(i) / 10.0);
    }

    // Verify get returns set values
    for (int i = 0; i < ltf->getNumCoefficients(); ++i) {
        EXPECT_NEAR(ltf->getCoefficient(i), static_cast<double>(i) / 10.0, 1e-6);
    }
}

// ============================================================================
// Harmonic Baking Tests
// ============================================================================

TEST_F(LayeredTransferFunctionTest, HasNonZeroHarmonics_ReturnsFalseForAllZero) {
    // Set WT mix but no harmonics
    ltf->setCoefficient(0, 0.8);  // WT mix (not a harmonic)

    // All harmonics (indices 1-40) are zero
    for (int i = 1; i <= 40; ++i) {
        ltf->setCoefficient(i, 0.0);
    }

    EXPECT_FALSE(ltf->hasNonZeroHarmonics());
}

TEST_F(LayeredTransferFunctionTest, HasNonZeroHarmonics_ReturnsTrueForNonZero) {
    // Set one harmonic to non-zero
    ltf->setCoefficient(3, 0.5);

    EXPECT_TRUE(ltf->hasNonZeroHarmonics());
}

TEST_F(LayeredTransferFunctionTest, HasNonZeroHarmonics_UsesEpsilonThreshold) {
    // Set harmonic below epsilon threshold (should be treated as zero)
    ltf->setCoefficient(3, 1e-7);  // Below HARMONIC_EPSILON (1e-6)

    EXPECT_FALSE(ltf->hasNonZeroHarmonics());

    // Set harmonic above epsilon threshold
    ltf->setCoefficient(3, 1e-5);  // Above HARMONIC_EPSILON

    EXPECT_TRUE(ltf->hasNonZeroHarmonics());
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_NoOpForZeroHarmonics) {
    // Set WT mix and base layer
    ltf->setCoefficient(0, 1.0);
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 256.0);
    }
    ltf->updateComposite();

    // Capture base layer before baking
    std::vector<double> baseLayerBefore(256);
    for (int i = 0; i < 256; ++i) {
        baseLayerBefore[i] = ltf->getBaseLayerValue(i);
    }

    // Bake with all-zero harmonics
    bool baked = ltf->bakeHarmonicsToBase();

    // Should return false (no-op)
    EXPECT_FALSE(baked);

    // Base layer should be unchanged
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), baseLayerBefore[i], 1e-12);
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_TransfersCompositeToBase) {
    // Set base layer and harmonics
    ltf->setCoefficient(0, 1.0);  // Full WT mix
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, static_cast<double>(i) / 512.0);  // Half range
    }
    ltf->setCoefficient(3, 0.5);  // Add 3rd harmonic
    ltf->updateComposite();

    // Capture composite BEFORE baking
    std::vector<double> compositeBefore(256);
    for (int i = 0; i < 256; ++i) {
        compositeBefore[i] = ltf->getCompositeValue(i);
    }

    // Bake
    bool baked = ltf->bakeHarmonicsToBase();

    EXPECT_TRUE(baked);

    // Base layer should now match old composite (visual identity preserved)
    for (int i = 0; i < 256; ++i) {
        double baseAfter = ltf->getBaseLayerValue(i);
        EXPECT_NEAR(compositeBefore[i], baseAfter, 1e-12)
            << "Visual discontinuity at index " << i;
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_ZerosHarmonicCoefficients) {
    // Set harmonics
    ltf->setCoefficient(3, 0.5);
    ltf->setCoefficient(5, 0.3);
    ltf->setCoefficient(7, 0.2);

    EXPECT_TRUE(ltf->hasNonZeroHarmonics());

    // Bake
    ltf->bakeHarmonicsToBase();

    // All harmonics should be zero
    EXPECT_FALSE(ltf->hasNonZeroHarmonics());

    // Verify individual harmonics are zero
    for (int i = 1; i <= 40; ++i) {
        EXPECT_NEAR(ltf->getCoefficient(i), 0.0, 1e-12);
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_SetsWTToOne) {
    // Set WT mix to 0 (common case in Harmonic mode)
    ltf->setCoefficient(0, 0.0);
    ltf->setCoefficient(3, 0.5);

    // Bake
    ltf->bakeHarmonicsToBase();

    // WT coefficient MUST be set to 1.0 to make the baked base layer visible
    // If WT remains at 0, the result would be: composite = 0 * base = 0 (flat line bug)
    EXPECT_NEAR(ltf->getCoefficient(0), 1.0, 1e-12);
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_IdempotentMultipleCalls) {
    // Set harmonics and bake
    ltf->setCoefficient(3, 0.5);
    ltf->bakeHarmonicsToBase();

    // Capture base layer after first bake
    std::vector<double> baseAfterFirstBake(256);
    for (int i = 0; i < 256; ++i) {
        baseAfterFirstBake[i] = ltf->getBaseLayerValue(i);
    }

    // Second bake should be no-op
    bool secondBake = ltf->bakeHarmonicsToBase();
    EXPECT_FALSE(secondBake);

    // Base layer should be unchanged
    for (int i = 0; i < 256; ++i) {
        EXPECT_NEAR(ltf->getBaseLayerValue(i), baseAfterFirstBake[i], 1e-12);
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_RecalculatesNormalizationScalar) {
    // Set up curve that requires normalization
    ltf->setCoefficient(0, 1.0);
    for (int i = 0; i < 256; ++i) {
        ltf->setBaseLayerValue(i, 2.0);  // Out of range
    }
    ltf->setCoefficient(3, 0.5);
    ltf->updateComposite();

    // Capture normalization scalar before baking
    double normScalarBefore = ltf->getNormalizationScalar();
    EXPECT_LT(normScalarBefore, 1.0);  // Should be scaled down

    // Capture composite values (for visual continuity check)
    std::vector<double> compositeBefore(256);
    for (int i = 0; i < 256; ++i) {
        compositeBefore[i] = ltf->getCompositeValue(i);
    }

    // Bake
    ltf->bakeHarmonicsToBase();

    // Normalization scalar recalculates because base layer now contains normalized values
    // This is correct behavior - the scalar adapts to the new base layer state
    double normScalarAfter = ltf->getNormalizationScalar();
    EXPECT_NEAR(normScalarAfter, 1.0, 0.1);  // Should be close to 1.0 now

    // What matters is visual continuity (composite unchanged)
    for (int i = 0; i < 256; ++i) {
        EXPECT_DOUBLE_EQ(compositeBefore[i], ltf->getCompositeValue(i));
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_VisualContinuityBitLevel) {
    // Set complex harmonic configuration
    ltf->setCoefficient(0, 1.0);
    ltf->setCoefficient(1, 0.3);
    ltf->setCoefficient(3, 0.5);
    ltf->setCoefficient(5, 0.2);
    ltf->updateComposite();

    // Capture composite BEFORE baking
    std::vector<double> compositeBefore(256);
    for (int i = 0; i < 256; ++i) {
        compositeBefore[i] = ltf->getCompositeValue(i);
    }

    // Bake
    ltf->bakeHarmonicsToBase();

    // Composite AFTER baking should be identical (bit-level)
    for (int i = 0; i < 256; ++i) {
        double compositeAfter = ltf->getCompositeValue(i);
        EXPECT_DOUBLE_EQ(compositeBefore[i], compositeAfter)
            << "Visual discontinuity at index " << i;
    }
}

TEST_F(LayeredTransferFunctionTest, BakeHarmonics_Performance) {
    // Setup: non-trivial harmonics
    ltf->setCoefficient(0, 1.0);
    for (int h = 1; h <= 40; ++h) {
        ltf->setCoefficient(h, 0.1 / h);
    }
    ltf->updateComposite();

    // Measure baking time
    auto start = std::chrono::high_resolution_clock::now();
    ltf->bakeHarmonicsToBase();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be <1ms as claimed in feature plan
    EXPECT_LT(duration.count(), 1000) << "Baking took " << duration.count() << "μs";
}

TEST_F(LayeredTransferFunctionTest, GetHarmonicCoefficients_ReturnsAllCoefficients) {
    // Set known coefficients
    ltf->setCoefficient(0, 0.8);  // WT mix
    ltf->setCoefficient(1, 0.3);
    ltf->setCoefficient(3, 0.5);
    ltf->setCoefficient(40, 0.1);

    // Get all coefficients
    auto coeffs = ltf->getHarmonicCoefficients();

    // Verify size
    EXPECT_EQ(coeffs.size(), 41u);

    // Verify values
    EXPECT_NEAR(coeffs[0], 0.8, 1e-12);  // WT mix
    EXPECT_NEAR(coeffs[1], 0.3, 1e-12);
    EXPECT_NEAR(coeffs[3], 0.5, 1e-12);
    EXPECT_NEAR(coeffs[40], 0.1, 1e-12);
}

TEST_F(LayeredTransferFunctionTest, SetHarmonicCoefficients_SetsAllCoefficients) {
    // Create coefficient array
    std::array<double, 41> coeffs{};
    coeffs[0] = 0.9;  // WT mix
    coeffs[1] = 0.4;
    coeffs[3] = 0.6;
    coeffs[40] = 0.2;

    // Set all coefficients
    ltf->setHarmonicCoefficients(coeffs);

    // Verify they were set
    EXPECT_NEAR(ltf->getCoefficient(0), 0.9, 1e-12);
    EXPECT_NEAR(ltf->getCoefficient(1), 0.4, 1e-12);
    EXPECT_NEAR(ltf->getCoefficient(3), 0.6, 1e-12);
    EXPECT_NEAR(ltf->getCoefficient(40), 0.2, 1e-12);
}

TEST_F(LayeredTransferFunctionTest, SetHarmonicCoefficients_UpdatesComposite) {
    // Set initial state
    ltf->setCoefficient(0, 1.0);
    ltf->updateComposite();
    double compositeBefore = ltf->getCompositeValue(100);

    // Set new coefficients via array
    std::array<double, 41> coeffs{};
    coeffs[0] = 1.0;
    coeffs[3] = 0.5;  // Add 3rd harmonic
    ltf->setHarmonicCoefficients(coeffs);

    // Composite should have changed (harmonics added)
    double compositeAfter = ltf->getCompositeValue(100);
    EXPECT_NE(compositeBefore, compositeAfter);
}

TEST_F(LayeredTransferFunctionTest, GetSetHarmonicCoefficients_RoundTrip) {
    // Set known coefficients
    for (int i = 0; i <= 40; ++i) {
        ltf->setCoefficient(i, static_cast<double>(i) / 100.0);
    }

    // Get coefficients
    auto coeffs = ltf->getHarmonicCoefficients();

    // Clear coefficients
    std::array<double, 41> zeros{};
    ltf->setHarmonicCoefficients(zeros);

    // Restore original coefficients
    ltf->setHarmonicCoefficients(coeffs);

    // Verify restoration
    for (int i = 0; i <= 40; ++i) {
        EXPECT_NEAR(ltf->getCoefficient(i), static_cast<double>(i) / 100.0, 1e-12);
    }
}

}  // namespace dsp_core_test
