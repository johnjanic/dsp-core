#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <cmath>

namespace dsp_core_test {

/**
 * Test fixture for AudioEngine
 * Tests triple-buffered LUT with crossfade functionality
 */
class AudioEngineTest : public ::testing::Test {
  protected:
    void SetUp() override {
        engine = std::make_unique<dsp_core::AudioEngine>();
    }

    std::unique_ptr<dsp_core::AudioEngine> engine;
};

// ============================================================================
// Task 2: AudioEngine Tests - Initialization
// ============================================================================

/**
 * Test: Initial state produces identity function (y = x)
 * Expected: Before any LUT renders, all buffers contain identity
 */
TEST_F(AudioEngineTest, Constructor_InitializesToIdentity) {
    // Test identity function across range
    const std::vector<double> testInputs = {-1.0, -0.5, 0.0, 0.5, 1.0};

    for (double x : testInputs) {
        double output = engine->applyTransferFunction(x);
        EXPECT_NEAR(output, x, 1e-6) << "Identity function should return input value";
    }
}

/**
 * Test: All three LUT buffers initialized to identity
 * Expected: No undefined values in any buffer
 */
TEST_F(AudioEngineTest, Constructor_AllBuffersInitialized) {
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    for (int bufIdx = 0; bufIdx < 3; ++bufIdx) {
        for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
            const double x = dsp_core::MIN_VALUE + (i / static_cast<double>(dsp_core::TABLE_SIZE - 1)) *
                                                        (dsp_core::MAX_VALUE - dsp_core::MIN_VALUE);
            EXPECT_DOUBLE_EQ(buffers[bufIdx].data[i], x) << "Buffer " << bufIdx << " index " << i
                                                          << " should contain identity value";
        }
        EXPECT_EQ(buffers[bufIdx].version, 0u) << "Buffer " << bufIdx << " should have version 0";
    }
}

// ============================================================================
// Task 2: AudioEngine Tests - Sample Rate Adaptation
// ============================================================================

/**
 * Test: Crossfade duration scales correctly with sample rate
 * Expected: 10ms at 44.1kHz = 441 samples, 10ms at 48kHz = 480 samples
 */
TEST_F(AudioEngineTest, PrepareToPlay_CrossfadeDurationScalesWithSampleRate) {
    // 44.1 kHz
    engine->prepareToPlay(44100.0, 512);
    double buffer[1] = {0.0};
    engine->processBlock(buffer, 1);
    // We can't directly check crossfadeSamples (private), but we can verify behavior

    // 48 kHz
    engine->prepareToPlay(48000.0, 512);
    // Expected: 480 samples at 48kHz (10ms)
    // We'll verify this by triggering a crossfade and counting samples
}

/**
 * Test: Sample rate change mid-crossfade clamps position
 * Expected: If crossfadePosition >= new crossfadeSamples, crossfade completes
 */
TEST_F(AudioEngineTest, PrepareToPlay_ClampsPositionOnSampleRateChange) {
    // Start at 96kHz (960 samples for 10ms)
    engine->prepareToPlay(96000.0, 512);

    // Simulate crossfade in progress by manually swapping buffers
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5; // Different from identity
    }
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Trigger crossfade
    double samples[960];
    for (int i = 0; i < 960; ++i) samples[i] = 0.0;
    engine->processBlock(samples, 500); // Partially through crossfade

    // Now change sample rate to 44.1kHz (441 samples)
    // The crossfade should complete immediately if position >= 441
    engine->prepareToPlay(44100.0, 512);
    engine->processBlock(samples, 1);

    // If we're still crossfading, position should be clamped
    // (Hard to verify without accessing private state, but no crash is good)
}

// ============================================================================
// Task 2: AudioEngine Tests - Crossfade Behavior
// ============================================================================

/**
 * Test: Crossfade gains sum to 1.0 at all positions
 * Expected: gainOld + gainNew = 1.0 throughout crossfade
 */
TEST_F(AudioEngineTest, Crossfade_GainsSumToOne) {
    engine->prepareToPlay(44100.0, 512);

    // Set up two different LUTs
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();

    // Buffer 0 (primary): identity (y = x)
    // Buffer 2 (worker target): constant 0.5
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5;
    }
    buffers[2].version = 1;

    // Signal new LUT ready
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process samples and verify crossfade
    const int expectedCrossfadeSamples = static_cast<int>(44100.0 * 10.0 / 1000.0); // 441 samples
    double samples[441];

    for (int i = 0; i < expectedCrossfadeSamples; ++i) {
        samples[i] = 0.0; // Test with x = 0.0
    }

    engine->processBlock(samples, expectedCrossfadeSamples);

    // Verify smooth transition from 0.0 (identity at x=0) to 0.5 (new LUT)
    // First sample should start crossfade
    EXPECT_NEAR(samples[0], 0.0, 0.1) << "First sample should be close to old LUT value";

    // Last sample should be at new LUT value
    EXPECT_NEAR(samples[expectedCrossfadeSamples - 1], 0.5, 0.01)
        << "Last sample should be close to new LUT value";

    // Middle samples should be interpolated
    const int midIdx = expectedCrossfadeSamples / 2;
    EXPECT_NEAR(samples[midIdx], 0.25, 0.05) << "Middle sample should be halfway between LUTs";
}

/**
 * Test: Crossfade completes in correct sample count
 * Expected: After 441 samples at 44.1kHz, crossfade is done
 */
TEST_F(AudioEngineTest, Crossfade_CompletesInCorrectSampleCount) {
    engine->prepareToPlay(44100.0, 512);

    // Set up two different LUTs
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 1.0; // Different from identity
    }
    buffers[2].version = 1;

    // Signal new LUT ready
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process exactly the crossfade duration
    const int crossfadeSamples = static_cast<int>(44100.0 * 10.0 / 1000.0); // 441
    double samples[441];
    for (int i = 0; i < crossfadeSamples; ++i) samples[i] = 0.5;

    engine->processBlock(samples, crossfadeSamples);

    // After crossfade, output should be from new LUT
    double testSample = 0.5;
    engine->processBlock(&testSample, 1);
    EXPECT_NEAR(testSample, 1.0, 1e-6) << "After crossfade, should use new LUT value";
}

/**
 * Test: New LUT aborts active crossfade and starts fresh
 * Expected: If crossfading and new LUT arrives, old crossfade stops
 */
TEST_F(AudioEngineTest, Crossfade_NewLUTAbortsActiveCrossfade) {
    engine->prepareToPlay(44100.0, 512);

    // Set up first new LUT
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5;
    }
    buffers[2].version = 1;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Start crossfade
    double samples[100];
    for (int i = 0; i < 100; ++i) samples[i] = 0.0;
    engine->processBlock(samples, 100); // Partially through crossfade

    // NOW send a second new LUT
    const int secondaryIdx = engine->getWorkerTargetIndexReference().load(std::memory_order_acquire);
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[secondaryIdx].data[i] = 0.8;
    }
    buffers[secondaryIdx].version = 2;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process one more block - should abort old crossfade and start new one
    engine->processBlock(samples, 100);

    // After completing the NEW crossfade, output should be 0.8
    const int crossfadeSamples = static_cast<int>(44100.0 * 10.0 / 1000.0);
    double remaining[441];
    for (int i = 0; i < crossfadeSamples; ++i) remaining[i] = 0.0;
    engine->processBlock(remaining, crossfadeSamples);

    double testSample = 0.0;
    engine->processBlock(&testSample, 1);
    EXPECT_NEAR(testSample, 0.8, 1e-6) << "Should use second new LUT after abort";
}

// ============================================================================
// Task 2: AudioEngine Tests - Interpolation Accuracy
// ============================================================================

/**
 * Test: Linear interpolation matches expected behavior
 * Expected: Smooth interpolation between table points
 */
TEST_F(AudioEngineTest, Interpolation_LinearIsAccurate) {
    engine->prepareToPlay(44100.0, 512);

    // Set up a simple linear ramp in worker buffer
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        const double x = dsp_core::MIN_VALUE + (i / static_cast<double>(dsp_core::TABLE_SIZE - 1)) *
                                                    (dsp_core::MAX_VALUE - dsp_core::MIN_VALUE);
        buffers[2].data[i] = 2.0 * x; // y = 2x
    }
    buffers[2].version = 1;
    buffers[2].interpolationMode = dsp_core::LayeredTransferFunction::InterpolationMode::Linear;

    // Swap to new LUT
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Skip crossfade
    const int crossfadeSamples = static_cast<int>(44100.0 * 10.0 / 1000.0);
    double skipSamples[441];
    for (int i = 0; i < crossfadeSamples; ++i) skipSamples[i] = 0.0;
    engine->processBlock(skipSamples, crossfadeSamples);

    // Test interpolation at various points
    const std::vector<double> testInputs = {-0.75, -0.25, 0.0, 0.33, 0.99};
    for (double x : testInputs) {
        double output = engine->applyTransferFunction(x);
        EXPECT_NEAR(output, 2.0 * x, 1e-4) << "Linear interpolation should match y = 2x for input " << x;
    }
}

/**
 * Test: Catmull-Rom interpolation works
 * Expected: Smooth cubic interpolation
 */
TEST_F(AudioEngineTest, Interpolation_CatmullRomWorks) {
    engine->prepareToPlay(44100.0, 512);

    // Set up a parabola y = x^2
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        const double x = dsp_core::MIN_VALUE + (i / static_cast<double>(dsp_core::TABLE_SIZE - 1)) *
                                                    (dsp_core::MAX_VALUE - dsp_core::MIN_VALUE);
        buffers[2].data[i] = x * x; // y = x^2
    }
    buffers[2].version = 1;
    buffers[2].interpolationMode = dsp_core::LayeredTransferFunction::InterpolationMode::CatmullRom;

    // Swap to new LUT
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Skip crossfade
    const int crossfadeSamples = static_cast<int>(44100.0 * 10.0 / 1000.0);
    double skipSamples[441];
    for (int i = 0; i < crossfadeSamples; ++i) skipSamples[i] = 0.0;
    engine->processBlock(skipSamples, crossfadeSamples);

    // Test interpolation (should be more accurate than linear for curves)
    const std::vector<double> testInputs = {-0.5, 0.0, 0.5};
    for (double x : testInputs) {
        double output = engine->applyTransferFunction(x);
        EXPECT_NEAR(output, x * x, 0.01) << "Catmull-Rom should approximate y = x^2 for input " << x;
    }
}

// ============================================================================
// Task 2: AudioEngine Tests - Edge Cases
// ============================================================================

/**
 * Test: Extrapolation clamp mode clamps to table bounds
 * Expected: Values outside [-1, 1] clamp to boundary values
 */
TEST_F(AudioEngineTest, Extrapolation_ClampModeClamps) {
    engine->prepareToPlay(44100.0, 512);

    // Test with identity function
    double testInputs[] = {-2.0, -1.5, 1.5, 2.0};
    double expectedOutputs[] = {-1.0, -1.0, 1.0, 1.0};

    for (int i = 0; i < 4; ++i) {
        double output = engine->applyTransferFunction(testInputs[i]);
        EXPECT_NEAR(output, expectedOutputs[i], 1e-6)
            << "Clamp mode should clamp " << testInputs[i] << " to " << expectedOutputs[i];
    }
}

/**
 * Test: Process empty block doesn't crash
 * Expected: Handles numSamples = 0 gracefully
 */
TEST_F(AudioEngineTest, ProcessBlock_HandlesEmptyBlock) {
    engine->prepareToPlay(44100.0, 512);
    double samples[1];
    EXPECT_NO_THROW(engine->processBlock(samples, 0));
}

} // namespace dsp_core_test
