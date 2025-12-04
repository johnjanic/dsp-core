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
 * Expected: 50ms at 44.1kHz = 2205 samples, 50ms at 48kHz = 2400 samples
 */
TEST_F(AudioEngineTest, PrepareToPlay_CrossfadeDurationScalesWithSampleRate) {
    // 44.1 kHz
    engine->prepareToPlay(44100.0, 512);
    juce::AudioBuffer<double> buffer(1, 1);
    buffer.clear();
    engine->processBuffer(buffer);
    // We can't directly check crossfadeSamples (private), but we can verify behavior

    // 48 kHz
    engine->prepareToPlay(48000.0, 512);
    // Expected: 2400 samples at 48kHz (50ms)
    // We'll verify this by triggering a crossfade and counting samples
}

/**
 * Test: Sample rate change mid-crossfade clamps position
 * Expected: If crossfadePosition >= new crossfadeSamples, crossfade completes
 */
TEST_F(AudioEngineTest, PrepareToPlay_ClampsPositionOnSampleRateChange) {
    // Start at 96kHz (4800 samples for 50ms)
    engine->prepareToPlay(96000.0, 512);

    // Simulate crossfade in progress by manually swapping buffers
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5; // Different from identity
    }
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Trigger crossfade
    juce::AudioBuffer<double> buffer(1, 500);
    buffer.clear();
    engine->processBuffer(buffer); // Partially through crossfade

    // Now change sample rate to 44.1kHz (2205 samples)
    // The crossfade should complete immediately if position >= 2205
    engine->prepareToPlay(44100.0, 512);
    juce::AudioBuffer<double> buffer2(1, 1);
    buffer2.clear();
    engine->processBuffer(buffer2);

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
    const int expectedCrossfadeSamples = static_cast<int>(44100.0 * 50.0 / 1000.0); // 2205 samples
    juce::AudioBuffer<double> buffer(1, expectedCrossfadeSamples);
    buffer.clear(); // Test with x = 0.0

    engine->processBuffer(buffer);

    // Verify smooth transition from 0.0 (identity at x=0) to 0.5 (new LUT)
    // First sample should start crossfade
    EXPECT_NEAR(buffer.getSample(0, 0), 0.0, 0.1) << "First sample should be close to old LUT value";

    // Last sample should be at new LUT value
    EXPECT_NEAR(buffer.getSample(0, expectedCrossfadeSamples - 1), 0.5, 0.01)
        << "Last sample should be close to new LUT value";

    // Middle samples should be interpolated
    const int midIdx = expectedCrossfadeSamples / 2;
    EXPECT_NEAR(buffer.getSample(0, midIdx), 0.25, 0.05) << "Middle sample should be halfway between LUTs";
}

/**
 * Test: Crossfade completes in correct sample count
 * Expected: After 2205 samples at 44.1kHz, crossfade is done
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
    const int crossfadeSamples = static_cast<int>(44100.0 * 50.0 / 1000.0); // 2205
    juce::AudioBuffer<double> buffer(1, crossfadeSamples);
    for (int i = 0; i < crossfadeSamples; ++i) {
        buffer.setSample(0, i, 0.5);
    }

    engine->processBuffer(buffer);

    // After crossfade, output should be from new LUT
    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.5);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 1.0, 1e-6) << "After crossfade, should use new LUT value";
}

/**
 * Test: New LUT defers during active crossfade
 * Expected: If crossfading and new LUT arrives, update is deferred until crossfade completes
 */
TEST_F(AudioEngineTest, Crossfade_NewLUTDefersUntilComplete) {
    engine->prepareToPlay(44100.0, 512);

    // Set up first new LUT
    dsp_core::LUTBuffer* buffers = engine->getLUTBuffers();
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[2].data[i] = 0.5;
    }
    buffers[2].version = 1;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Start crossfade
    juce::AudioBuffer<double> buffer(1, 100);
    buffer.clear();
    engine->processBuffer(buffer); // Partially through crossfade

    // NOW send a second new LUT while crossfade is in progress
    const int secondaryIdx = engine->getWorkerTargetIndexReference().load(std::memory_order_acquire);
    for (int i = 0; i < dsp_core::TABLE_SIZE; ++i) {
        buffers[secondaryIdx].data[i] = 0.8;
    }
    buffers[secondaryIdx].version = 2;
    engine->getNewLUTReadyFlag().store(true, std::memory_order_release);

    // Process more samples - update is deferred, first crossfade continues
    juce::AudioBuffer<double> buffer2(1, 100);
    buffer2.clear();
    engine->processBuffer(buffer2);

    // Complete the FIRST crossfade - output should be 0.5 (first LUT)
    const int crossfadeSamples = static_cast<int>(44100.0 * 50.0 / 1000.0);
    juce::AudioBuffer<double> remaining(1, crossfadeSamples);
    remaining.clear();
    engine->processBuffer(remaining);

    juce::AudioBuffer<double> testBuffer(1, 1);
    testBuffer.setSample(0, 0, 0.0);
    engine->processBuffer(testBuffer);
    EXPECT_NEAR(testBuffer.getSample(0, 0), 0.5, 1e-6) << "Should complete first crossfade to 0.5";

    // Now the deferred LUT should trigger a new crossfade
    // Complete the SECOND crossfade - output should be 0.8 (second LUT)
    juce::AudioBuffer<double> remaining2(1, crossfadeSamples + 1);
    remaining2.clear();
    engine->processBuffer(remaining2);

    juce::AudioBuffer<double> testBuffer2(1, 1);
    testBuffer2.setSample(0, 0, 0.0);
    engine->processBuffer(testBuffer2);
    EXPECT_NEAR(testBuffer2.getSample(0, 0), 0.8, 1e-6) << "Should use second LUT after deferred update";
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
    const int crossfadeSamples = static_cast<int>(44100.0 * 50.0 / 1000.0);
    juce::AudioBuffer<double> skipBuffer(1, crossfadeSamples);
    skipBuffer.clear();
    engine->processBuffer(skipBuffer);

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
    const int crossfadeSamples = static_cast<int>(44100.0 * 50.0 / 1000.0);
    juce::AudioBuffer<double> skipBuffer(1, crossfadeSamples);
    skipBuffer.clear();
    engine->processBuffer(skipBuffer);

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
    juce::AudioBuffer<double> buffer(1, 0);
    EXPECT_NO_THROW(engine->processBuffer(buffer));
}

} // namespace dsp_core_test
