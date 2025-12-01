#include <dsp_core/dsp_core.h>
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <iostream>

namespace dsp_core_test {

/**
 * DIAGNOSTIC TEST: Verify SeamlessTransferFunction audio path
 *
 * This test traces the full data flow from editing model to audio output
 * to diagnose why audio remains identity after painting/preset loading.
 */
class SeamlessTransferFunctionDiagnosticTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // CRITICAL: Initialize JUCE message manager for timer support
        // Without this, juce::Timer callbacks (used by poller) won't fire!
        juce::MessageManager::getInstance();

        stf = std::make_unique<dsp_core::SeamlessTransferFunction>();
        stf->prepareToPlay(44100.0, 512);
    }

    void TearDown() override {
        stf->releaseResources();
        stf.reset();
    }

    // Helper: Pump message queue to allow timer callbacks to fire
    void pumpMessageQueue(int milliseconds = 50) {
        // Simple sleep - JUCE timers fire on message thread
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    }

    std::unique_ptr<dsp_core::SeamlessTransferFunction> stf;
};

/**
 * TEST: Verify initial state is identity (y = x)
 */
TEST_F(SeamlessTransferFunctionDiagnosticTest, InitialState_IsIdentity) {
    std::cout << "\n[DIAGNOSTIC] Testing initial state..." << std::endl;

    // Test audio output (should be identity before startSeamlessUpdates)
    double testSamples[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    stf->processBlock(testSamples, 5);

    std::cout << "[DIAGNOSTIC] Initial audio output:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  x = " << (-1.0 + i * 0.5) << " → y = " << testSamples[i] << std::endl;
        EXPECT_NEAR(testSamples[i], -1.0 + i * 0.5, 1e-6) << "Initial state should be identity";
    }
}

/**
 * TEST: Verify editing model can be modified
 */
TEST_F(SeamlessTransferFunctionDiagnosticTest, EditingModel_CanBeModified) {
    std::cout << "\n[DIAGNOSTIC] Testing editing model modification..." << std::endl;

    auto& editingModel = stf->getEditingModel();

    // Set base layer to y = 0.5 (constant)
    std::cout << "[DIAGNOSTIC] Setting base layer to y = 0.5..." << std::endl;
    for (int i = 0; i < 16384; ++i) {
        editingModel.setBaseLayerValue(i, 0.5);
    }

    // Verify editing model was updated
    EXPECT_NEAR(editingModel.getBaseLayerValue(0), 0.5, 1e-6);
    EXPECT_NEAR(editingModel.getBaseLayerValue(8192), 0.5, 1e-6);
    std::cout << "[DIAGNOSTIC] Editing model updated successfully" << std::endl;

    // Trigger composite update
    editingModel.updateComposite();
    std::cout << "[DIAGNOSTIC] Composite updated" << std::endl;

    // Check version counter
    uint64_t version = editingModel.getVersion();
    std::cout << "[DIAGNOSTIC] Version counter: " << version << std::endl;
    EXPECT_GT(version, 0u) << "Version should have incremented";
}

/**
 * TEST: Verify worker thread renders LUT after edit
 */
TEST_F(SeamlessTransferFunctionDiagnosticTest, WorkerThread_RendersAfterEdit) {
    std::cout << "\n[DIAGNOSTIC] Testing worker thread rendering..." << std::endl;

    // Start seamless updates (creates worker thread and poller)
    std::cout << "[DIAGNOSTIC] Starting seamless updates..." << std::endl;
    stf->startSeamlessUpdates();

    // Wait for initial render (triggered by startSeamlessUpdates)
    std::cout << "[DIAGNOSTIC] Waiting for initial render (100ms)..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Modify editing model
    auto& editingModel = stf->getEditingModel();
    std::cout << "[DIAGNOSTIC] Setting coefficient[1] to 0.8..." << std::endl;
    editingModel.setCoefficient(1, 0.8);

    uint64_t version = editingModel.getVersion();
    std::cout << "[DIAGNOSTIC] Version after edit: " << version << std::endl;

    // Wait for worker thread to process (poller at 25Hz = 40ms interval)
    std::cout << "[DIAGNOSTIC] Waiting for worker to render (50ms)..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Check visualizer LUT (should be updated)
    const auto& vizLUT = stf->getVisualizerLUT();
    std::cout << "[DIAGNOSTIC] Visualizer LUT samples:" << std::endl;
    std::cout << "  viz[0] = " << vizLUT[0] << " (expected: -1.0)" << std::endl;
    std::cout << "  viz[8192] = " << vizLUT[8192] << " (expected: ~0.0)" << std::endl;
    std::cout << "  viz[16383] = " << vizLUT[16383] << " (expected: ~1.0)" << std::endl;
}

/**
 * TEST: CRITICAL - Verify audio output changes after edit
 */
TEST_F(SeamlessTransferFunctionDiagnosticTest, CRITICAL_AudioOutput_ChangesAfterEdit) {
    std::cout << "\n[DIAGNOSTIC] ===== CRITICAL TEST: Audio Output After Edit =====" << std::endl;

    // Start seamless updates
    std::cout << "[DIAGNOSTIC] Starting seamless updates..." << std::endl;
    stf->startSeamlessUpdates();
    pumpMessageQueue(100); // Allow initial render to complete

    // Modify editing model to y = 0.5 (constant)
    auto& editingModel = stf->getEditingModel();
    std::cout << "[DIAGNOSTIC] Setting base layer to y = 0.5 (constant)..." << std::endl;
    for (int i = 0; i < 16384; ++i) {
        editingModel.setBaseLayerValue(i, 0.5);
    }
    editingModel.setCoefficient(0, 1.0); // WT mix = 100%
    editingModel.updateComposite();

    uint64_t version = editingModel.getVersion();
    std::cout << "[DIAGNOSTIC] Version after edit: " << version << std::endl;

    // CRITICAL: Pump message queue to allow poller timer to fire!
    std::cout << "[DIAGNOSTIC] Pumping message queue for worker render (100ms)..." << std::endl;
    pumpMessageQueue(100); // This allows the 25Hz timer to fire and detect version change

    // Process audio to trigger LUT swap
    double testSamples[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    std::cout << "[DIAGNOSTIC] Processing audio block (triggers checkForNewLUT)..." << std::endl;
    stf->processBlock(testSamples, 5);

    // Wait for crossfade to complete (10ms at 44.1kHz = 441 samples)
    std::cout << "[DIAGNOSTIC] Processing 500 more samples to complete crossfade..." << std::endl;
    double crossfadeSamples[500];
    for (int i = 0; i < 500; ++i) crossfadeSamples[i] = 0.0;
    stf->processBlock(crossfadeSamples, 500);

    // NOW check audio output
    double finalSamples[5] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    stf->processBlock(finalSamples, 5);

    std::cout << "\n[DIAGNOSTIC] ===== AUDIO OUTPUT AFTER EDIT =====" << std::endl;
    for (int i = 0; i < 5; ++i) {
        double input = -1.0 + i * 0.5;
        double output = finalSamples[i];
        std::cout << "  x = " << input << " → y = " << output;
        if (std::abs(output - 0.5) < 0.1) {
            std::cout << " [CORRECT: y=0.5]" << std::endl;
        } else if (std::abs(output - input) < 0.1) {
            std::cout << " [ERROR: Still identity!]" << std::endl;
        } else {
            std::cout << " [UNKNOWN]" << std::endl;
        }
    }

    std::cout << "\n[DIAGNOSTIC] Expected: All outputs should be ~0.5" << std::endl;
    std::cout << "[DIAGNOSTIC] If outputs are still identity, worker/audio path is broken!" << std::endl;

    // Assertions
    for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(finalSamples[i], 0.5, 0.1) << "Audio output should be y=0.5, not identity!";
    }
}

/**
 * TEST: Verify editing model composite matches expected value
 */
TEST_F(SeamlessTransferFunctionDiagnosticTest, EditingModel_CompositeIsCorrect) {
    std::cout << "\n[DIAGNOSTIC] Testing editing model composite..." << std::endl;

    auto& editingModel = stf->getEditingModel();

    // Set base layer to y = 0.5
    for (int i = 0; i < 16384; ++i) {
        editingModel.setBaseLayerValue(i, 0.5);
    }
    editingModel.setCoefficient(0, 1.0); // WT mix = 100%
    editingModel.updateComposite();

    // Check composite values
    std::cout << "[DIAGNOSTIC] Editing model composite values:" << std::endl;
    std::cout << "  composite[0] = " << editingModel.getCompositeValue(0) << " (expected: 0.5)" << std::endl;
    std::cout << "  composite[8192] = " << editingModel.getCompositeValue(8192) << " (expected: 0.5)" << std::endl;
    std::cout << "  composite[16383] = " << editingModel.getCompositeValue(16383) << " (expected: 0.5)" << std::endl;

    EXPECT_NEAR(editingModel.getCompositeValue(0), 0.5, 0.1);
    EXPECT_NEAR(editingModel.getCompositeValue(8192), 0.5, 0.1);
    EXPECT_NEAR(editingModel.getCompositeValue(16383), 0.5, 0.1);
}

} // namespace dsp_core_test
