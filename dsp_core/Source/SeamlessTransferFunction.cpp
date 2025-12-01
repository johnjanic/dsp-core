#include "SeamlessTransferFunction.h"
#include "SeamlessTransferFunctionImpl.h"
#include <juce_core/juce_core.h>

namespace dsp_core {

//==============================================================================
// Pimpl - Private implementation
//==============================================================================

/**
 * SeamlessTransferFunction::Impl - Private implementation with all components
 *
 * Components:
 *   - editingModel: LayeredTransferFunction (message thread only)
 *   - audioEngine: AudioEngine (audio thread)
 *   - renderer: LUTRendererThread (worker thread, created in startSeamlessUpdates)
 *   - poller: TransferFunctionDirtyPoller (message thread, created in startSeamlessUpdates)
 *   - visualizerLUT: UI-owned LUT buffer (message thread only)
 *   - visualizerCallback: Callback for visualizer repaint (message thread)
 *
 * Lifecycle:
 *   1. Constructor: Creates editingModel and audioEngine (both initialized to identity)
 *   2. startSeamlessUpdates(): Creates renderer and poller, triggers initial render
 *   3. prepareToPlay(): Configures audioEngine crossfade duration
 *   4. processBlock(): Audio thread processes samples with crossfade
 *   5. stopSeamlessUpdates(): Stops poller and renderer
 *   6. Destructor: Ensures synchronous cleanup
 */
class SeamlessTransferFunction::Impl {
  public:
    Impl()
        : editingModel(TABLE_SIZE, MIN_VALUE, MAX_VALUE) {
        // editingModel initialized to identity
        // audioEngine initialized to identity LUTs (in AudioEngine constructor)
        // renderer and poller are null (created in startSeamlessUpdates)

        // Initialize visualizer LUT to identity (same as editingModel)
        for (int i = 0; i < TABLE_SIZE; ++i) {
            const double x = MIN_VALUE + (i / static_cast<double>(TABLE_SIZE - 1)) * (MAX_VALUE - MIN_VALUE);
            visualizerLUT[i] = x;
        }
    }

    // Editing model (message thread only)
    LayeredTransferFunction editingModel;

    // Audio engine (audio thread)
    AudioEngine audioEngine;

    // Worker thread and poller (created in startSeamlessUpdates, null until then)
    std::unique_ptr<LUTRendererThread> renderer;
    std::unique_ptr<TransferFunctionDirtyPoller> poller;

    // Visualizer state (message thread only)
    // NOTE: Visualizer shows the latest rendered LUT (target curve that audio
    // is crossfading toward or has reached). This may be ahead of what's
    // currently playing if a crossfade is still in progress.
    std::array<double, TABLE_SIZE> visualizerLUT;
    std::function<void()> visualizerCallback;
};

//==============================================================================
// SeamlessTransferFunction Implementation
//==============================================================================

SeamlessTransferFunction::SeamlessTransferFunction()
    : pimpl(std::make_unique<Impl>()) {
    // editingModel already initialized in Impl constructor
    // audioEngine already initialized (identity LUTs)
    // Worker thread and poller NOT created yet (deferred to startSeamlessUpdates)
}

SeamlessTransferFunction::~SeamlessTransferFunction() {
    // CRITICAL: Stop async operations before destruction
    // releaseResources() is async, so we need synchronous stop here

    // Stop poller first (no more jobs enqueued)
    if (pimpl->poller) {
        pimpl->poller->stopTimer();
    }

    // Stop worker thread (finish current job, then exit)
    if (pimpl->renderer) {
        pimpl->renderer->stopThread(2000); // 2 second timeout
    }

    // pimpl destruction handles the rest
}

LayeredTransferFunction& SeamlessTransferFunction::getEditingModel() {
    return pimpl->editingModel;
}

const LayeredTransferFunction& SeamlessTransferFunction::getEditingModel() const {
    return pimpl->editingModel;
}

double SeamlessTransferFunction::applyTransferFunction(double x) const {
    return pimpl->audioEngine.applyTransferFunction(x);
}

void SeamlessTransferFunction::processBuffer(juce::AudioBuffer<double>& buffer) const {
    // Audio thread: process entire multi-channel buffer with shared crossfade state
    // Change detection happens on the Editor's timer (25Hz) via notifyEditingModelChanged()
    pimpl->audioEngine.processBuffer(buffer);
}

void SeamlessTransferFunction::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // Can be called from audio thread - no message thread check
    pimpl->audioEngine.prepareToPlay(sampleRate, samplesPerBlock);
}

void SeamlessTransferFunction::releaseResources() {
    // CRITICAL FIX: Do NOT stop seamless updates here!
    //
    // WHY: DAWs call releaseResources() unpredictably (e.g., when stopping playback,
    // or even during initialization). The seamless update system should stay alive
    // for the plugin's entire lifetime because:
    //
    // 1. Poller runs on message thread (UI-related, not an audio resource)
    // 2. Worker thread is idle when not rendering (no CPU overhead)
    // 3. Audio engine is always needed (not just during playback)
    //
    // The only time we should stop seamless updates is in the destructor.
    //
    // PREVIOUS BUG: Calling stopSeamlessUpdates() here destroyed the poller,
    // breaking UI updates after DAW called releaseResources().
}

void SeamlessTransferFunction::startSeamlessUpdates() {
    // VERIFY: Called on message thread
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // VERIFY: Not already started
    jassert(pimpl->renderer == nullptr && pimpl->poller == nullptr);

    // Create worker thread with visualizer integration
    auto visualizerCallback = [this]() {
        if (pimpl->visualizerCallback) {
            pimpl->visualizerCallback();
        }
    };

    pimpl->renderer = std::make_unique<LUTRendererThread>(
        pimpl->audioEngine.getWorkerTargetIndexReference(),
        pimpl->audioEngine.getNewLUTReadyFlag(),
        visualizerCallback);

    // Pass visualizer LUT buffer pointer to worker thread
    pimpl->renderer->setVisualizerLUTPointer(&pimpl->visualizerLUT);

    // Pass LUT buffers pointer to worker thread (for direct writes)
    pimpl->renderer->setLUTBuffersPointer(pimpl->audioEngine.getLUTBuffers());

    // Start worker thread
    pimpl->renderer->startThread(juce::Thread::Priority::normal);

    // Create poller (NO TIMER - Editor will push changes via notifyEditingModelChanged)
    pimpl->poller = std::make_unique<TransferFunctionDirtyPoller>(pimpl->editingModel, *pimpl->renderer);
    // NOTE: Do NOT start poller's timer! Editor's timer handles change detection now.

    // Trigger initial render (ensures visualizer shows correct curve immediately)
    pimpl->poller->forceRender();
}

void SeamlessTransferFunction::stopSeamlessUpdates() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (pimpl->poller) {
        pimpl->poller->stopTimer();
        pimpl->poller.reset();
    }

    if (pimpl->renderer) {
        pimpl->renderer->stopThread(1000); // 1 second timeout
        pimpl->renderer.reset();
    }
}

void SeamlessTransferFunction::notifyEditingModelChanged() {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());

    // Directly trigger the poller's timer callback to capture and enqueue a render job
    // This is called from the Editor's timer (25Hz) when it detects version changes
    if (pimpl && pimpl->poller) {
        pimpl->poller->timerCallback();
    }
}

const std::array<double, SeamlessTransferFunction::TABLE_SIZE>&
SeamlessTransferFunction::getVisualizerLUT() const {
    return pimpl->visualizerLUT;
}

void SeamlessTransferFunction::setVisualizerCallback(std::function<void()> callback) {
    jassert(juce::MessageManager::getInstance()->isThisTheMessageThread());
    pimpl->visualizerCallback = std::move(callback);
}

} // namespace dsp_core
