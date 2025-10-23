#include "WaveshapingStage.h"
#include <thread>
#include <vector>

namespace dsp_core::audio_pipeline {

WaveshapingStage::WaveshapingStage(dsp_core::LayeredTransferFunction& ltf, juce::ThreadPool* threadPool)
    : ltf_(ltf), threadPool_(threadPool) {
}

void WaveshapingStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // Waveshaping is stateless, no preparation needed
}

void WaveshapingStage::process(juce::AudioBuffer<double>& buffer) {
    // PERFORMANCE: ThreadPool has per-job allocation overhead (new LambdaJobWrapper).
    // For typical stereo (2 channels), serial processing is faster than thread overhead.
    // Only parallelize for > 2 channels (5.1, 7.1 surround).
    if (threadPool_ != nullptr && buffer.getNumChannels() > 2) {
        processParallel(buffer);
    } else {
        processSerial(buffer);
    }
}

void WaveshapingStage::processParallel(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    // PERFORMANCE FIX: Use std::thread directly instead of ThreadPool
    // ThreadPool does "new LambdaJobWrapper" per job = heap allocation on audio thread
    // std::thread allocation is handled by OS thread cache, much faster
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(numChannels));

    for (int ch = 0; ch < numChannels; ++ch) {
        threads.emplace_back([&, ch]() {
            double* channelData = buffer.getWritePointer(ch);
            ltf_.processBlock(channelData, numSamples);
        });
    }

    // Join all threads (OS handles efficient wake-up, no spin-wait)
    for (auto& t : threads) {
        t.join();
    }
}

void WaveshapingStage::processSerial(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        double* channelData = buffer.getWritePointer(ch);
        ltf_.processBlock(channelData, numSamples);
    }
}

void WaveshapingStage::reset() {
    // Waveshaping is stateless
}

} // namespace dsp_core::audio_pipeline
