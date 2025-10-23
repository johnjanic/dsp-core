#include "WaveshapingStage.h"

namespace dsp_core::audio_pipeline {

WaveshapingStage::WaveshapingStage(dsp_core::LayeredTransferFunction& ltf, juce::ThreadPool* threadPool)
    : ltf_(ltf), threadPool_(threadPool) {
}

void WaveshapingStage::prepareToPlay(double sampleRate, int samplesPerBlock) {
    // Waveshaping is stateless, no preparation needed
}

void WaveshapingStage::process(juce::AudioBuffer<double>& buffer) {
    if (threadPool_ != nullptr && buffer.getNumChannels() > 1) {
        processParallel(buffer);
    } else {
        processSerial(buffer);
    }
}

void WaveshapingStage::processParallel(juce::AudioBuffer<double>& buffer) {
    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    std::atomic<int> channelsProcessed{0};

    for (int ch = 0; ch < numChannels; ++ch) {
        threadPool_->addJob([&, ch]() {
            double* channelData = buffer.getWritePointer(ch);
            ltf_.processBlock(channelData, numSamples);
            channelsProcessed.fetch_add(1, std::memory_order_release);
        });
    }

    // Wait for all channels (with timeout)
    const int timeout = 100;
    const auto startTime = juce::Time::getMillisecondCounterHiRes();
    while (channelsProcessed.load(std::memory_order_acquire) < numChannels) {
        if (juce::Time::getMillisecondCounterHiRes() - startTime > timeout) {
            jassertfalse;
            break;
        }
        juce::Thread::yield();
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
