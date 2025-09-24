#include "AudioHistoryBuffer.h"

AudioHistoryBuffer::AudioHistoryBuffer(int historySize)
    : buffer(historySize, 0.0f), writePos(0), size(historySize)
{
}

void AudioHistoryBuffer::pushSamples(const float* samples, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        buffer[writePos] = samples[i];
        writePos = (writePos + 1) % size;
    }
}

void AudioHistoryBuffer::getHistory(float* outSamples, int numSamples) const
{
    jassert(numSamples <= size);

    int start = (writePos - numSamples + size) % size;
    int firstPart = juce::jmin(size - start, numSamples);
    int secondPart = numSamples - firstPart;

    std::memcpy(outSamples, buffer.data() + start, firstPart * sizeof(float));
    if (secondPart > 0)
        std::memcpy(outSamples + firstPart, buffer.data(), secondPart * sizeof(float));
}
