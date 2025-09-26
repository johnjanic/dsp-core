#pragma once

#include <JuceHeader.h>

class AudioHistoryBuffer
{
public:
    AudioHistoryBuffer(int historySize);

    void pushSamples(const float* samples, int numSamples);
    void getHistory(float* outSamples, int numSamples) const;
    int getSize() const { return size; }
    void clear() { std::fill(buffer.begin(), buffer.end(), 0.0f); writePos = 0; }

private:
    std::vector<float> buffer;
    int writePos = 0;
    int size = 0;
};
#pragma once
