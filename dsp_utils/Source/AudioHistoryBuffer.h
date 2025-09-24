#pragma once

#include <JuceHeader.h>

class AudioHistoryBuffer
{
public:
    AudioHistoryBuffer(int historySize);

    void pushSamples(const float* samples, int numSamples);
    void getHistory(float* outSamples, int numSamples) const;
    int getSize() const { return size; }
private:
    std::vector<float> buffer;
    int writePos = 0;
    int size = 0;
};
#pragma once
