#pragma once

#include <JuceHeader.h>

class AudioHistoryBuffer
{
public:
    AudioHistoryBuffer(int historySize);

    void pushSamples(const double* samples, int numSamples);
    void getHistory(double* outSamples, int numSamples) const;
    int getSize() const { return size; }
    void clear() { std::fill(buffer.begin(), buffer.end(), 0.0); writePos = 0; }

private:
    std::vector<double> buffer;
    int writePos = 0;
    int size = 0;
};
#pragma once
