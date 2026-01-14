#pragma once

#include "Decibels.h"
#include "SmoothedValue.h"
#include <platform/AudioBuffer.h>
#include <algorithm>

namespace dsp {

/**
 * @brief Gain processor with smoothed parameter changes.
 *
 * Applies gain to audio buffers with click-free parameter transitions.
 * Uses linear interpolation for smooth gain changes.
 *
 * @tparam T The sample type (typically float or double)
 *
 * Usage:
 * @code
 * dsp::Gain<double> gain;
 * gain.prepare(sampleRate);
 * gain.setGainDecibels(-6.0);  // Smoothly ramp to -6dB
 *
 * // In processBlock:
 * gain.processBlock(samples, numSamples);
 * @endcode
 */
template<typename T>
class Gain
{
public:
    /** Default constructor with unity gain. */
    Gain() = default;

    /**
     * @brief Prepare the gain processor for playback.
     *
     * @param sampleRate The sample rate in Hz.
     * @param rampTimeSeconds Time for gain changes to ramp (default 0.01s).
     */
    void prepare(double sampleRate, double rampTimeSeconds = 0.01) noexcept
    {
        smoothedGain.reset(sampleRate, rampTimeSeconds);
        smoothedGain.setCurrentAndTargetValue(targetGain);
    }

    /**
     * @brief Reset the gain processor state.
     *
     * Clears smoothing state. Call before processing after a pause.
     */
    void reset() noexcept
    {
        smoothedGain.setCurrentAndTargetValue(targetGain);
    }

    /**
     * @brief Set gain as a linear multiplier.
     *
     * @param newGain Linear gain value (1.0 = unity).
     */
    void setGainLinear(T newGain) noexcept
    {
        targetGain = newGain;
        smoothedGain.setTargetValue(newGain);
    }

    /**
     * @brief Set gain in decibels.
     *
     * @param newGainDecibels Gain in dB (0.0 = unity).
     */
    void setGainDecibels(T newGainDecibels) noexcept
    {
        setGainLinear(Decibels::decibelsToGain(newGainDecibels));
    }

    /**
     * @brief Get the current target gain (linear).
     */
    [[nodiscard]] T getGainLinear() const noexcept
    {
        return targetGain;
    }

    /**
     * @brief Get the current target gain in decibels.
     */
    [[nodiscard]] T getGainDecibels() const noexcept
    {
        return Decibels::gainToDecibels(targetGain);
    }

    /**
     * @brief Check if gain is currently ramping.
     */
    [[nodiscard]] bool isSmoothing() const noexcept
    {
        return smoothedGain.isSmoothing();
    }

    /**
     * @brief Process a single sample.
     *
     * @param input Input sample.
     * @return Gained output sample.
     */
    [[nodiscard]] T processSample(T input) noexcept
    {
        return input * smoothedGain.getNextValue();
    }

    /**
     * @brief Process AudioBuffer in-place.
     *
     * @param buffer AudioBuffer to process.
     */
    void processBlock(platform::AudioBuffer<T>& buffer) noexcept
    {
        processBlockImpl(buffer.getArrayOfWritePointers(),
                         buffer.getNumChannels(),
                         buffer.getNumSamples());
    }

    /**
     * @brief Process with separate input/output AudioBuffers.
     *
     * @param input Input AudioBuffer.
     * @param output Output AudioBuffer.
     */
    void processBlock(const platform::AudioBuffer<T>& input,
                      platform::AudioBuffer<T>& output) noexcept
    {
        processBlockImpl(input.getArrayOfReadPointers(),
                         output.getArrayOfWritePointers(),
                         std::min(input.getNumChannels(), output.getNumChannels()),
                         std::min(input.getNumSamples(), output.getNumSamples()));
    }

private:
    // Internal implementation using raw pointers for efficiency
    void processBlockImpl(T** channelData, int numChannels, int numSamples) noexcept
    {
        if (smoothedGain.isSmoothing())
        {
            for (int sample = 0; sample < numSamples; ++sample)
            {
                const T gain = smoothedGain.getNextValue();
                for (int channel = 0; channel < numChannels; ++channel)
                {
                    channelData[channel][sample] *= gain;
                }
            }
        }
        else
        {
            const T gain = smoothedGain.getTargetValue();
            if (gain != T(1))
            {
                for (int channel = 0; channel < numChannels; ++channel)
                {
                    for (int sample = 0; sample < numSamples; ++sample)
                    {
                        channelData[channel][sample] *= gain;
                    }
                }
            }
        }
    }

    void processBlockImpl(const T* const* inputData, T** outputData,
                          int numChannels, int numSamples) noexcept
    {
        if (smoothedGain.isSmoothing())
        {
            for (int sample = 0; sample < numSamples; ++sample)
            {
                const T gain = smoothedGain.getNextValue();
                for (int channel = 0; channel < numChannels; ++channel)
                {
                    outputData[channel][sample] = inputData[channel][sample] * gain;
                }
            }
        }
        else
        {
            const T gain = smoothedGain.getTargetValue();
            for (int channel = 0; channel < numChannels; ++channel)
            {
                for (int sample = 0; sample < numSamples; ++sample)
                {
                    outputData[channel][sample] = inputData[channel][sample] * gain;
                }
            }
        }
    }

    SmoothedValue<T> smoothedGain{T(1)};
    T targetGain = T(1);
};

} // namespace dsp
