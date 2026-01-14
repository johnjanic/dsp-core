#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <memory>
#include <algorithm>

namespace dsp {

/**
 * @brief Half-band IIR filter coefficients for oversampling.
 *
 * These coefficients are optimized for audio oversampling with:
 * - High stopband attenuation (>70dB)
 * - Minimal passband ripple (<0.1dB)
 * - Low group delay variation
 */
namespace HalfBandCoefficients {
    // 8th order elliptic half-band IIR coefficients
    // Designed for 0.1dB passband ripple, 70dB stopband attenuation
    // Format: {a1, a2, b0, b1, b2} for each biquad section

    // First allpass branch coefficients (odd samples)
    constexpr std::array<double, 4> allpass1Coeffs = {
        0.07986642623635751,
        0.5453536510711322,
        0.28382934487410993,
        0.8344118914807379
    };

    // Second allpass branch coefficients (even samples)
    constexpr std::array<double, 4> allpass2Coeffs = {
        0.0,
        0.1858185225636498,
        0.6572984716640783,
        0.9320575784592673
    };
}

/**
 * @brief Single stage 2x oversampling using half-band IIR filter.
 *
 * Implements polyphase half-band IIR filtering for efficient
 * 2x upsampling and downsampling.
 *
 * @tparam T Sample type (float or double)
 */
template<typename T>
class OversamplingStage
{
public:
    OversamplingStage()
    {
        reset();
    }

    void reset()
    {
        // Clear allpass filter states
        for (auto& state : allpass1State) state = T(0);
        for (auto& state : allpass2State) state = T(0);
        lastInput = T(0);
    }

    /**
     * @brief Upsample by factor of 2.
     *
     * @param input Input sample.
     * @param output Output array (must hold 2 samples).
     */
    void upsample(T input, T* output)
    {
        // Polyphase upsampling:
        // Process through both allpass branches
        T ap1 = processAllpass1(input);
        T ap2 = processAllpass2(lastInput);
        lastInput = input;

        // Interleave outputs - no attenuation on upsample
        output[0] = ap1 + ap2;
        output[1] = ap1 - ap2;
    }

    /**
     * @brief Downsample by factor of 2.
     *
     * @param input Input array (must hold 2 samples).
     * @return Downsampled output sample.
     */
    T downsample(const T* input)
    {
        // Polyphase downsampling:
        // Process even and odd samples through respective branches
        T ap1 = processAllpass1(input[0]);
        T ap2 = processAllpass2(input[1]);

        // Scale by 0.5 for unity gain through round-trip
        return (ap1 + ap2) * T(0.5);
    }

private:
    // Process through first allpass branch
    T processAllpass1(T input)
    {
        T output = input;
        for (size_t i = 0; i < HalfBandCoefficients::allpass1Coeffs.size(); ++i)
        {
            T coeff = static_cast<T>(HalfBandCoefficients::allpass1Coeffs[i]);
            T temp = output + coeff * allpass1State[i];
            output = allpass1State[i] - coeff * temp;
            allpass1State[i] = temp;
        }
        return output;
    }

    // Process through second allpass branch
    T processAllpass2(T input)
    {
        T output = input;
        for (size_t i = 0; i < HalfBandCoefficients::allpass2Coeffs.size(); ++i)
        {
            T coeff = static_cast<T>(HalfBandCoefficients::allpass2Coeffs[i]);
            T temp = output + coeff * allpass2State[i];
            output = allpass2State[i] - coeff * temp;
            allpass2State[i] = temp;
        }
        return output;
    }

    std::array<T, 4> allpass1State{};
    std::array<T, 4> allpass2State{};
    T lastInput{};
};


/**
 * @brief Multi-stage oversampling processor.
 *
 * Provides up to 16x oversampling (order 0-4) using cascaded
 * half-band IIR filter stages. Each order doubles the sample rate.
 *
 * Order 0: 1x (bypass)
 * Order 1: 2x oversampling
 * Order 2: 4x oversampling
 * Order 3: 8x oversampling
 * Order 4: 16x oversampling
 *
 * @tparam T Sample type (float or double)
 *
 * Usage:
 * @code
 * dsp::Oversampling<float> oversampler(2, 2);  // 2 channels, 4x oversampling
 * oversampler.prepare(maxBlockSize);
 *
 * // In processBlock:
 * T** upsampled = oversampler.processSamplesUp(inputPtrs, numInputSamples);
 * int oversampledSize = numInputSamples * oversampler.getOversamplingFactor();
 * // Process at higher sample rate...
 * oversampler.processSamplesDown(outputPtrs, numInputSamples);
 * @endcode
 */
template<typename T>
class Oversampling
{
public:
    /**
     * @brief Construct oversampling processor.
     *
     * @param numChannels Number of audio channels.
     * @param order Oversampling order (0-4). Factor = 2^order.
     */
    Oversampling(int numChannels, int order)
        : numChannels_(numChannels)
        , order_(std::clamp(order, 0, 4))
        , factor_(1 << order_)
    {
        // Create stages for each channel
        stages_.resize(numChannels_);
        for (auto& channelStages : stages_)
        {
            channelStages.resize(order_);
        }
    }

    /**
     * @brief Prepare for processing.
     *
     * @param maxBlockSize Maximum input block size to handle.
     */
    void prepare(int maxBlockSize)
    {
        maxBlockSize_ = maxBlockSize;

        // Allocate work buffers for maximum oversampled size
        int maxOversampledSize = maxBlockSize * factor_;

        // Allocate internal buffer storage
        oversampledData_.resize(numChannels_);
        oversampledPtrs_.resize(numChannels_);
        for (int ch = 0; ch < numChannels_; ++ch)
        {
            oversampledData_[ch].resize(maxOversampledSize, T(0));
            oversampledPtrs_[ch] = oversampledData_[ch].data();
        }

        // Intermediate buffer for cascaded stages
        if (order_ > 1)
        {
            tempBuffer_.resize(maxOversampledSize);
        }

        reset();
    }

    /**
     * @brief Reset all filter states.
     */
    void reset()
    {
        for (auto& channelStages : stages_)
        {
            for (auto& stage : channelStages)
            {
                stage.reset();
            }
        }

        // Clear oversampled buffers
        for (auto& channelData : oversampledData_)
        {
            std::fill(channelData.begin(), channelData.end(), T(0));
        }
    }

    /**
     * @brief Get the oversampling factor.
     */
    [[nodiscard]] int getOversamplingFactor() const noexcept
    {
        return factor_;
    }

    /**
     * @brief Get the oversampling order.
     */
    [[nodiscard]] int getOrder() const noexcept
    {
        return order_;
    }

    /**
     * @brief Get number of channels.
     */
    [[nodiscard]] int getNumChannels() const noexcept
    {
        return numChannels_;
    }

    /**
     * @brief Get latency in input samples.
     *
     * Each stage adds latency due to the IIR filter group delay.
     */
    [[nodiscard]] int getLatencyInSamples() const noexcept
    {
        // Half-band IIR filter latency per stage (approximate)
        // The actual value depends on the filter design
        // JUCE uses ~4-8 samples per stage
        return order_ * 4;
    }

    /**
     * @brief Upsample multi-channel audio.
     *
     * @param inputChannels Array of pointers to input channel data.
     * @param numInputSamples Number of samples in each input channel.
     * @return Array of pointers to oversampled channel data (size = numInputSamples * factor).
     */
    T** processSamplesUp(const T* const* inputChannels, int numInputSamples)
    {
        const int outputSamples = numInputSamples * factor_;
        currentOversampledSize_ = outputSamples;

        if (order_ == 0)
        {
            // Bypass mode - just copy
            for (int ch = 0; ch < numChannels_; ++ch)
            {
                std::copy(inputChannels[ch], inputChannels[ch] + numInputSamples,
                          oversampledData_[ch].data());
            }
            return oversampledPtrs_.data();
        }

        for (int channel = 0; channel < numChannels_; ++channel)
        {
            const T* input = inputChannels[channel];
            T* output = oversampledData_[channel].data();

            // First stage: 1x -> 2x
            int currentSize = numInputSamples;
            const T* currentInput = input;
            T* currentOutput = (order_ == 1) ? output : tempBuffer_.data();

            for (int i = 0; i < currentSize; ++i)
            {
                stages_[channel][0].upsample(currentInput[i], &currentOutput[i * 2]);
            }
            currentSize *= 2;

            // Subsequent stages: 2x -> 4x -> 8x -> 16x
            for (int stage = 1; stage < order_; ++stage)
            {
                currentInput = currentOutput;
                currentOutput = (stage == order_ - 1) ? output : tempBuffer_.data();

                // Need to swap buffers for in-place processing
                if (currentInput == currentOutput)
                {
                    // Copy to temp, then process
                    std::copy(currentInput, currentInput + currentSize, tempBuffer_.data());
                    currentInput = tempBuffer_.data();
                }

                for (int i = 0; i < currentSize; ++i)
                {
                    stages_[channel][stage].upsample(currentInput[i], &currentOutput[i * 2]);
                }
                currentSize *= 2;
            }
        }

        return oversampledPtrs_.data();
    }

    /**
     * @brief Downsample multi-channel audio.
     *
     * @param outputChannels Array of pointers to output channel data.
     * @param numOutputSamples Number of samples expected in each output channel.
     */
    void processSamplesDown(T** outputChannels, int numOutputSamples)
    {
        if (order_ == 0)
        {
            // Bypass mode - just copy
            for (int ch = 0; ch < numChannels_; ++ch)
            {
                std::copy(oversampledData_[ch].data(),
                          oversampledData_[ch].data() + numOutputSamples,
                          outputChannels[ch]);
            }
            return;
        }

        int currentSize = numOutputSamples * factor_;

        for (int channel = 0; channel < numChannels_; ++channel)
        {
            T* output = outputChannels[channel];
            const T* currentInput = oversampledData_[channel].data();
            T* currentOutput = tempBuffer_.data();

            // Process stages in reverse order
            for (int stage = order_ - 1; stage >= 0; --stage)
            {
                currentOutput = (stage == 0) ? output : tempBuffer_.data();
                int outputSize = currentSize / 2;

                for (int i = 0; i < outputSize; ++i)
                {
                    currentOutput[i] = stages_[channel][stage].downsample(&currentInput[i * 2]);
                }

                currentInput = currentOutput;
                currentSize = outputSize;
            }
        }
    }

    /**
     * @brief Get array of pointers to internal oversampled buffers.
     *
     * Use after processSamplesUp() to access oversampled data for processing.
     */
    [[nodiscard]] T** getOversampledBuffers() noexcept
    {
        return oversampledPtrs_.data();
    }

    [[nodiscard]] T* const* getOversampledBuffers() const noexcept
    {
        return oversampledPtrs_.data();
    }

    /**
     * @brief Get the current oversampled buffer size (after last processSamplesUp call).
     */
    [[nodiscard]] int getOversampledSize() const noexcept
    {
        return currentOversampledSize_;
    }

private:
    int numChannels_;
    int order_;
    int factor_;
    int maxBlockSize_ = 0;
    int currentOversampledSize_ = 0;

    // Per-channel, per-stage filters
    std::vector<std::vector<OversamplingStage<T>>> stages_;

    // Internal oversampled buffer storage
    std::vector<std::vector<T>> oversampledData_;
    std::vector<T*> oversampledPtrs_;

    // Temporary buffer for cascaded processing
    std::vector<T> tempBuffer_;
};

} // namespace dsp
