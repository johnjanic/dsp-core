#pragma once

#include <cmath>
#include <algorithm>
#include <cassert>

namespace dsp {

/**
 * @brief Linear parameter smoothing for click-free parameter changes.
 *
 * Provides smooth linear interpolation between parameter values over a
 * configurable time period. Used to avoid audio artifacts when parameters
 * change abruptly.
 *
 * @tparam T The value type (typically float or double)
 *
 * Usage:
 * @code
 * dsp::SmoothedValue<float> gain;
 * gain.reset(sampleRate, 0.01);  // 10ms ramp time
 * gain.setTargetValue(0.5f);     // Smoothly ramp to 0.5
 *
 * // In audio callback:
 * for (int i = 0; i < numSamples; ++i)
 *     output[i] = input[i] * gain.getNextValue();
 * @endcode
 */
template<typename T>
class SmoothedValue
{
public:
    /** Default constructor. Creates uninitialized smoother. */
    SmoothedValue() = default;

    /** Construct with initial value. */
    explicit SmoothedValue(T initialValue) noexcept
        : currentValue(initialValue)
        , targetValue(initialValue)
    {
    }

    /**
     * @brief Reset the smoother for a new sample rate and ramp time.
     *
     * @param sampleRate The sample rate in Hz.
     * @param rampLengthInSeconds The time to ramp between values.
     */
    void reset(double sampleRate, double rampLengthInSeconds) noexcept
    {
        assert(sampleRate > 0);
        assert(rampLengthInSeconds >= 0);

        stepsToTarget = static_cast<int>(std::floor(rampLengthInSeconds * sampleRate));
        currentValue = targetValue;
        countdown = 0;
    }

    /**
     * @brief Set the number of steps (samples) for the ramp.
     *
     * Alternative to reset() when sample rate is not changing.
     *
     * @param numSteps Number of samples for the ramp.
     */
    void reset(int numSteps) noexcept
    {
        assert(numSteps >= 0);
        stepsToTarget = numSteps;
        currentValue = targetValue;
        countdown = 0;
    }

    /**
     * @brief Immediately set both current and target value (no smoothing).
     *
     * Use for initialization or when smoothing is not desired.
     *
     * @param newValue The value to set immediately.
     */
    void setCurrentAndTargetValue(T newValue) noexcept
    {
        targetValue = newValue;
        currentValue = newValue;
        countdown = 0;
    }

    /**
     * @brief Set a new target value to smooth towards.
     *
     * The current value will linearly interpolate to the target over
     * the configured ramp time.
     *
     * @param newValue The target value to ramp to.
     */
    void setTargetValue(T newValue) noexcept
    {
        if (newValue == targetValue)
            return;

        targetValue = newValue;

        if (stepsToTarget <= 0)
        {
            currentValue = targetValue;
            countdown = 0;
            return;
        }

        countdown = stepsToTarget;
        step = (targetValue - currentValue) / static_cast<T>(countdown);
    }

    /**
     * @brief Get the next smoothed value and advance one sample.
     *
     * Call this once per sample in the audio processing loop.
     *
     * @return The current smoothed value.
     */
    [[nodiscard]] T getNextValue() noexcept
    {
        if (countdown <= 0)
            return targetValue;

        --countdown;
        currentValue += step;

        // Snap to target when done to avoid floating point drift
        if (countdown <= 0)
            currentValue = targetValue;

        return currentValue;
    }

    /**
     * @brief Skip ahead by multiple samples.
     *
     * @param numSamples Number of samples to skip.
     * @return The value after skipping.
     */
    T skip(int numSamples) noexcept
    {
        if (countdown <= 0)
            return targetValue;

        if (numSamples >= countdown)
        {
            currentValue = targetValue;
            countdown = 0;
            return currentValue;
        }

        currentValue += step * static_cast<T>(numSamples);
        countdown -= numSamples;
        return currentValue;
    }

    /**
     * @brief Get the target value (the value being smoothed towards).
     */
    [[nodiscard]] T getTargetValue() const noexcept
    {
        return targetValue;
    }

    /**
     * @brief Get the current interpolated value without advancing.
     */
    [[nodiscard]] T getCurrentValue() const noexcept
    {
        return countdown > 0 ? currentValue : targetValue;
    }

    /**
     * @brief Check if smoothing is currently in progress.
     *
     * @return true if ramping to target, false if at target.
     */
    [[nodiscard]] bool isSmoothing() const noexcept
    {
        return countdown > 0;
    }

    /**
     * @brief Get number of samples remaining until target is reached.
     */
    [[nodiscard]] int getNumStepsRemaining() const noexcept
    {
        return countdown;
    }

private:
    T currentValue{};
    T targetValue{};
    T step{};
    int stepsToTarget = 0;
    int countdown = 0;
};

} // namespace dsp
