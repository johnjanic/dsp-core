#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace dsp {

/**
 * @brief Decibel conversion utilities.
 *
 * Provides conversion between decibels (dB) and linear gain values.
 * Template functions support float and double precision.
 */
namespace Decibels {

    /**
     * @brief Default floor for dB calculations.
     *
     * Gains below this threshold return negative infinity dB.
     * Matches JUCE default behavior.
     */
    template<typename T>
    constexpr T defaultMinusInfinitydB = static_cast<T>(-100.0);

    /**
     * @brief Convert decibels to linear gain.
     *
     * @param decibels The decibel value to convert.
     * @return Linear gain value. 0dB = 1.0, -6dB ≈ 0.5, +6dB ≈ 2.0
     *
     * Formula: gain = 10^(dB/20)
     */
    template<typename T>
    [[nodiscard]] constexpr T decibelsToGain(T decibels) noexcept
    {
        return decibels > defaultMinusInfinitydB<T>
            ? std::pow(static_cast<T>(10.0), decibels * static_cast<T>(0.05))
            : T{};
    }

    /**
     * @brief Convert decibels to linear gain with custom floor.
     *
     * @param decibels The decibel value to convert.
     * @param minusInfinityDb Values at or below this return 0.
     * @return Linear gain value.
     */
    template<typename T>
    [[nodiscard]] constexpr T decibelsToGain(T decibels, T minusInfinityDb) noexcept
    {
        return decibels > minusInfinityDb
            ? std::pow(static_cast<T>(10.0), decibels * static_cast<T>(0.05))
            : T{};
    }

    /**
     * @brief Convert linear gain to decibels.
     *
     * @param gain The linear gain value to convert.
     * @return Decibel value. Returns -100dB for zero/negative gain.
     *
     * Formula: dB = 20 * log10(gain)
     */
    template<typename T>
    [[nodiscard]] constexpr T gainToDecibels(T gain) noexcept
    {
        return gain > T{}
            ? std::max(defaultMinusInfinitydB<T>,
                       static_cast<T>(std::log10(gain)) * static_cast<T>(20.0))
            : defaultMinusInfinitydB<T>;
    }

    /**
     * @brief Convert linear gain to decibels with custom floor.
     *
     * @param gain The linear gain value to convert.
     * @param minusInfinityDb Minimum dB value to return.
     * @return Decibel value, clamped to minusInfinityDb.
     */
    template<typename T>
    [[nodiscard]] constexpr T gainToDecibels(T gain, T minusInfinityDb) noexcept
    {
        return gain > T{}
            ? std::max(minusInfinityDb,
                       static_cast<T>(std::log10(gain)) * static_cast<T>(20.0))
            : minusInfinityDb;
    }

} // namespace Decibels
} // namespace dsp
