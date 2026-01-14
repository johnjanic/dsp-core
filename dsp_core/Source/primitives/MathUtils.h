#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace dsp {

// =============================================================================
// Math Constants
// =============================================================================

template <typename T>
struct MathConstants
{
    static constexpr T pi = static_cast<T>(3.14159265358979323846);
    static constexpr T twoPi = static_cast<T>(6.28318530717958647692);
    static constexpr T halfPi = static_cast<T>(1.57079632679489661923);
    static constexpr T euler = static_cast<T>(2.71828182845904523536);
    static constexpr T sqrt2 = static_cast<T>(1.41421356237309504880);
    static constexpr T invSqrt2 = static_cast<T>(0.70710678118654752440);
};

// =============================================================================
// Clamping and Mapping Utilities
// =============================================================================

/**
 * @brief Clamp a value to a range [min, max].
 *
 * Drop-in replacement for juce::jlimit.
 */
template <typename T>
[[nodiscard]] constexpr T clamp(T min, T max, T value) noexcept
{
    return std::clamp(value, min, max);
}

/**
 * @brief Map a value from one range to another.
 *
 * Drop-in replacement for juce::jmap.
 *
 * @param value The value to map
 * @param inMin The minimum of the input range
 * @param inMax The maximum of the input range
 * @param outMin The minimum of the output range
 * @param outMax The maximum of the output range
 * @return The mapped value
 */
template <typename T>
[[nodiscard]] constexpr T mapValue(T value, T inMin, T inMax, T outMin, T outMax) noexcept
{
    return outMin + (value - inMin) * (outMax - outMin) / (inMax - inMin);
}

// =============================================================================
// Comparison Utilities
// =============================================================================

/**
 * @brief Check if two floating point values are approximately equal.
 */
template <typename T>
[[nodiscard]] constexpr bool approximatelyEqual(
    T a,
    T b,
    T tolerance = std::numeric_limits<T>::epsilon() * T(100)) noexcept
{
    return std::abs(a - b) <= tolerance;
}

/**
 * @brief Check if a value is approximately zero.
 */
template <typename T>
[[nodiscard]] constexpr bool approximatelyZero(
    T value,
    T tolerance = std::numeric_limits<T>::epsilon() * T(100)) noexcept
{
    return std::abs(value) <= tolerance;
}

// =============================================================================
// Range Utilities
// =============================================================================

template <typename T>
struct Range
{
    T start{};
    T end{};

    constexpr Range() = default;
    constexpr Range(T s, T e) : start(s), end(e) {}

    [[nodiscard]] constexpr T getStart() const noexcept { return start; }
    [[nodiscard]] constexpr T getEnd() const noexcept { return end; }
    [[nodiscard]] constexpr T getLength() const noexcept { return end - start; }

    [[nodiscard]] constexpr T clipValue(T value) const noexcept
    {
        return std::clamp(value, start, end);
    }

    [[nodiscard]] constexpr bool contains(T value) const noexcept
    {
        return value >= start && value <= end;
    }

    [[nodiscard]] constexpr bool isEmpty() const noexcept { return end <= start; }
};

// =============================================================================
// Interpolation Utilities
// =============================================================================

/**
 * @brief Linear interpolation between two values.
 */
template <typename T>
[[nodiscard]] constexpr T lerp(T a, T b, T t) noexcept
{
    return a + t * (b - a);
}

} // namespace dsp
