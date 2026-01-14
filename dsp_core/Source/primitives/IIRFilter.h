#pragma once

#include <cmath>
#include <array>
#include <algorithm>
#include <platform/AudioBuffer.h>

namespace dsp {

/**
 * @brief IIR Biquad Filter Coefficients.
 *
 * Stores normalized filter coefficients for a second-order IIR section.
 * Provides static factory methods for common filter types.
 *
 * @tparam T The coefficient type (float or double)
 */
template<typename T>
struct IIRCoefficients
{
    // Feedforward coefficients (numerator)
    T b0 = T(1);
    T b1 = T(0);
    T b2 = T(0);

    // Feedback coefficients (denominator) - a0 normalized to 1
    T a1 = T(0);
    T a2 = T(0);

    /**
     * @brief Create first-order high-pass filter coefficients.
     *
     * Single-pole high-pass filter (6 dB/octave slope).
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The -3dB cutoff frequency in Hz.
     * @return Coefficients for the high-pass filter.
     */
    static IIRCoefficients<T> makeFirstOrderHighPass(double sampleRate, double frequency)
    {
        const T n = static_cast<T>(std::tan(M_PI * frequency / sampleRate));
        const T a0 = T(1) + n;

        IIRCoefficients<T> c;
        c.b0 = T(1) / a0;
        c.b1 = -c.b0;
        c.b2 = T(0);
        c.a1 = (n - T(1)) / a0;
        c.a2 = T(0);
        return c;
    }

    /**
     * @brief Create first-order low-pass filter coefficients.
     *
     * Single-pole low-pass filter (6 dB/octave slope).
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The -3dB cutoff frequency in Hz.
     * @return Coefficients for the low-pass filter.
     */
    static IIRCoefficients<T> makeFirstOrderLowPass(double sampleRate, double frequency)
    {
        const T n = static_cast<T>(std::tan(M_PI * frequency / sampleRate));
        const T a0 = T(1) + n;

        IIRCoefficients<T> c;
        c.b0 = n / a0;
        c.b1 = c.b0;
        c.b2 = T(0);
        c.a1 = (n - T(1)) / a0;
        c.a2 = T(0);
        return c;
    }

    /**
     * @brief Create second-order (biquad) high-pass filter coefficients.
     *
     * Butterworth high-pass filter (12 dB/octave slope).
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The -3dB cutoff frequency in Hz.
     * @return Coefficients for the high-pass filter.
     */
    static IIRCoefficients<T> makeHighPass(double sampleRate, double frequency)
    {
        return makeHighPass(sampleRate, frequency, static_cast<T>(1.0 / std::sqrt(2.0)));
    }

    /**
     * @brief Create second-order high-pass filter with Q control.
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The -3dB cutoff frequency in Hz.
     * @param Q The quality factor (0.707 for Butterworth).
     * @return Coefficients for the high-pass filter.
     */
    static IIRCoefficients<T> makeHighPass(double sampleRate, double frequency, T Q)
    {
        const T w0 = static_cast<T>(2.0 * M_PI * frequency / sampleRate);
        const T cosw0 = std::cos(w0);
        const T sinw0 = std::sin(w0);
        const T alpha = sinw0 / (T(2) * Q);

        const T a0 = T(1) + alpha;

        IIRCoefficients<T> c;
        c.b0 = ((T(1) + cosw0) / T(2)) / a0;
        c.b1 = (-(T(1) + cosw0)) / a0;
        c.b2 = c.b0;
        c.a1 = (T(-2) * cosw0) / a0;
        c.a2 = (T(1) - alpha) / a0;
        return c;
    }

    /**
     * @brief Create second-order (biquad) low-pass filter coefficients.
     *
     * Butterworth low-pass filter (12 dB/octave slope).
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The -3dB cutoff frequency in Hz.
     * @return Coefficients for the low-pass filter.
     */
    static IIRCoefficients<T> makeLowPass(double sampleRate, double frequency)
    {
        return makeLowPass(sampleRate, frequency, static_cast<T>(1.0 / std::sqrt(2.0)));
    }

    /**
     * @brief Create second-order low-pass filter with Q control.
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The -3dB cutoff frequency in Hz.
     * @param Q The quality factor (0.707 for Butterworth).
     * @return Coefficients for the low-pass filter.
     */
    static IIRCoefficients<T> makeLowPass(double sampleRate, double frequency, T Q)
    {
        const T w0 = static_cast<T>(2.0 * M_PI * frequency / sampleRate);
        const T cosw0 = std::cos(w0);
        const T sinw0 = std::sin(w0);
        const T alpha = sinw0 / (T(2) * Q);

        const T a0 = T(1) + alpha;

        IIRCoefficients<T> c;
        c.b0 = ((T(1) - cosw0) / T(2)) / a0;
        c.b1 = (T(1) - cosw0) / a0;
        c.b2 = c.b0;
        c.a1 = (T(-2) * cosw0) / a0;
        c.a2 = (T(1) - alpha) / a0;
        return c;
    }

    /**
     * @brief Create band-pass filter coefficients.
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The center frequency in Hz.
     * @param Q The quality factor (bandwidth control).
     * @return Coefficients for the band-pass filter.
     */
    static IIRCoefficients<T> makeBandPass(double sampleRate, double frequency, T Q)
    {
        const T w0 = static_cast<T>(2.0 * M_PI * frequency / sampleRate);
        const T cosw0 = std::cos(w0);
        const T sinw0 = std::sin(w0);
        const T alpha = sinw0 / (T(2) * Q);

        const T a0 = T(1) + alpha;

        IIRCoefficients<T> c;
        c.b0 = (sinw0 / T(2)) / a0;
        c.b1 = T(0);
        c.b2 = (-sinw0 / T(2)) / a0;
        c.a1 = (T(-2) * cosw0) / a0;
        c.a2 = (T(1) - alpha) / a0;
        return c;
    }

    /**
     * @brief Create notch (band-reject) filter coefficients.
     *
     * @param sampleRate The sample rate in Hz.
     * @param frequency The notch center frequency in Hz.
     * @param Q The quality factor (notch width control).
     * @return Coefficients for the notch filter.
     */
    static IIRCoefficients<T> makeNotch(double sampleRate, double frequency, T Q)
    {
        const T w0 = static_cast<T>(2.0 * M_PI * frequency / sampleRate);
        const T cosw0 = std::cos(w0);
        const T sinw0 = std::sin(w0);
        const T alpha = sinw0 / (T(2) * Q);

        const T a0 = T(1) + alpha;

        IIRCoefficients<T> c;
        c.b0 = T(1) / a0;
        c.b1 = (T(-2) * cosw0) / a0;
        c.b2 = c.b0;
        c.a1 = c.b1;
        c.a2 = (T(1) - alpha) / a0;
        return c;
    }
};


/**
 * @brief IIR Biquad Filter.
 *
 * Second-order IIR filter using Direct Form II Transposed implementation.
 * This is the most numerically stable form for floating point computation.
 *
 * @tparam T The sample type (float or double)
 */
template<typename T>
class IIRFilter
{
public:
    using Coefficients = IIRCoefficients<T>;

    /** Default constructor with pass-through coefficients. */
    IIRFilter() = default;

    /** Construct with specific coefficients. */
    explicit IIRFilter(const Coefficients& coeffs)
        : coefficients(coeffs)
    {
    }

    /**
     * @brief Set new filter coefficients.
     *
     * @param newCoeffs The new coefficients to use.
     */
    void setCoefficients(const Coefficients& newCoeffs) noexcept
    {
        coefficients = newCoeffs;
    }

    /**
     * @brief Get the current filter coefficients.
     */
    [[nodiscard]] const Coefficients& getCoefficients() const noexcept
    {
        return coefficients;
    }

    /**
     * @brief Reset filter state (clear delay lines).
     *
     * Call after changing coefficients or to clear transients.
     */
    void reset() noexcept
    {
        s1 = T(0);
        s2 = T(0);
    }

    /**
     * @brief Process a single sample.
     *
     * Direct Form II Transposed implementation.
     *
     * @param x Input sample.
     * @return Filtered output sample.
     */
    [[nodiscard]] T processSample(T x) noexcept
    {
        const T y = coefficients.b0 * x + s1;
        s1 = coefficients.b1 * x - coefficients.a1 * y + s2;
        s2 = coefficients.b2 * x - coefficients.a2 * y;
        return y;
    }

    /**
     * @brief Process a single channel of an AudioBuffer in-place.
     *
     * @param buffer AudioBuffer to process.
     * @param channel Channel index to process.
     */
    void processBlock(platform::AudioBuffer<T>& buffer, int channel) noexcept
    {
        T* samples = buffer.getWritePointer(channel);
        const int numSamples = buffer.getNumSamples();
        for (int i = 0; i < numSamples; ++i)
        {
            samples[i] = processSample(samples[i]);
        }
    }

private:
    Coefficients coefficients;
    T s1 = T(0);  // First delay state
    T s2 = T(0);  // Second delay state
};

} // namespace dsp
