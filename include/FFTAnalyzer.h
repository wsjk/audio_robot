/**
 * @file FFTAnalyzer.h
 * @brief Header file for Fast Fourier Transform audio analysis
 *
 * This class analyzes audio data using FFT (Fast Fourier Transform) to
 * convert time-domain audio signals into frequency-domain data.
 *
 * Key Concepts for Beginners:
 * - Time Domain: Audio as amplitude over time (what we hear)
 * - Frequency Domain: Audio broken down into different frequencies (pitches)
 * - FFT: Fast algorithm to convert time domain to frequency domain
 * - Frequency Bands: Groupings of frequencies (bass, mid, treble)
 *   - Bass: Low frequencies (20-250 Hz) - deep sounds
 *   - Mid: Middle frequencies (250-2000 Hz) - vocals, most instruments
 *   - Treble: High frequencies (2000-20000 Hz) - cymbals, high notes
 * - Magnitude: How strong a particular frequency is in the audio
 *
 * @author Audio Reactive Robot Project
 * @date 2026
 */

#ifndef FFT_ANALYZER_H
#define FFT_ANALYZER_H

#include <vector>
#include <fftw3.h>

/**
 * @class FFTAnalyzer
 * @brief Performs FFT analysis on audio data to extract frequency information
 *
 * This class takes audio samples (time-domain) and converts them to
 * frequency information (frequency-domain) using the FFTW3 library.
 * It can tell you how much bass, mid, and treble is in the audio.
 *
 * Example usage:
 * @code
 *   FFTAnalyzer fft(2048, 44100);
 *
 *   // Get audio from AudioCapture
 *   const auto& audioBuffer = audio.getBuffer();
 *
 *   // Analyze the audio
 *   fft.analyze(audioBuffer);
 *
 *   // Get frequency band levels (0.0 to 1.0)
 *   float bass = fft.getBassLevel();    // How much bass?
 *   float mid = fft.getMidLevel();      // How much mid?
 *   float treble = fft.getTrebleLevel(); // How much treble?
 *
 *   // Use these values to control servos, lights, etc.
 * @endcode
 */
class FFTAnalyzer {
public:
    /**
     * @brief Constructor for FFTAnalyzer
     *
     * @param fftSize Size of the FFT (should match audio buffer size)
     *                Must be a power of 2 (e.g., 1024, 2048, 4096)
     *                Larger = better frequency resolution but more CPU
     * @param sampleRate Sample rate of the audio in Hz (e.g., 44100)
     *                   Needed to convert FFT bins to actual frequencies
     */
    FFTAnalyzer(unsigned int fftSize, unsigned int sampleRate);

    /**
     * @brief Destructor - frees FFT resources
     */
    ~FFTAnalyzer();

    /**
     * @brief Perform FFT analysis on audio buffer
     *
     * Takes the audio samples and computes the FFT, which tells us
     * which frequencies are present and how strong they are.
     *
     * @param audioBuffer Vector of audio samples (typically from AudioCapture)
     *                    Size should match fftSize from constructor
     */
    void analyze(const std::vector<float>& audioBuffer);

    /**
     * @brief Get the bass frequency level (20-250 Hz)
     *
     * Returns a value indicating how much bass (low frequency) energy
     * is in the current audio. Good for detecting kicks, bass guitars, etc.
     *
     * @return Bass level from 0.0 (no bass) to 1.0 (maximum bass)
     */
    float getBassLevel() const;

    /**
     * @brief Get the mid frequency level (250-2000 Hz)
     *
     * Returns a value indicating mid-range energy. Most vocals and
     * many instruments fall into this range.
     *
     * @return Mid level from 0.0 (no mids) to 1.0 (maximum mids)
     */
    float getMidLevel() const;

    /**
     * @brief Get the treble frequency level (2000-20000 Hz)
     *
     * Returns a value indicating high frequency energy. Good for
     * detecting cymbals, hi-hats, high vocals, etc.
     *
     * @return Treble level from 0.0 (no treble) to 1.0 (maximum treble)
     */
    float getTrebleLevel() const;

    /**
     * @brief Get the magnitude at a specific frequency
     *
     * This lets you check the level of any specific frequency.
     *
     * @param frequency The frequency in Hz to check (e.g., 440 for A4 note)
     * @return Magnitude at that frequency (0.0 to 1.0)
     */
    float getMagnitudeAtFrequency(float frequency) const;

    /**
     * @brief Get magnitude in a custom frequency range
     *
     * This lets you define your own frequency band and get its level.
     *
     * @param minFreq Minimum frequency in Hz
     * @param maxFreq Maximum frequency in Hz
     * @return Average magnitude in that frequency range (0.0 to 1.0)
     */
    float getFrequencyRangeLevel(float minFreq, float maxFreq) const;

    /**
     * @brief Get the raw FFT magnitude spectrum
     *
     * For advanced users who want to access all frequency bins directly.
     * Each element represents the magnitude of a specific frequency.
     *
     * @return Const reference to the magnitude spectrum
     */
    const std::vector<float>& getMagnitudeSpectrum() const;

    /**
     * @brief Get the frequency corresponding to a specific FFT bin
     *
     * Helper function to convert from bin index to actual frequency in Hz.
     *
     * @param bin The FFT bin index
     * @return Frequency in Hz for that bin
     */
    float binToFrequency(unsigned int bin) const;

    /**
     * @brief Get the FFT bin corresponding to a specific frequency
     *
     * Helper function to convert from frequency in Hz to bin index.
     *
     * @param frequency Frequency in Hz
     * @return FFT bin index closest to that frequency
     */
    unsigned int frequencyToBin(float frequency) const;

private:
    unsigned int fftSize_;          ///< Size of the FFT (number of samples)
    unsigned int sampleRate_;       ///< Audio sample rate in Hz

    // FFTW (Fastest Fourier Transform in the West) library data structures
    float* fftInput_;               ///< Input buffer for FFT (time-domain samples)
    fftwf_complex* fftOutput_;      ///< Output buffer for FFT (complex frequency data)
    fftwf_plan fftPlan_;            ///< FFTW "plan" - pre-computed FFT algorithm

    std::vector<float> magnitudeSpectrum_; ///< Magnitude of each frequency bin

    // Cached frequency band levels
    float bassLevel_;               ///< Current bass level
    float midLevel_;                ///< Current mid level
    float trebleLevel_;             ///< Current treble level

    /**
     * @brief Compute magnitude spectrum from complex FFT output
     *
     * FFT gives us complex numbers (real + imaginary parts).
     * We compute the magnitude using sqrt(real^2 + imaginary^2).
     * This represents the "strength" of each frequency.
     */
    void computeMagnitudeSpectrum();

    /**
     * @brief Calculate frequency band levels (bass, mid, treble)
     *
     * After computing the magnitude spectrum, this function averages
     * the magnitudes in specific frequency ranges to get band levels.
     */
    void calculateFrequencyBands();

    /**
     * @brief Apply window function to audio before FFT
     *
     * A window function reduces artifacts at the edges of the FFT.
     * We use a Hann window, which smoothly tapers the signal to zero
     * at both ends of the buffer.
     *
     * @param buffer Audio buffer to apply window to
     */
    void applyWindow(std::vector<float>& buffer);
};

#endif // FFT_ANALYZER_H

