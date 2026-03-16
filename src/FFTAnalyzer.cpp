/**
 * @file FFTAnalyzer.cpp
 * @brief Implementation of Fast Fourier Transform audio analysis
 *
 * This file implements the FFT analysis functionality. FFT is a mathematical
 * algorithm that converts audio from time-domain (amplitude over time) to
 * frequency-domain (showing which frequencies are present).
 *
 * Think of it like a prism splitting white light into a rainbow - FFT splits
 * audio into its component frequencies.
 */

#include "FFTAnalyzer.h"
#include <iostream>
#include <cmath>
#include <algorithm>

/**
 * Constructor: Set up FFT analyzer with specified size and sample rate
 */
FFTAnalyzer::FFTAnalyzer(unsigned int fftSize, unsigned int sampleRate)
    : fftSize_(fftSize)
    , sampleRate_(sampleRate)
    , fftInput_(nullptr)
    , fftOutput_(nullptr)
    , fftPlan_(nullptr)
    , bassLevel_(0.0f)
    , midLevel_(0.0f)
    , trebleLevel_(0.0f)
{
    // Allocate memory for FFT input (time-domain samples)
    // fftwf_malloc ensures proper alignment for SIMD optimization
    fftInput_ = (float*) fftwf_malloc(sizeof(float) * fftSize_);

    // Allocate memory for FFT output (frequency-domain complex numbers)
    // FFT output is half the size + 1 because of symmetry in real signals
    fftOutput_ = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * (fftSize_ / 2 + 1));

    // Create the FFT "plan" - this pre-computes the FFT algorithm
    // FFTW_ESTIMATE is faster to create but slightly slower to execute
    // For better performance, use FFTW_MEASURE (but takes longer to initialize)
    fftPlan_ = fftwf_plan_dft_r2c_1d(fftSize_, fftInput_, fftOutput_, FFTW_ESTIMATE);

    // Pre-allocate the magnitude spectrum vector
    magnitudeSpectrum_.resize(fftSize_ / 2 + 1, 0.0f);

    if (!fftPlan_) {
        std::cerr << "ERROR: Failed to create FFT plan" << std::endl;
    } else {
        std::cout << "FFT Analyzer initialized:" << std::endl;
        std::cout << "  FFT Size: " << fftSize_ << std::endl;
        std::cout << "  Sample Rate: " << sampleRate_ << " Hz" << std::endl;
        std::cout << "  Frequency Resolution: "
                  << (float)sampleRate_ / (float)fftSize_ << " Hz/bin" << std::endl;
    }
}

/**
 * Destructor: Free all FFT resources
 */
FFTAnalyzer::~FFTAnalyzer() {
    if (fftPlan_) {
        fftwf_destroy_plan(fftPlan_);
    }
    if (fftInput_) {
        fftwf_free(fftInput_);
    }
    if (fftOutput_) {
        fftwf_free(fftOutput_);
    }
}

/**
 * Perform FFT analysis on an audio buffer
 *
 * This is the main function that does all the work:
 * 1. Apply window function to reduce edge artifacts
 * 2. Execute the FFT
 * 3. Compute magnitudes from complex numbers
 * 4. Calculate frequency band levels
 */
void FFTAnalyzer::analyze(const std::vector<float>& audioBuffer) {
    // Check that buffer size matches FFT size
    if (audioBuffer.size() != fftSize_) {
        std::cerr << "ERROR: Buffer size (" << audioBuffer.size()
                  << ") doesn't match FFT size (" << fftSize_ << ")" << std::endl;
        return;
    }

    // Copy audio data to FFT input buffer
    // We copy instead of using directly because we'll apply a window
    for (size_t i = 0; i < fftSize_; ++i) {
        fftInput_[i] = audioBuffer[i];
    }

    // Apply Hann window to reduce spectral leakage
    // (explained in detail below)
    for (size_t i = 0; i < fftSize_; ++i) {
        float window = 0.5f * (1.0f - cos(2.0f * M_PI * i / (fftSize_ - 1)));
        fftInput_[i] *= window;
    }

    // Execute the FFT - this is where the magic happens!
    // The input (time-domain) is transformed into output (frequency-domain)
    fftwf_execute(fftPlan_);

    // Compute magnitude spectrum from complex FFT output
    computeMagnitudeSpectrum();

    // Calculate bass, mid, and treble levels
    calculateFrequencyBands();
}

/**
 * Compute magnitude spectrum from complex FFT output
 *
 * FFT gives us complex numbers (real + imaginary parts) for each frequency.
 * The magnitude tells us how strong that frequency is in the audio.
 *
 * Magnitude = sqrt(real² + imaginary²)
 */
void FFTAnalyzer::computeMagnitudeSpectrum() {
    // We only process the first half of the FFT output
    // because the second half is a mirror image (for real signals)
    for (size_t i = 0; i < magnitudeSpectrum_.size(); ++i) {
        // Get real and imaginary parts
        float real = fftOutput_[i][0];
        float imag = fftOutput_[i][1];

        // Compute magnitude: sqrt(real² + imaginary²)
        float magnitude = sqrt(real * real + imag * imag);

        // Normalize by FFT size to get consistent results
        // regardless of buffer size
        magnitudeSpectrum_[i] = magnitude / fftSize_;
    }
}

/**
 * Calculate frequency band levels (bass, mid, treble)
 *
 * This averages the magnitudes in specific frequency ranges:
 * - Bass: 20-250 Hz (kicks, bass guitar, low rumble)
 * - Mid: 250-2000 Hz (vocals, most instruments)
 * - Treble: 2000-20000 Hz (cymbals, high hats, brilliance)
 */
void FFTAnalyzer::calculateFrequencyBands() {
    // Define frequency ranges (in Hz)
    const float bassMin = 20.0f;
    const float bassMax = 250.0f;
    const float midMin = 250.0f;
    const float midMax = 2000.0f;
    const float trebleMin = 2000.0f;
    const float trebleMax = 20000.0f;

    // Calculate bass level
    bassLevel_ = getFrequencyRangeLevel(bassMin, bassMax);

    // Calculate mid level
    midLevel_ = getFrequencyRangeLevel(midMin, midMax);

    // Calculate treble level
    trebleLevel_ = getFrequencyRangeLevel(trebleMin, trebleMax);
}

/**
 * Get the bass frequency level
 */
float FFTAnalyzer::getBassLevel() const {
    return bassLevel_;
}

/**
 * Get the mid frequency level
 */
float FFTAnalyzer::getMidLevel() const {
    return midLevel_;
}

/**
 * Get the treble frequency level
 */
float FFTAnalyzer::getTrebleLevel() const {
    return trebleLevel_;
}

/**
 * Get magnitude at a specific frequency
 */
float FFTAnalyzer::getMagnitudeAtFrequency(float frequency) const {
    // Convert frequency to FFT bin index
    unsigned int bin = frequencyToBin(frequency);

    // Check if bin is valid
    if (bin >= magnitudeSpectrum_.size()) {
        return 0.0f;
    }

    return magnitudeSpectrum_[bin];
}

/**
 * Get average magnitude in a frequency range
 *
 * This averages all the FFT bins that fall within the specified range.
 */
float FFTAnalyzer::getFrequencyRangeLevel(float minFreq, float maxFreq) const {
    // Convert frequencies to bin indices
    unsigned int minBin = frequencyToBin(minFreq);
    unsigned int maxBin = frequencyToBin(maxFreq);

    // Clamp to valid range
    minBin = std::min(minBin, static_cast<unsigned int>(magnitudeSpectrum_.size() - 1));
    maxBin = std::min(maxBin, static_cast<unsigned int>(magnitudeSpectrum_.size() - 1));

    // If range is invalid, return 0
    if (minBin >= maxBin) {
        return 0.0f;
    }

    // Calculate average magnitude in this range
    float sum = 0.0f;
    for (unsigned int i = minBin; i <= maxBin; ++i) {
        sum += magnitudeSpectrum_[i];
    }

    float average = sum / (maxBin - minBin + 1);

    // Normalize to roughly 0.0-1.0 range
    // This scaling factor is empirically determined and may need adjustment
    float normalized = average * 10.0f;

    // Clamp to 0.0-1.0
    return std::min(std::max(normalized, 0.0f), 1.0f);
}

/**
 * Get the raw magnitude spectrum
 */
const std::vector<float>& FFTAnalyzer::getMagnitudeSpectrum() const {
    return magnitudeSpectrum_;
}

/**
 * Convert FFT bin index to frequency in Hz
 *
 * Formula: frequency = (bin * sampleRate) / fftSize
 *
 * Example: With 44100 Hz sample rate and 2048 FFT size:
 *   - Bin 0 = 0 Hz (DC offset)
 *   - Bin 1 = 21.5 Hz
 *   - Bin 2 = 43.1 Hz
 *   - etc.
 */
float FFTAnalyzer::binToFrequency(unsigned int bin) const {
    return (float)bin * (float)sampleRate_ / (float)fftSize_;
}

/**
 * Convert frequency in Hz to FFT bin index
 *
 * This is the inverse of binToFrequency()
 */
unsigned int FFTAnalyzer::frequencyToBin(float frequency) const {
    // Formula: bin = (frequency * fftSize) / sampleRate
    unsigned int bin = (unsigned int)((frequency * fftSize_) / sampleRate_);

    // Clamp to valid range
    return std::min(bin, static_cast<unsigned int>(magnitudeSpectrum_.size() - 1));
}

/**
 * Apply Hann window to audio buffer
 *
 * What is a window function and why do we need it?
 *
 * When we perform FFT, we're analyzing a finite chunk of audio. The FFT
 * assumes this chunk repeats forever, but in reality the audio at the
 * start and end of our chunk probably don't match. This creates artifacts
 * (fake frequencies) in the FFT output.
 *
 * A window function smoothly fades the audio to zero at both ends,
 * eliminating these artifacts. The Hann window is a popular choice.
 *
 * Formula: window[i] = 0.5 * (1 - cos(2*π*i / (N-1)))
 */
void FFTAnalyzer::applyWindow(std::vector<float>& buffer) {
    for (size_t i = 0; i < buffer.size(); ++i) {
        float window = 0.5f * (1.0f - cos(2.0f * M_PI * i / (buffer.size() - 1)));
        buffer[i] *= window;
    }
}

