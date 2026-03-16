/**
 * @file test_fft.cpp
 * @brief Test program for FFT analysis functionality
 *
 * This program demonstrates how to use the FFTAnalyzer class.
 * It captures audio, performs FFT analysis, and displays frequency levels.
 *
 * Usage: sudo ./test_fft
 */

#include "AudioCapture.h"
#include "FFTAnalyzer.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

/**
 * Draw a simple text-based bar chart
 */
void drawBar(const std::string& label, float value, int width = 40) {
    int bars = static_cast<int>(value * width);
    bars = std::min(bars, width);

    std::cout << std::setw(12) << std::left << label << " |";
    for (int i = 0; i < width; ++i) {
        if (i < bars) {
            std::cout << "█";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "| " << std::fixed << std::setprecision(2) << value << std::endl;
}

/**
 * Display frequency spectrum as ASCII art
 */
void displaySpectrum(const FFTAnalyzer& fft, int numBands = 20) {
    const float maxFreq = 20000.0f;  // Maximum audible frequency
    const float freqStep = maxFreq / numBands;

    std::cout << "\nFrequency Spectrum:" << std::endl;

    for (int i = 0; i < numBands; ++i) {
        float minFreq = i * freqStep;
        float maxFreqBand = (i + 1) * freqStep;
        float level = fft.getFrequencyRangeLevel(minFreq, maxFreqBand);

        // Format frequency label
        std::string label;
        if (maxFreqBand < 1000) {
            label = std::to_string((int)maxFreqBand) + "Hz";
        } else {
            label = std::to_string((int)(maxFreqBand / 1000)) + "kHz";
        }

        drawBar(label, level, 30);
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        FFT Analysis Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Configuration
    const char* DEVICE = "default";
    const unsigned int SAMPLE_RATE = 44100;
    const unsigned int BUFFER_SIZE = 2048;
    const unsigned int TEST_DURATION_SEC = 30;

    // Initialize audio capture
    std::cout << "Initializing audio capture..." << std::endl;
    AudioCapture audio(DEVICE, SAMPLE_RATE, BUFFER_SIZE);

    if (!audio.initialize()) {
        std::cerr << "FATAL: Failed to initialize audio" << std::endl;
        return 1;
    }

    // Initialize FFT analyzer
    std::cout << "Initializing FFT analyzer..." << std::endl;
    FFTAnalyzer fft(BUFFER_SIZE, SAMPLE_RATE);

    std::cout << "\nAnalyzing audio for " << TEST_DURATION_SEC << " seconds..." << std::endl;
    std::cout << "Play some music or make sounds!\n" << std::endl;

    // Calculate how many captures we need
    unsigned int capturesPerSecond = SAMPLE_RATE / BUFFER_SIZE;
    unsigned int totalCaptures = capturesPerSecond * TEST_DURATION_SEC;

    // Analysis loop
    for (unsigned int i = 0; i < totalCaptures; ++i) {
        // Capture audio
        int samplesRead = audio.capture();
        if (samplesRead <= 0) {
            std::cerr << "\nERROR: Failed to capture audio" << std::endl;
            continue;
        }

        // Analyze with FFT
        const std::vector<float>& buffer = audio.getBuffer();
        fft.analyze(buffer);

        // Get frequency band levels
        float bass = fft.getBassLevel();
        float mid = fft.getMidLevel();
        float treble = fft.getTrebleLevel();

        // Clear screen (ANSI escape code)
        std::cout << "\033[2J\033[1;1H";

        // Display results
        std::cout << "========================================" << std::endl;
        std::cout << "        FFT Analysis Results" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;

        std::cout << "Frequency Bands:" << std::endl;
        std::cout << "----------------" << std::endl;
        drawBar("Bass (20-250Hz)", bass, 40);
        drawBar("Mid (250-2kHz)", mid, 40);
        drawBar("Treble (2k-20kHz)", treble, 40);
        std::cout << std::endl;

        // Display some specific frequencies
        std::cout << "Notable Frequencies:" << std::endl;
        std::cout << "-------------------" << std::endl;
        float freq60Hz = fft.getMagnitudeAtFrequency(60.0f);   // AC hum
        float freq440Hz = fft.getMagnitudeAtFrequency(440.0f); // A4 note
        float freq1kHz = fft.getMagnitudeAtFrequency(1000.0f); // 1 kHz tone

        drawBar("60 Hz (AC hum)", freq60Hz, 40);
        drawBar("440 Hz (A4 note)", freq440Hz, 40);
        drawBar("1 kHz", freq1kHz, 40);
        std::cout << std::endl;

        // Display simple spectrum
        displaySpectrum(fft, 15);

        std::cout << "\nPress Ctrl+C to stop..." << std::endl;

        // Update rate (refresh display 10 times per second)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n\nTest complete!" << std::endl;

    // Cleanup
    audio.cleanup();

    return 0;
}

