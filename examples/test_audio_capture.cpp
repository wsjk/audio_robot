/**
 * @file test_audio_capture.cpp
 * @brief Test program for audio capture functionality
 *
 * This simple program demonstrates how to use the AudioCapture class.
 * It captures audio for a few seconds and shows the audio levels.
 *
 * Usage: sudo ./test_audio_capture
 */

#include "AudioCapture.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <cmath>

/**
 * Calculate RMS (Root Mean Square) level of audio buffer
 *
 * RMS gives us a measure of the overall "loudness" of the audio.
 * It's like an average, but for audio signals.
 *
 * Formula: RMS = sqrt(sum of squares / count)
 */
float calculateRMS(const std::vector<float>& buffer) {
    float sum = 0.0f;
    for (float sample : buffer) {
        sum += sample * sample;
    }
    return sqrt(sum / buffer.size());
}

/**
 * Find peak (maximum absolute value) in audio buffer
 */
float calculatePeak(const std::vector<float>& buffer) {
    float peak = 0.0f;
    for (float sample : buffer) {
        float absSample = fabs(sample);
        if (absSample > peak) {
            peak = absSample;
        }
    }
    return peak;
}

/**
 * Draw a simple text-based VU meter
 */
void drawVUMeter(float level, int width = 50) {
    int bars = static_cast<int>(level * width);
    bars = std::min(bars, width);

    std::cout << "|";
    for (int i = 0; i < width; ++i) {
        if (i < bars) {
            std::cout << "█";
        } else {
            std::cout << " ";
        }
    }
    std::cout << "| " << std::fixed << std::setprecision(2) << level;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "    Audio Capture Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Configuration
    const char* DEVICE = "default";
    const unsigned int SAMPLE_RATE = 44100;
    const unsigned int BUFFER_SIZE = 2048;
    const unsigned int TEST_DURATION_SEC = 10;

    // Create audio capture object
    std::cout << "Creating AudioCapture object..." << std::endl;
    AudioCapture audio(DEVICE, SAMPLE_RATE, BUFFER_SIZE);

    // Initialize
    std::cout << "Initializing audio device..." << std::endl;
    if (!audio.initialize()) {
        std::cerr << "FATAL: Failed to initialize audio" << std::endl;
        std::cerr << "\nTroubleshooting tips:" << std::endl;
        std::cerr << "1. Run 'arecord -l' to list audio devices" << std::endl;
        std::cerr << "2. Try recording: arecord -D default -f S16_LE -r 44100 -d 5 test.wav" << std::endl;
        std::cerr << "3. Make sure USB microphone is plugged in" << std::endl;
        return 1;
    }

    std::cout << "\nCapturing audio for " << TEST_DURATION_SEC << " seconds..." << std::endl;
    std::cout << "Make some noise to see the levels!\n" << std::endl;

    // Calculate how many captures we need
    unsigned int capturesPerSecond = SAMPLE_RATE / BUFFER_SIZE;
    unsigned int totalCaptures = capturesPerSecond * TEST_DURATION_SEC;

    // Capture loop
    for (unsigned int i = 0; i < totalCaptures; ++i) {
        // Capture audio
        int samplesRead = audio.capture();
        if (samplesRead <= 0) {
            std::cerr << "\nERROR: Failed to capture audio" << std::endl;
            continue;
        }

        // Get the buffer
        const std::vector<float>& buffer = audio.getBuffer();

        // Calculate levels
        float rms = calculateRMS(buffer);
        float peak = calculatePeak(buffer);

        // Display
        std::cout << "\rRMS: ";
        drawVUMeter(rms, 30);
        std::cout << "  Peak: ";
        drawVUMeter(peak, 30);
        std::cout << "    ";
        std::cout.flush();

        // Small delay for readability
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::cout << "\n\nTest complete!" << std::endl;

    // Cleanup
    audio.cleanup();

    return 0;
}

