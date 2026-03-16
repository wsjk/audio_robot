/**
 * @file main.cpp
 * @brief Main application for audio reactive robot
 *
 * This program ties everything together:
 * 1. Captures audio from USB microphone
 * 2. Analyzes audio using FFT to get bass, mid, and treble levels
 * 3. Controls servos based on the frequency levels
 *
 * You can modify the mapping between audio frequencies and servo movements
 * to create different behaviors.
 *
 * @author Audio Reactive Robot Project
 * @date 2026
 */

#include "AudioCapture.h"
#include "FFTAnalyzer.h"
#include "ServoController.h"
#include <iostream>
#include <iomanip>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include <cmath>

// ============================================================================
// Configuration Constants
// ============================================================================

// Audio settings
const char* AUDIO_DEVICE = "default";  // Change to "hw:1,0" if default doesn't work
const unsigned int SAMPLE_RATE = 44100;  // CD-quality audio (44.1 kHz)
const unsigned int BUFFER_SIZE = 2048;   // Must be power of 2 for FFT

// Servo settings (GPIO pin numbers using BCM numbering)
const std::vector<unsigned int> SERVO_PINS = {17, 18, 27, 22};
const unsigned int MIN_PULSE = 500;     // Minimum pulse width in microseconds
const unsigned int MAX_PULSE = 2500;    // Maximum pulse width in microseconds

// Control parameters
const float SMOOTHING_FACTOR = 0.3f;    // Lower = smoother but slower response (0.0-1.0)
const float SENSITIVITY = 2.0f;         // Multiplier for audio levels (higher = more reactive)
const unsigned int UPDATE_RATE_MS = 20; // Update servos every 20ms (50 Hz)

// Global flag for clean shutdown
std::atomic<bool> running(true);

/**
 * Signal handler for graceful shutdown
 *
 * When user presses Ctrl+C, this function is called.
 * We set running = false to exit the main loop cleanly.
 */
void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "\n\nReceived shutdown signal, cleaning up..." << std::endl;
        running = false;
    }
}

/**
 * Map audio level (0.0-1.0) to servo angle (0-180 degrees)
 *
 * This function converts an audio level to a servo angle.
 * You can modify this to create different movement patterns.
 *
 * @param level Audio level (0.0 = silent, 1.0 = maximum)
 * @param sensitivity Multiplier for the level
 * @return Servo angle in degrees (0-180)
 */
float mapLevelToAngle(float level, float sensitivity = 1.0f) {
    // Apply sensitivity
    level *= sensitivity;

    // Clamp to 0.0-1.0 range
    level = std::min(std::max(level, 0.0f), 1.0f);

    // Map to servo angle (0-180 degrees)
    // You could also map to a smaller range, e.g., 45-135 for less extreme movement
    return level * 180.0f;
}

/**
 * Apply exponential smoothing to reduce jitter
 *
 * Servos can jitter if they change position too quickly. This function
 * smooths the values over time using exponential smoothing.
 *
 * Formula: smoothed = (alpha * new) + ((1 - alpha) * old)
 *
 * @param newValue New value from audio analysis
 * @param oldValue Previous smoothed value
 * @param alpha Smoothing factor (0.0 = no change, 1.0 = no smoothing)
 * @return Smoothed value
 */
float smoothValue(float newValue, float oldValue, float alpha) {
    return (alpha * newValue) + ((1.0f - alpha) * oldValue);
}

/**
 * Print status information to console
 *
 * Shows current audio levels and servo positions for debugging.
 */
void printStatus(float bass, float mid, float treble,
                const std::vector<float>& servoAngles) {
    // Use \r to overwrite the same line (creates a "live updating" effect)
    std::cout << "\r";
    std::cout << "Bass: " << std::fixed << std::setprecision(2) << bass << " | ";
    std::cout << "Mid: " << mid << " | ";
    std::cout << "Treble: " << treble << " | ";
    std::cout << "Servos: [";
    for (size_t i = 0; i < servoAngles.size(); ++i) {
        std::cout << (int)servoAngles[i];
        if (i < servoAngles.size() - 1) std::cout << "° ";
    }
    std::cout << "°]   ";
    std::cout.flush();  // Force output to appear immediately
}

/**
 * Main function
 */
int main(int argc, char* argv[]) {
    // Print welcome message
    std::cout << "========================================" << std::endl;
    std::cout << "  Audio Reactive Robot for Raspberry Pi" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Set up signal handler for Ctrl+C
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    // ========================================================================
    // Initialize Audio Capture
    // ========================================================================
    std::cout << "Initializing audio capture..." << std::endl;
    AudioCapture audio(AUDIO_DEVICE, SAMPLE_RATE, BUFFER_SIZE);

    if (!audio.initialize()) {
        std::cerr << "FATAL: Failed to initialize audio capture" << std::endl;
        std::cerr << "Try running 'arecord -l' to list available audio devices" << std::endl;
        return 1;
    }

    // ========================================================================
    // Initialize FFT Analyzer
    // ========================================================================
    std::cout << "Initializing FFT analyzer..." << std::endl;
    FFTAnalyzer fft(BUFFER_SIZE, SAMPLE_RATE);

    // ========================================================================
    // Initialize Servo Controllers
    // ========================================================================
    std::cout << "Initializing " << SERVO_PINS.size() << " servos..." << std::endl;
    MultiServoController servos(SERVO_PINS, MIN_PULSE, MAX_PULSE);

    if (!servos.initialize()) {
        std::cerr << "FATAL: Failed to initialize servos" << std::endl;
        std::cerr << "Make sure pigpiod is running: sudo systemctl start pigpiod" << std::endl;
        return 1;
    }

    // Move all servos to center position
    std::cout << "Moving servos to center position..." << std::endl;
    servos.setAllAngles(90.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // ========================================================================
    // Main Loop
    // ========================================================================
    std::cout << "\nStarting audio reactive control..." << std::endl;
    std::cout << "Press Ctrl+C to stop\n" << std::endl;

    // Variables for smoothing
    float smoothedBass = 0.0f;
    float smoothedMid = 0.0f;
    float smoothedTreble = 0.0f;

    // Servo angle targets
    std::vector<float> targetAngles(SERVO_PINS.size(), 90.0f);

    // Main control loop
    while (running) {
        // Step 1: Capture audio samples
        int samplesRead = audio.capture();
        if (samplesRead <= 0) {
            std::cerr << "\nERROR: Failed to capture audio" << std::endl;
            continue;  // Try again
        }

        // Step 2: Analyze audio with FFT
        const std::vector<float>& audioBuffer = audio.getBuffer();
        fft.analyze(audioBuffer);

        // Step 3: Get frequency band levels
        float bassLevel = fft.getBassLevel();
        float midLevel = fft.getMidLevel();
        float trebleLevel = fft.getTrebleLevel();

        // Step 4: Apply smoothing to reduce jitter
        smoothedBass = smoothValue(bassLevel, smoothedBass, SMOOTHING_FACTOR);
        smoothedMid = smoothValue(midLevel, smoothedMid, SMOOTHING_FACTOR);
        smoothedTreble = smoothValue(trebleLevel, smoothedTreble, SMOOTHING_FACTOR);

        // Step 5: Map audio levels to servo angles
        // You can customize this mapping to create different behaviors!

        // Example 1: Each servo controlled by a different frequency band
        if (SERVO_PINS.size() >= 3) {
            targetAngles[0] = mapLevelToAngle(smoothedBass, SENSITIVITY);
            targetAngles[1] = mapLevelToAngle(smoothedMid, SENSITIVITY);
            targetAngles[2] = mapLevelToAngle(smoothedTreble, SENSITIVITY);

            // Fourth servo could be controlled by overall level
            if (SERVO_PINS.size() >= 4) {
                float overallLevel = (smoothedBass + smoothedMid + smoothedTreble) / 3.0f;
                targetAngles[3] = mapLevelToAngle(overallLevel, SENSITIVITY);
            }
        }

        // Example 2: Uncomment this for alternate behavior
        // All servos respond to bass (good for dancing to the beat)
        /*
        for (size_t i = 0; i < targetAngles.size(); ++i) {
            targetAngles[i] = mapLevelToAngle(smoothedBass, SENSITIVITY * 1.5f);
        }
        */

        // Example 3: Uncomment this for wave-like motion
        /*
        for (size_t i = 0; i < targetAngles.size(); ++i) {
            float phase = (float)i / (float)targetAngles.size();
            float level = smoothedBass * sin(phase * M_PI);
            targetAngles[i] = mapLevelToAngle(level, SENSITIVITY);
        }
        */

        // Step 6: Update servo positions
        servos.setAngles(targetAngles);

        // Step 7: Print status (optional, can be commented out for less CPU usage)
        printStatus(smoothedBass, smoothedMid, smoothedTreble, targetAngles);

        // Step 8: Small delay to control update rate
        std::this_thread::sleep_for(std::chrono::milliseconds(UPDATE_RATE_MS));
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    std::cout << "\n\nShutting down..." << std::endl;

    // Return servos to center position
    std::cout << "Returning servos to center..." << std::endl;
    servos.setAllAngles(90.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Clean up resources
    std::cout << "Cleaning up resources..." << std::endl;
    servos.cleanup();
    audio.cleanup();

    std::cout << "Goodbye!" << std::endl;
    return 0;
}

