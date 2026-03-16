/**
 * @file test_servo.cpp
 * @brief Test program for servo control functionality
 *
 * This program demonstrates how to use the ServoController class.
 * It moves servos through various patterns to test functionality.
 *
 * Usage: sudo ./test_servo
 *
 * WARNING: Make sure servos are properly connected with external power!
 */

#include "ServoController.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>

// GPIO pins for servos (BCM numbering)
const std::vector<unsigned int> SERVO_PINS = {17, 18, 27, 22};

// Global flag for clean shutdown
std::atomic<bool> running(true);

/**
 * Signal handler for Ctrl+C
 */
void signalHandler(int signal) {
    if (signal == SIGINT) {
        std::cout << "\n\nReceived shutdown signal..." << std::endl;
        running = false;
    }
}

/**
 * Test 1: Move servos to specific positions
 */
void testPositions(MultiServoController& servos) {
    std::cout << "\n--- Test 1: Position Control ---" << std::endl;

    std::cout << "Moving to 0 degrees..." << std::endl;
    servos.setAllAngles(0.0f);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "Moving to 90 degrees (center)..." << std::endl;
    servos.setAllAngles(90.0f);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "Moving to 180 degrees..." << std::endl;
    servos.setAllAngles(180.0f);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    std::cout << "Returning to center..." << std::endl;
    servos.setAllAngles(90.0f);
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

/**
 * Test 2: Sweep pattern
 */
void testSweep(MultiServoController& servos) {
    std::cout << "\n--- Test 2: Sweep Pattern ---" << std::endl;

    const int steps = 90;
    const int delayMs = 20;

    // Sweep from 0 to 180
    std::cout << "Sweeping 0 -> 180 degrees..." << std::endl;
    for (int i = 0; i <= steps; ++i) {
        if (!running) break;
        float angle = (float)i * 180.0f / steps;
        servos.setAllAngles(angle);
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    }

    // Sweep from 180 to 0
    std::cout << "Sweeping 180 -> 0 degrees..." << std::endl;
    for (int i = steps; i >= 0; --i) {
        if (!running) break;
        float angle = (float)i * 180.0f / steps;
        servos.setAllAngles(angle);
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    }

    std::cout << "Returning to center..." << std::endl;
    servos.setAllAngles(90.0f);
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

/**
 * Test 3: Wave pattern (each servo offset in phase)
 */
void testWave(MultiServoController& servos) {
    std::cout << "\n--- Test 3: Wave Pattern ---" << std::endl;

    const int duration = 5;  // seconds
    const int stepsPerSecond = 50;
    const int totalSteps = duration * stepsPerSecond;

    std::cout << "Creating wave pattern for " << duration << " seconds..." << std::endl;

    for (int step = 0; step < totalSteps; ++step) {
        if (!running) break;

        std::vector<float> angles(servos.getNumServos());

        for (unsigned int i = 0; i < servos.getNumServos(); ++i) {
            // Create sinusoidal wave with phase offset for each servo
            float time = (float)step / stepsPerSecond;
            float phase = (float)i / servos.getNumServos() * 2.0f * M_PI;
            float angle = 90.0f + 45.0f * sin(2.0f * M_PI * time + phase);
            angles[i] = angle;
        }

        servos.setAngles(angles);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / stepsPerSecond));
    }

    std::cout << "Returning to center..." << std::endl;
    servos.setAllAngles(90.0f);
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

/**
 * Test 4: Individual servo control
 */
void testIndividual(MultiServoController& servos) {
    std::cout << "\n--- Test 4: Individual Servo Control ---" << std::endl;

    for (unsigned int i = 0; i < servos.getNumServos(); ++i) {
        if (!running) break;

        std::cout << "Moving servo " << i << " (GPIO "
                  << servos.getServo(i).getGpioPin() << ")..." << std::endl;

        // Move this servo while others stay at center
        servos.setServoAngle(i, 0.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        servos.setServoAngle(i, 180.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        servos.setServoAngle(i, 90.0f);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

/**
 * Test 5: Smooth movement
 */
void testSmoothMove(MultiServoController& servos) {
    std::cout << "\n--- Test 5: Smooth Movement ---" << std::endl;

    std::cout << "Smoothly moving all servos 90 -> 0 degrees (2 seconds)..." << std::endl;
    for (unsigned int i = 0; i < servos.getNumServos(); ++i) {
        servos.getServo(i).smoothMove(0.0f, 2000);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "Smoothly moving all servos 0 -> 180 degrees (3 seconds)..." << std::endl;
    for (unsigned int i = 0; i < servos.getNumServos(); ++i) {
        servos.getServo(i).smoothMove(180.0f, 3000);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    std::cout << "Returning to center..." << std::endl;
    for (unsigned int i = 0; i < servos.getNumServos(); ++i) {
        servos.getServo(i).smoothMove(90.0f, 1500);
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "        Servo Control Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Set up signal handler
    signal(SIGINT, signalHandler);

    // Initialize servo controller
    std::cout << "Initializing " << SERVO_PINS.size() << " servos..." << std::endl;
    std::cout << "GPIO Pins (BCM): ";
    for (unsigned int pin : SERVO_PINS) {
        std::cout << pin << " ";
    }
    std::cout << std::endl << std::endl;

    MultiServoController servos(SERVO_PINS);

    if (!servos.initialize()) {
        std::cerr << "FATAL: Failed to initialize servos" << std::endl;
        std::cerr << "\nTroubleshooting tips:" << std::endl;
        std::cerr << "1. Make sure pigpiod daemon is running:" << std::endl;
        std::cerr << "   sudo systemctl start pigpiod" << std::endl;
        std::cerr << "2. Run with sudo" << std::endl;
        std::cerr << "3. Check that servos are properly connected" << std::endl;
        std::cerr << "4. Verify GPIO pin numbers are correct" << std::endl;
        return 1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "WARNING: Make sure servos are properly" << std::endl;
    std::cout << "connected with external power supply!" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "\nPress Ctrl+C to stop at any time\n" << std::endl;

    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Run tests
    if (running) testPositions(servos);
    if (running) testSweep(servos);
    if (running) testWave(servos);
    if (running) testIndividual(servos);
    if (running) testSmoothMove(servos);

    // Cleanup
    std::cout << "\n\nTests complete!" << std::endl;
    std::cout << "Returning servos to center and cleaning up..." << std::endl;
    servos.setAllAngles(90.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    servos.cleanup();

    std::cout << "Done!" << std::endl;
    return 0;
}

