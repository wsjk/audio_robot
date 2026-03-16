/**
 * @file ServoController.cpp
 * @brief Implementation of servo motor control via GPIO
 *
 * This file implements servo control using PWM (Pulse Width Modulation).
 * Servos are controlled by sending pulses at regular intervals (50 Hz).
 * The width of each pulse determines the servo's position.
 *
 * The pigpio library handles the low-level PWM generation for us.
 */

#include "ServoController.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <pigpio.h>

/**
 * Constructor: Set up servo controller with pin and pulse range
 */
ServoController::ServoController(unsigned int gpioPin,
                                 unsigned int minPulse,
                                 unsigned int maxPulse)
    : gpioPin_(gpioPin)
    , minPulse_(minPulse)
    , maxPulse_(maxPulse)
    , currentPulse_(1500)  // Start at center position
    , initialized_(false)
    , enabled_(false)
    , pigpioHandle_(-1)
{
}

/**
 * Destructor: Clean up GPIO resources
 */
ServoController::~ServoController() {
    cleanup();
}

/**
 * Initialize the servo controller
 *
 * This connects to the pigpio daemon and sets up the GPIO pin for PWM output.
 * The pigpio daemon must be running (sudo systemctl start pigpiod).
 */
bool ServoController::initialize() {
    if (initialized_) {
        return true;
    }

    // Connect to pigpio daemon
    // pigpioHandle_ will be >= 0 on success, < 0 on error
    pigpioHandle_ = gpioInitialise();
    if (pigpioHandle_ < 0) {
        std::cerr << "ERROR: Failed to initialize pigpio (error code: "
                  << pigpioHandle_ << ")" << std::endl;
        std::cerr << "Make sure pigpiod daemon is running: sudo systemctl start pigpiod"
                  << std::endl;
        return false;
    }

    // Set the GPIO pin as an output
    int result = gpioSetMode(gpioPin_, PI_OUTPUT);
    if (result != 0) {
        std::cerr << "ERROR: Failed to set GPIO " << gpioPin_
                  << " as output (error code: " << result << ")" << std::endl;
        cleanup();
        return false;
    }

    // Set initial pulse width to center position (1500 µs = 90 degrees)
    result = gpioServo(gpioPin_, currentPulse_);
    if (result != 0) {
        std::cerr << "ERROR: Failed to set servo pulse on GPIO " << gpioPin_
                  << " (error code: " << result << ")" << std::endl;
        cleanup();
        return false;
    }

    initialized_ = true;
    enabled_ = true;

    std::cout << "Servo initialized on GPIO " << gpioPin_ << std::endl;
    std::cout << "  Pulse range: " << minPulse_ << "-" << maxPulse_ << " µs" << std::endl;
    std::cout << "  Initial position: 90° (center)" << std::endl;

    return true;
}

/**
 * Set servo to a specific angle (0-180 degrees)
 *
 * This is the easiest way to control the servo. The angle is converted
 * to a pulse width and sent to the servo.
 */
bool ServoController::setAngle(float angle) {
    if (!initialized_) {
        std::cerr << "ERROR: Servo not initialized" << std::endl;
        return false;
    }

    if (!isValidAngle(angle)) {
        std::cerr << "ERROR: Invalid angle " << angle
                  << " (must be 0-180)" << std::endl;
        return false;
    }

    // Convert angle to pulse width
    unsigned int pulseWidth = angleToPulseWidth(angle);

    // Set the pulse width
    return setPulseWidth(pulseWidth);
}

/**
 * Set servo using raw pulse width in microseconds
 *
 * For users who want precise control or whose servos have non-standard ranges.
 */
bool ServoController::setPulseWidth(unsigned int pulseWidth) {
    if (!initialized_) {
        std::cerr << "ERROR: Servo not initialized" << std::endl;
        return false;
    }

    if (!isValidPulseWidth(pulseWidth)) {
        std::cerr << "ERROR: Invalid pulse width " << pulseWidth
                  << " (must be " << minPulse_ << "-" << maxPulse_ << ")" << std::endl;
        return false;
    }

    if (!enabled_) {
        std::cerr << "WARNING: Servo is disabled" << std::endl;
        return false;
    }

    // Set the servo pulse width using pigpio
    // gpioServo() sets the PWM pulse width on the GPIO pin
    int result = gpioServo(gpioPin_, pulseWidth);
    if (result != 0) {
        std::cerr << "ERROR: Failed to set servo pulse (error code: "
                  << result << ")" << std::endl;
        return false;
    }

    currentPulse_ = pulseWidth;
    return true;
}

/**
 * Get the current servo angle
 */
float ServoController::getCurrentAngle() const {
    return pulseWidthToAngle(currentPulse_);
}

/**
 * Get the current pulse width
 */
unsigned int ServoController::getCurrentPulseWidth() const {
    return currentPulse_;
}

/**
 * Get the GPIO pin being used
 */
unsigned int ServoController::getGpioPin() const {
    return gpioPin_;
}

/**
 * Disable the servo (stops sending pulses)
 *
 * This turns off PWM, allowing the servo to move freely and reducing
 * power consumption and servo buzzing.
 */
void ServoController::disable() {
    if (initialized_ && enabled_) {
        // Setting pulse width to 0 turns off PWM
        gpioServo(gpioPin_, 0);
        enabled_ = false;
        std::cout << "Servo on GPIO " << gpioPin_ << " disabled" << std::endl;
    }
}

/**
 * Enable the servo (resumes sending pulses)
 */
void ServoController::enable() {
    if (initialized_ && !enabled_) {
        // Resume PWM with the last pulse width
        gpioServo(gpioPin_, currentPulse_);
        enabled_ = true;
        std::cout << "Servo on GPIO " << gpioPin_ << " enabled" << std::endl;
    }
}

/**
 * Check if servo is initialized
 */
bool ServoController::isInitialized() const {
    return initialized_;
}

/**
 * Smoothly move servo from current position to target angle
 *
 * This creates smooth motion by moving the servo in small steps over time.
 * Much more natural-looking than instant position changes.
 */
bool ServoController::smoothMove(float targetAngle, unsigned int durationMs,
                                 unsigned int stepsPerSecond) {
    if (!initialized_) {
        std::cerr << "ERROR: Servo not initialized" << std::endl;
        return false;
    }

    if (!isValidAngle(targetAngle)) {
        std::cerr << "ERROR: Invalid target angle " << targetAngle << std::endl;
        return false;
    }

    // Get current and target positions
    float currentAngle = getCurrentAngle();
    float totalMovement = targetAngle - currentAngle;

    // Calculate number of steps and delay between steps
    unsigned int totalSteps = (durationMs * stepsPerSecond) / 1000;
    if (totalSteps == 0) totalSteps = 1;

    unsigned int delayMs = durationMs / totalSteps;
    float angleStep = totalMovement / totalSteps;

    // Move in small increments
    for (unsigned int i = 0; i < totalSteps; ++i) {
        float newAngle = currentAngle + (angleStep * (i + 1));
        setAngle(newAngle);

        // Wait before next step
        std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
    }

    // Ensure we end exactly at target angle
    setAngle(targetAngle);

    return true;
}

/**
 * Set minimum and maximum pulse widths
 *
 * Allows calibration for different servo models.
 */
void ServoController::setPulseRange(unsigned int minPulse, unsigned int maxPulse) {
    if (minPulse >= maxPulse) {
        std::cerr << "ERROR: minPulse must be less than maxPulse" << std::endl;
        return;
    }

    minPulse_ = minPulse;
    maxPulse_ = maxPulse;

    std::cout << "Servo pulse range updated: " << minPulse_
              << "-" << maxPulse_ << " µs" << std::endl;
}

/**
 * Clean up GPIO resources
 */
void ServoController::cleanup() {
    if (initialized_) {
        // Turn off PWM
        gpioServo(gpioPin_, 0);

        // Disconnect from pigpio daemon
        gpioTerminate();

        initialized_ = false;
        enabled_ = false;
        pigpioHandle_ = -1;

        std::cout << "Servo on GPIO " << gpioPin_ << " cleaned up" << std::endl;
    }
}

/**
 * Convert angle (0-180°) to pulse width (microseconds)
 *
 * Uses linear interpolation between minPulse and maxPulse.
 *
 * Formula: pulse = minPulse + (angle / 180) * (maxPulse - minPulse)
 */
unsigned int ServoController::angleToPulseWidth(float angle) const {
    // Linear interpolation
    float ratio = angle / 180.0f;
    unsigned int pulse = minPulse_ + (unsigned int)(ratio * (maxPulse_ - minPulse_));
    return pulse;
}

/**
 * Convert pulse width to angle
 *
 * Inverse of angleToPulseWidth()
 */
float ServoController::pulseWidthToAngle(unsigned int pulseWidth) const {
    // Inverse linear interpolation
    float ratio = (float)(pulseWidth - minPulse_) / (float)(maxPulse_ - minPulse_);
    return ratio * 180.0f;
}

/**
 * Validate that an angle is within valid range (0-180)
 */
bool ServoController::isValidAngle(float angle) const {
    return (angle >= 0.0f && angle <= 180.0f);
}

/**
 * Validate that a pulse width is within configured range
 */
bool ServoController::isValidPulseWidth(unsigned int pulseWidth) const {
    return (pulseWidth >= minPulse_ && pulseWidth <= maxPulse_);
}

// ============================================================================
// MultiServoController Implementation
// ============================================================================

/**
 * Constructor: Create controller for multiple servos
 */
MultiServoController::MultiServoController(const std::vector<unsigned int>& gpioPins,
                                         unsigned int minPulse,
                                         unsigned int maxPulse)
{
    // Create a ServoController for each GPIO pin
    for (unsigned int pin : gpioPins) {
        servos_.emplace_back(pin, minPulse, maxPulse);
    }

    std::cout << "Multi-servo controller created for " << servos_.size()
              << " servos" << std::endl;
}

/**
 * Destructor
 */
MultiServoController::~MultiServoController() {
    cleanup();
}

/**
 * Initialize all servos
 */
bool MultiServoController::initialize() {
    bool allSuccess = true;

    for (size_t i = 0; i < servos_.size(); ++i) {
        std::cout << "Initializing servo " << i << "..." << std::endl;
        if (!servos_[i].initialize()) {
            std::cerr << "ERROR: Failed to initialize servo " << i << std::endl;
            allSuccess = false;
        }
    }

    return allSuccess;
}

/**
 * Set angle for a specific servo
 */
bool MultiServoController::setServoAngle(unsigned int servoIndex, float angle) {
    if (servoIndex >= servos_.size()) {
        std::cerr << "ERROR: Invalid servo index " << servoIndex << std::endl;
        return false;
    }

    return servos_[servoIndex].setAngle(angle);
}

/**
 * Set angles for all servos at once
 */
bool MultiServoController::setAngles(const std::vector<float>& angles) {
    if (angles.size() != servos_.size()) {
        std::cerr << "ERROR: Number of angles (" << angles.size()
                  << ") doesn't match number of servos (" << servos_.size() << ")"
                  << std::endl;
        return false;
    }

    bool allSuccess = true;
    for (size_t i = 0; i < servos_.size(); ++i) {
        if (!servos_[i].setAngle(angles[i])) {
            allSuccess = false;
        }
    }

    return allSuccess;
}

/**
 * Set all servos to the same angle
 */
bool MultiServoController::setAllAngles(float angle) {
    bool allSuccess = true;

    for (auto& servo : servos_) {
        if (!servo.setAngle(angle)) {
            allSuccess = false;
        }
    }

    return allSuccess;
}

/**
 * Get the number of servos being controlled
 */
unsigned int MultiServoController::getNumServos() const {
    return servos_.size();
}

/**
 * Get reference to a specific servo controller
 */
ServoController& MultiServoController::getServo(unsigned int servoIndex) {
    if (servoIndex >= servos_.size()) {
        throw std::out_of_range("Servo index out of range");
    }
    return servos_[servoIndex];
}

/**
 * Clean up all servos
 */
void MultiServoController::cleanup() {
    for (auto& servo : servos_) {
        servo.cleanup();
    }
}

