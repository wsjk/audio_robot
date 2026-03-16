/**
 * @file ServoController.h
 * @brief Header file for controlling servo motors via GPIO
 *
 * This class provides an interface to control servo motors connected
 * to the Raspberry Pi's GPIO pins using the pigpio library.
 *
 * Key Concepts for Beginners:
 * - Servo Motor: A motor that can rotate to a specific angle (typically 0-180°)
 * - GPIO: General Purpose Input/Output pins on the Raspberry Pi
 * - PWM: Pulse Width Modulation - a technique to control servos
 * - Pulse Width: Duration of the "high" signal in microseconds
 *   - ~500-1000 µs = 0 degrees
 *   - ~1500 µs = 90 degrees (center)
 *   - ~2000-2500 µs = 180 degrees
 * - BCM Numbering: GPIO pin numbering system (GPIO 17 is different from physical pin 17)
 *
 * Important Safety Notes:
 * - Always use an external power supply for servos (they draw too much current for Pi)
 * - Connect grounds together (Pi GND and servo power supply GND)
 * - Only connect servo signal wires to GPIO pins
 * - Check your servo's specs for correct pulse width range
 *
 * @author Audio Reactive Robot Project
 * @date 2026
 */

#ifndef SERVO_CONTROLLER_H
#define SERVO_CONTROLLER_H

#include <vector>
#include <string>

/**
 * @class ServoController
 * @brief Controls a servo motor connected to a GPIO pin
 *
 * This class uses the pigpio library to generate PWM signals that control
 * servo position. Servos are controlled by sending pulses of specific widths
 * at regular intervals (typically 50 Hz / every 20ms).
 *
 * Example usage:
 * @code
 *   // Create controller for GPIO pin 17
 *   ServoController servo(17);
 *
 *   // Initialize the servo
 *   if (servo.initialize()) {
 *       // Move servo to center position (90 degrees)
 *       servo.setAngle(90);
 *
 *       // Or use raw pulse width (1500 microseconds = center)
 *       servo.setPulseWidth(1500);
 *
 *       // Cleanup when done
 *       servo.cleanup();
 *   }
 * @endcode
 */
class ServoController {
public:
    /**
     * @brief Constructor for ServoController
     *
     * @param gpioPin GPIO pin number (BCM numbering, e.g., 17, 18, 27)
     *                See pinout.xyz for Raspberry Pi pin layout
     * @param minPulse Minimum pulse width in microseconds (default: 500)
     *                 Corresponds to 0 degrees
     * @param maxPulse Maximum pulse width in microseconds (default: 2500)
     *                 Corresponds to 180 degrees
     *
     * Note: Different servos may have different pulse width ranges.
     * Common ranges: 500-2500, 1000-2000. Check your servo's datasheet.
     */
    ServoController(unsigned int gpioPin,
                   unsigned int minPulse = 500,
                   unsigned int maxPulse = 2500);

    /**
     * @brief Destructor - cleans up GPIO resources
     */
    ~ServoController();

    /**
     * @brief Initialize the servo controller
     *
     * This connects to the pigpio daemon and sets up the GPIO pin
     * for PWM output. Must be called before controlling the servo.
     *
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Set servo to a specific angle
     *
     * This is the easiest way to control the servo. Just specify
     * an angle and the servo will move to that position.
     *
     * @param angle Desired angle in degrees (0 to 180)
     *              0 = fully counter-clockwise
     *              90 = center position
     *              180 = fully clockwise
     * @return true if successful, false if angle out of range
     */
    bool setAngle(float angle);

    /**
     * @brief Set servo using raw pulse width
     *
     * For advanced users who want precise control. Sets the exact
     * pulse width in microseconds.
     *
     * @param pulseWidth Pulse width in microseconds (typically 500-2500)
     * @return true if successful, false if pulse width out of range
     */
    bool setPulseWidth(unsigned int pulseWidth);

    /**
     * @brief Get the current servo angle
     *
     * @return Current angle in degrees (0-180)
     */
    float getCurrentAngle() const;

    /**
     * @brief Get the current pulse width
     *
     * @return Current pulse width in microseconds
     */
    unsigned int getCurrentPulseWidth() const;

    /**
     * @brief Get the GPIO pin being used
     *
     * @return GPIO pin number (BCM numbering)
     */
    unsigned int getGpioPin() const;

    /**
     * @brief Disable the servo (stops sending pulses)
     *
     * This turns off PWM output, allowing the servo to move freely.
     * Useful to save power or reduce servo buzzing when not in use.
     */
    void disable();

    /**
     * @brief Enable the servo (resumes sending pulses)
     *
     * Re-enables PWM output after a disable() call.
     */
    void enable();

    /**
     * @brief Check if servo is initialized and ready
     *
     * @return true if ready to control servo
     */
    bool isInitialized() const;

    /**
     * @brief Smoothly move servo from current position to target angle
     *
     * Instead of jumping to the target angle instantly, this moves
     * the servo gradually over a specified duration. This creates
     * smoother, more natural motion.
     *
     * @param targetAngle Desired final angle in degrees (0-180)
     * @param durationMs How long the movement should take in milliseconds
     * @param stepsPerSecond Number of position updates per second (default: 50)
     * @return true if successful
     */
    bool smoothMove(float targetAngle, unsigned int durationMs, unsigned int stepsPerSecond = 50);

    /**
     * @brief Set minimum and maximum pulse widths
     *
     * Allows you to calibrate the servo's range if the defaults don't work.
     * Some servos have different ranges like 1000-2000 µs.
     *
     * @param minPulse Minimum pulse width in microseconds
     * @param maxPulse Maximum pulse width in microseconds
     */
    void setPulseRange(unsigned int minPulse, unsigned int maxPulse);

    /**
     * @brief Clean up and release GPIO resources
     *
     * Call this when you're done using the servo to properly
     * shut down the GPIO pin.
     */
    void cleanup();

private:
    unsigned int gpioPin_;          ///< GPIO pin number (BCM)
    unsigned int minPulse_;         ///< Minimum pulse width in µs (0°)
    unsigned int maxPulse_;         ///< Maximum pulse width in µs (180°)
    unsigned int currentPulse_;     ///< Current pulse width in µs
    bool initialized_;              ///< Whether GPIO is initialized
    bool enabled_;                  ///< Whether servo is enabled
    int pigpioHandle_;              ///< Handle to pigpio daemon connection

    /**
     * @brief Convert angle (0-180°) to pulse width (microseconds)
     *
     * Uses linear interpolation between minPulse and maxPulse.
     *
     * @param angle Angle in degrees
     * @return Pulse width in microseconds
     */
    unsigned int angleToPulseWidth(float angle) const;

    /**
     * @brief Convert pulse width to angle
     *
     * Inverse of angleToPulseWidth().
     *
     * @param pulseWidth Pulse width in microseconds
     * @return Angle in degrees
     */
    float pulseWidthToAngle(unsigned int pulseWidth) const;

    /**
     * @brief Validate that an angle is within valid range
     *
     * @param angle Angle to check
     * @return true if angle is between 0 and 180
     */
    bool isValidAngle(float angle) const;

    /**
     * @brief Validate that a pulse width is within configured range
     *
     * @param pulseWidth Pulse width to check
     * @return true if pulse width is between minPulse and maxPulse
     */
    bool isValidPulseWidth(unsigned int pulseWidth) const;
};

/**
 * @class MultiServoController
 * @brief Controls multiple servos simultaneously
 *
 * This is a convenience class for controlling several servos at once,
 * useful for robots with multiple degrees of freedom.
 *
 * Example usage:
 * @code
 *   // Create controller for 4 servos on different GPIO pins
 *   std::vector<unsigned int> pins = {17, 18, 27, 22};
 *   MultiServoController servos(pins);
 *
 *   if (servos.initialize()) {
 *       // Set all servos to center
 *       servos.setAllAngles(90);
 *
 *       // Set individual servo
 *       servos.setServoAngle(0, 45);  // First servo to 45°
 *
 *       // Set multiple servos at once
 *       std::vector<float> angles = {0, 45, 90, 135};
 *       servos.setAngles(angles);
 *   }
 * @endcode
 */
class MultiServoController {
public:
    /**
     * @brief Constructor for MultiServoController
     *
     * @param gpioPins Vector of GPIO pin numbers for all servos
     * @param minPulse Minimum pulse width for all servos (default: 500)
     * @param maxPulse Maximum pulse width for all servos (default: 2500)
     */
    MultiServoController(const std::vector<unsigned int>& gpioPins,
                        unsigned int minPulse = 500,
                        unsigned int maxPulse = 2500);

    /**
     * @brief Destructor
     */
    ~MultiServoController();

    /**
     * @brief Initialize all servos
     *
     * @return true if all servos initialized successfully
     */
    bool initialize();

    /**
     * @brief Set angle for a specific servo
     *
     * @param servoIndex Index of the servo (0 to numServos-1)
     * @param angle Desired angle in degrees (0-180)
     * @return true if successful
     */
    bool setServoAngle(unsigned int servoIndex, float angle);

    /**
     * @brief Set angles for all servos at once
     *
     * @param angles Vector of angles (one per servo)
     * @return true if successful
     */
    bool setAngles(const std::vector<float>& angles);

    /**
     * @brief Set all servos to the same angle
     *
     * @param angle Angle to set for all servos (0-180)
     * @return true if successful
     */
    bool setAllAngles(float angle);

    /**
     * @brief Get the number of servos being controlled
     *
     * @return Number of servos
     */
    unsigned int getNumServos() const;

    /**
     * @brief Get reference to a specific servo controller
     *
     * Allows direct access to individual servos for advanced control.
     *
     * @param servoIndex Index of the servo (0 to numServos-1)
     * @return Reference to the ServoController object
     */
    ServoController& getServo(unsigned int servoIndex);

    /**
     * @brief Clean up all servos
     */
    void cleanup();

private:
    std::vector<ServoController> servos_; ///< Vector of individual servo controllers
};

#endif // SERVO_CONTROLLER_H

