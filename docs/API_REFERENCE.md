# API Reference

Complete reference for all classes and methods in the Audio Reactive Robot library.

---

## AudioCapture Class

**Header**: `AudioCapture.h`

Captures audio from USB microphone using ALSA library.

### Constructor

```cpp
AudioCapture(const std::string& deviceName, 
             unsigned int sampleRate, 
             unsigned int bufferSize)
```

**Parameters**:
- `deviceName`: ALSA device name (e.g., "default", "hw:1,0")
- `sampleRate`: Sample rate in Hz (typical: 44100 or 48000)
- `bufferSize`: Number of samples per buffer (power of 2 recommended)

**Example**:
```cpp
AudioCapture audio("default", 44100, 2048);
```

### Methods

#### initialize()

```cpp
bool initialize()
```

Opens and configures the audio device.

**Returns**: `true` on success, `false` on failure

**Example**:
```cpp
if (!audio.initialize()) {
    std::cerr << "Failed to initialize audio" << std::endl;
    return 1;
}
```

#### capture()

```cpp
int capture()
```

Captures audio samples from the microphone (blocking call).

**Returns**: Number of samples captured, or negative value on error

**Example**:
```cpp
int samplesRead = audio.capture();
if (samplesRead > 0) {
    // Process audio
}
```

#### getBuffer()

```cpp
const std::vector<float>& getBuffer() const
```

Returns the most recent audio samples as floating-point values (-1.0 to 1.0).

**Returns**: Const reference to audio buffer

**Example**:
```cpp
const std::vector<float>& buffer = audio.getBuffer();
for (float sample : buffer) {
    // Process each sample
}
```

#### getSampleRate()

```cpp
unsigned int getSampleRate() const
```

**Returns**: Current sample rate in Hz

#### getBufferSize()

```cpp
unsigned int getBufferSize() const
```

**Returns**: Buffer size in samples

#### isInitialized()

```cpp
bool isInitialized() const
```

**Returns**: `true` if device is initialized and ready

#### cleanup()

```cpp
void cleanup()
```

Closes the audio device and frees resources.

**Example**:
```cpp
audio.cleanup();
```

---

## FFTAnalyzer Class

**Header**: `FFTAnalyzer.h`

Performs Fast Fourier Transform analysis on audio data.

### Constructor

```cpp
FFTAnalyzer(unsigned int fftSize, unsigned int sampleRate)
```

**Parameters**:
- `fftSize`: Size of FFT (should match audio buffer size, power of 2)
- `sampleRate`: Audio sample rate in Hz

**Example**:
```cpp
FFTAnalyzer fft(2048, 44100);
```

### Methods

#### analyze()

```cpp
void analyze(const std::vector<float>& audioBuffer)
```

Performs FFT analysis on audio buffer.

**Parameters**:
- `audioBuffer`: Vector of audio samples (size must match fftSize)

**Example**:
```cpp
const std::vector<float>& buffer = audio.getBuffer();
fft.analyze(buffer);
```

#### getBassLevel()

```cpp
float getBassLevel() const
```

Gets bass frequency level (20-250 Hz).

**Returns**: Bass level from 0.0 (silent) to 1.0 (maximum)

**Example**:
```cpp
float bass = fft.getBassLevel();
if (bass > 0.5f) {
    std::cout << "Strong bass detected!" << std::endl;
}
```

#### getMidLevel()

```cpp
float getMidLevel() const
```

Gets mid frequency level (250-2000 Hz).

**Returns**: Mid level from 0.0 to 1.0

#### getTrebleLevel()

```cpp
float getTrebleLevel() const
```

Gets treble frequency level (2000-20000 Hz).

**Returns**: Treble level from 0.0 to 1.0

#### getMagnitudeAtFrequency()

```cpp
float getMagnitudeAtFrequency(float frequency) const
```

Gets magnitude at a specific frequency.

**Parameters**:
- `frequency`: Frequency in Hz to check

**Returns**: Magnitude at that frequency (0.0 to 1.0)

**Example**:
```cpp
// Check for 440 Hz tone (A4 note)
float a4 = fft.getMagnitudeAtFrequency(440.0f);
```

#### getFrequencyRangeLevel()

```cpp
float getFrequencyRangeLevel(float minFreq, float maxFreq) const
```

Gets average magnitude in a custom frequency range.

**Parameters**:
- `minFreq`: Minimum frequency in Hz
- `maxFreq`: Maximum frequency in Hz

**Returns**: Average magnitude in range (0.0 to 1.0)

**Example**:
```cpp
// Get level in human voice range
float voice = fft.getFrequencyRangeLevel(300.0f, 3000.0f);
```

#### getMagnitudeSpectrum()

```cpp
const std::vector<float>& getMagnitudeSpectrum() const
```

Gets the complete FFT magnitude spectrum.

**Returns**: Const reference to magnitude spectrum

**Example**:
```cpp
const std::vector<float>& spectrum = fft.getMagnitudeSpectrum();
for (size_t i = 0; i < spectrum.size(); ++i) {
    float freq = fft.binToFrequency(i);
    std::cout << freq << " Hz: " << spectrum[i] << std::endl;
}
```

#### binToFrequency()

```cpp
float binToFrequency(unsigned int bin) const
```

Converts FFT bin index to frequency in Hz.

**Parameters**:
- `bin`: FFT bin index

**Returns**: Frequency in Hz

#### frequencyToBin()

```cpp
unsigned int frequencyToBin(float frequency) const
```

Converts frequency in Hz to FFT bin index.

**Parameters**:
- `frequency`: Frequency in Hz

**Returns**: FFT bin index

---

## ServoController Class

**Header**: `ServoController.h`

Controls a single servo motor via GPIO.

### Constructor

```cpp
ServoController(unsigned int gpioPin, 
                unsigned int minPulse = 500, 
                unsigned int maxPulse = 2500)
```

**Parameters**:
- `gpioPin`: GPIO pin number (BCM numbering)
- `minPulse`: Minimum pulse width in microseconds (default: 500)
- `maxPulse`: Maximum pulse width in microseconds (default: 2500)

**Example**:
```cpp
ServoController servo(17);  // GPIO 17
```

### Methods

#### initialize()

```cpp
bool initialize()
```

Initializes the servo controller and connects to pigpio daemon.

**Returns**: `true` on success, `false` on failure

**Example**:
```cpp
if (!servo.initialize()) {
    std::cerr << "Failed to initialize servo" << std::endl;
    return 1;
}
```

#### setAngle()

```cpp
bool setAngle(float angle)
```

Sets servo to a specific angle.

**Parameters**:
- `angle`: Desired angle in degrees (0-180)

**Returns**: `true` on success, `false` on failure

**Example**:
```cpp
servo.setAngle(90.0f);   // Center position
servo.setAngle(0.0f);    // Minimum position
servo.setAngle(180.0f);  // Maximum position
```

#### setPulseWidth()

```cpp
bool setPulseWidth(unsigned int pulseWidth)
```

Sets servo using raw pulse width.

**Parameters**:
- `pulseWidth`: Pulse width in microseconds (typically 500-2500)

**Returns**: `true` on success, `false` on failure

**Example**:
```cpp
servo.setPulseWidth(1500);  // Center (90 degrees)
```

#### getCurrentAngle()

```cpp
float getCurrentAngle() const
```

**Returns**: Current servo angle in degrees

#### getCurrentPulseWidth()

```cpp
unsigned int getCurrentPulseWidth() const
```

**Returns**: Current pulse width in microseconds

#### getGpioPin()

```cpp
unsigned int getGpioPin() const
```

**Returns**: GPIO pin number being used

#### disable()

```cpp
void disable()
```

Disables the servo (stops sending pulses).

**Example**:
```cpp
servo.disable();  // Servo can now move freely
```

#### enable()

```cpp
void enable()
```

Re-enables the servo after disable().

#### isInitialized()

```cpp
bool isInitialized() const
```

**Returns**: `true` if servo is initialized

#### smoothMove()

```cpp
bool smoothMove(float targetAngle, 
                unsigned int durationMs, 
                unsigned int stepsPerSecond = 50)
```

Smoothly moves servo from current position to target angle.

**Parameters**:
- `targetAngle`: Target angle in degrees (0-180)
- `durationMs`: Duration of movement in milliseconds
- `stepsPerSecond`: Update rate (default: 50 Hz)

**Returns**: `true` on success

**Example**:
```cpp
// Move from current position to 180 degrees over 2 seconds
servo.smoothMove(180.0f, 2000);
```

#### setPulseRange()

```cpp
void setPulseRange(unsigned int minPulse, unsigned int maxPulse)
```

Sets custom pulse width range for servo calibration.

**Parameters**:
- `minPulse`: Minimum pulse width in microseconds
- `maxPulse`: Maximum pulse width in microseconds

**Example**:
```cpp
servo.setPulseRange(1000, 2000);  // For servos with 1-2ms range
```

#### cleanup()

```cpp
void cleanup()
```

Releases GPIO resources and disconnects from pigpio.

---

## MultiServoController Class

**Header**: `ServoController.h`

Controls multiple servos simultaneously.

### Constructor

```cpp
MultiServoController(const std::vector<unsigned int>& gpioPins,
                    unsigned int minPulse = 500,
                    unsigned int maxPulse = 2500)
```

**Parameters**:
- `gpioPins`: Vector of GPIO pin numbers
- `minPulse`: Minimum pulse width for all servos (default: 500)
- `maxPulse`: Maximum pulse width for all servos (default: 2500)

**Example**:
```cpp
std::vector<unsigned int> pins = {17, 18, 27, 22};
MultiServoController servos(pins);
```

### Methods

#### initialize()

```cpp
bool initialize()
```

Initializes all servos.

**Returns**: `true` if all servos initialized successfully

#### setServoAngle()

```cpp
bool setServoAngle(unsigned int servoIndex, float angle)
```

Sets angle for a specific servo.

**Parameters**:
- `servoIndex`: Index of servo (0 to numServos-1)
- `angle`: Desired angle in degrees (0-180)

**Returns**: `true` on success

**Example**:
```cpp
servos.setServoAngle(0, 45.0f);   // First servo to 45 degrees
servos.setServoAngle(1, 135.0f);  // Second servo to 135 degrees
```

#### setAngles()

```cpp
bool setAngles(const std::vector<float>& angles)
```

Sets angles for all servos at once.

**Parameters**:
- `angles`: Vector of angles (one per servo)

**Returns**: `true` on success

**Example**:
```cpp
std::vector<float> angles = {0.0f, 45.0f, 90.0f, 135.0f};
servos.setAngles(angles);
```

#### setAllAngles()

```cpp
bool setAllAngles(float angle)
```

Sets all servos to the same angle.

**Parameters**:
- `angle`: Angle for all servos (0-180)

**Returns**: `true` on success

**Example**:
```cpp
servos.setAllAngles(90.0f);  // All servos to center
```

#### getNumServos()

```cpp
unsigned int getNumServos() const
```

**Returns**: Number of servos being controlled

#### getServo()

```cpp
ServoController& getServo(unsigned int servoIndex)
```

Gets reference to a specific servo controller for advanced control.

**Parameters**:
- `servoIndex`: Index of servo (0 to numServos-1)

**Returns**: Reference to ServoController

**Example**:
```cpp
// Access individual servo for smooth move
servos.getServo(0).smoothMove(180.0f, 2000);
```

#### cleanup()

```cpp
void cleanup()
```

Cleans up all servos.

---

## Usage Examples

### Complete Audio Analysis Example

```cpp
#include "AudioCapture.h"
#include "FFTAnalyzer.h"
#include <iostream>

int main() {
    // Initialize audio
    AudioCapture audio("default", 44100, 2048);
    if (!audio.initialize()) {
        return 1;
    }
    
    // Initialize FFT
    FFTAnalyzer fft(2048, 44100);
    
    // Main loop
    for (int i = 0; i < 100; ++i) {
        // Capture audio
        audio.capture();
        const auto& buffer = audio.getBuffer();
        
        // Analyze
        fft.analyze(buffer);
        
        // Get results
        float bass = fft.getBassLevel();
        float mid = fft.getMidLevel();
        float treble = fft.getTrebleLevel();
        
        std::cout << "Bass: " << bass 
                  << " Mid: " << mid 
                  << " Treble: " << treble << std::endl;
    }
    
    return 0;
}
```

### Complete Servo Control Example

```cpp
#include "ServoController.h"
#include <vector>

int main() {
    // Initialize servos
    std::vector<unsigned int> pins = {17, 18, 27, 22};
    MultiServoController servos(pins);
    
    if (!servos.initialize()) {
        return 1;
    }
    
    // Move all to center
    servos.setAllAngles(90.0f);
    
    // Control individual servos
    servos.setServoAngle(0, 0.0f);
    servos.setServoAngle(1, 45.0f);
    servos.setServoAngle(2, 90.0f);
    servos.setServoAngle(3, 135.0f);
    
    // Cleanup
    servos.cleanup();
    
    return 0;
}
```

### Integrated Audio Reactive Example

```cpp
#include "AudioCapture.h"
#include "FFTAnalyzer.h"
#include "ServoController.h"

int main() {
    // Initialize all components
    AudioCapture audio("default", 44100, 2048);
    FFTAnalyzer fft(2048, 44100);
    std::vector<unsigned int> pins = {17, 18, 27};
    MultiServoController servos(pins);
    
    if (!audio.initialize() || !servos.initialize()) {
        return 1;
    }
    
    // Main loop
    while (true) {
        // Capture and analyze audio
        audio.capture();
        fft.analyze(audio.getBuffer());
        
        // Map frequency levels to servo angles
        float bassAngle = fft.getBassLevel() * 180.0f;
        float midAngle = fft.getMidLevel() * 180.0f;
        float trebleAngle = fft.getTrebleLevel() * 180.0f;
        
        // Update servos
        servos.setServoAngle(0, bassAngle);
        servos.setServoAngle(1, midAngle);
        servos.setServoAngle(2, trebleAngle);
    }
    
    return 0;
}
```

---

## Error Handling

All initialization methods return `bool` indicating success/failure:

```cpp
if (!audio.initialize()) {
    std::cerr << "ERROR: Audio initialization failed" << std::endl;
    // Check error messages printed to stderr
    return 1;
}
```

Methods print detailed error messages to `std::cerr` for debugging.

---

## Thread Safety

**Note**: These classes are NOT thread-safe. If using multiple threads:

1. Create separate objects per thread, OR
2. Use mutexes to protect shared access

**Example with mutex**:
```cpp
#include <mutex>

std::mutex audioMutex;
AudioCapture audio("default", 44100, 2048);

// Thread 1
{
    std::lock_guard<std::mutex> lock(audioMutex);
    audio.capture();
}

// Thread 2
{
    std::lock_guard<std::mutex> lock(audioMutex);
    const auto& buffer = audio.getBuffer();
}
```

---

## Performance Considerations

- **FFT Size**: Larger = better frequency resolution but more CPU
  - 1024: Low CPU, ~43 Hz resolution
  - 2048: Balanced, ~21 Hz resolution
  - 4096: High CPU, ~11 Hz resolution

- **Update Rate**: Servo update rate affects CPU usage
  - 50 Hz (20ms): Smooth, standard
  - 100 Hz (10ms): Very smooth, higher CPU
  - 20 Hz (50ms): Laggy but low CPU

- **Buffer Size**: Should match FFT size for efficiency

---

## License

MIT License - Free to use in your projects

