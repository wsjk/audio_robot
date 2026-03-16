# Audio Reactive Robot for Raspberry Pi 3B

A C++ project that creates an audio-reactive robot using USB microphone input, FFT audio analysis, and servo control via GPIO on Raspberry Pi 3B.

## Features

- **Audio Capture**: Captures audio from USB microphone using ALSA (Advanced Linux Sound Architecture)
- **FFT Analysis**: Performs Fast Fourier Transform to analyze frequency spectrum
- **Servo Control**: Controls servos via GPIO pins based on audio frequencies
- **Modular Design**: Reusable libraries for other projects

## Hardware Requirements

- Raspberry Pi 3B
- USB Microphone
- Servo motors (compatible with 3.3V/5V GPIO signals)
- External power supply for servos (recommended)
- Jumper wires

## Software Requirements

- Raspberry Pi OS (32-bit or 64-bit)
- C++ compiler (g++ 7.0 or later)
- CMake 3.10 or later
- ALSA development libraries
- FFTW3 library
- pigpio library for GPIO control

## Installation

### 1. Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git
sudo apt-get install -y libasound2-dev libfftw3-dev
sudo apt-get install -y pigpio python3-pigpio
```

### 2. Enable pigpio daemon

```bash
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

### 3. Build the Project

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

### Basic Example

```bash
# Run the main audio reactive program
sudo ./audio_reactive_robot

# Run with custom configuration
sudo ./audio_reactive_robot --device hw:1,0 --sample-rate 44100
```

### Running Examples

```bash
# Test audio capture
sudo ./examples/test_audio_capture

# Test FFT analysis
sudo ./examples/test_fft

# Test servo control
sudo ./examples/test_servo
```

## Project Structure

```
audio-reactive-robot/
├── src/                    # Source files
│   ├── main.cpp           # Main application
│   ├── AudioCapture.cpp   # Audio input handling
│   ├── FFTAnalyzer.cpp    # FFT processing
│   └── ServoController.cpp # Servo control
├── include/               # Header files
│   ├── AudioCapture.h
│   ├── FFTAnalyzer.h
│   └── ServoController.h
├── examples/              # Example programs
├── docs/                  # Documentation
└── CMakeLists.txt        # Build configuration
```

## Architecture

### AudioCapture Class
Handles capturing audio from USB microphone using ALSA library.

**Key Methods:**
- `initialize()`: Sets up audio device
- `capture()`: Reads audio samples
- `getBuffer()`: Returns audio buffer

### FFTAnalyzer Class
Performs Fast Fourier Transform on audio data to extract frequency information.

**Key Methods:**
- `analyze()`: Performs FFT on audio buffer
- `getBassLevel()`: Gets bass frequency magnitude (20-250 Hz)
- `getMidLevel()`: Gets mid frequency magnitude (250-2000 Hz)
- `getTrebleLevel()`: Gets treble frequency magnitude (2000-20000 Hz)

### ServoController Class
Controls servo motors via GPIO pins using pigpio library.

**Key Methods:**
- `initialize()`: Sets up GPIO pins
- `setAngle()`: Sets servo to specific angle (0-180 degrees)
- `setPulseWidth()`: Sets raw PWM pulse width

## Hardware Connections

### Servo Wiring

| Servo Wire | Connection |
|------------|------------|
| Red (Power)| External 5V Power Supply |
| Brown/Black (GND) | GND (shared with Pi) |
| Orange (Signal) | GPIO Pin (configurable) |

**Default GPIO Pins:**
- Servo 1: GPIO 17
- Servo 2: GPIO 18
- Servo 3: GPIO 27
- Servo 4: GPIO 22

**Important**: Always use an external power supply for servos. Do not power servos directly from Raspberry Pi's 5V pin.

## Configuration

Edit the configuration in `src/main.cpp` or pass command-line arguments:

```cpp
// Audio settings
const char* AUDIO_DEVICE = "default";  // USB mic device
const int SAMPLE_RATE = 44100;         // Samples per second
const int BUFFER_SIZE = 2048;          // FFT buffer size

// Servo pins (GPIO BCM numbering)
const int SERVO_PINS[] = {17, 18, 27, 22};
```

## API Reference

### AudioCapture

```cpp
// Create audio capture object
AudioCapture audio(DEVICE_NAME, SAMPLE_RATE, BUFFER_SIZE);

// Initialize the device
if (!audio.initialize()) {
    // Handle error
}

// Capture audio samples
int samples = audio.capture();

// Get the audio buffer
const std::vector<float>& buffer = audio.getBuffer();
```

### FFTAnalyzer

```cpp
// Create FFT analyzer
FFTAnalyzer fft(BUFFER_SIZE, SAMPLE_RATE);

// Analyze audio buffer
fft.analyze(audioBuffer);

// Get frequency band levels (0.0 to 1.0)
float bass = fft.getBassLevel();
float mid = fft.getMidLevel();
float treble = fft.getTrebleLevel();
```

### ServoController

```cpp
// Create servo controller
ServoController servo(GPIO_PIN);

// Initialize the servo
if (!servo.initialize()) {
    // Handle error
}

// Set servo angle (0-180 degrees)
servo.setAngle(90);

// Or set raw pulse width (500-2500 microseconds)
servo.setPulseWidth(1500);

// Clean up
servo.cleanup();
```

## Troubleshooting

### USB Microphone Not Detected

```bash
# List audio devices
arecord -l

# Test recording
arecord -D hw:1,0 -f S16_LE -r 44100 -d 5 test.wav
```

### Permission Denied for GPIO

Run with sudo or add user to gpio group:
```bash
sudo usermod -a -G gpio $USER
```

### pigpio Daemon Not Running

```bash
sudo systemctl status pigpiod
sudo systemctl start pigpiod
```

## License

MIT License - Feel free to use in your projects

## Contributing

Contributions welcome! Please submit pull requests or open issues.

## Future Enhancements

- [ ] Web interface for configuration
- [ ] Multiple frequency band detection
- [ ] Smooth servo transitions
- [ ] Beat detection
- [ ] Pattern recording and playback
- [ ] Support for other GPIO libraries

## Credits

- ALSA Library: https://www.alsa-project.org/
- FFTW3: http://www.fftw.org/
- pigpio: http://abyz.me.uk/rpi/pigpio/

