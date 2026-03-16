# Quick Start Guide

Get your audio reactive robot up and running in minutes!

## Prerequisites

- Raspberry Pi 3B with Raspberry Pi OS installed
- USB microphone
- 1-4 servo motors
- External 5V power supply for servos
- Jumper wires

## Step 1: Hardware Setup

1. **Connect USB Microphone**
   - Plug USB microphone into any USB port on Raspberry Pi

2. **Wire Servos** (⚠️ IMPORTANT)
   - **Power (Red)**: Connect to external 5V power supply (+)
   - **Ground (Black/Brown)**: Connect to external power supply (-) AND Pi GND
   - **Signal (Yellow/Orange)**: Connect to GPIO pins (default: 17, 18, 27, 22)
   
   **DO NOT power servos from Pi's 5V pins!**

3. **Common Ground**
   - Connect Pi GND to external power supply GND (critical!)

See [HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) for detailed wiring diagrams.

## Step 2: Install Dependencies

Run these commands on your Raspberry Pi:

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install build tools
sudo apt-get install -y build-essential cmake git

# Install required libraries
sudo apt-get install -y libasound2-dev libfftw3-dev pigpio python3-pigpio

# Enable and start pigpio daemon
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

## Step 3: Download Project

```bash
# Navigate to your projects directory
cd ~

# Clone or download the project
# (If you have it as a zip, extract it here)

# Navigate into project directory
cd audio-reactive-robot
```

## Step 4: Build Project

```bash
# Option 1: Use build script (recommended)
./build.sh

# Option 2: Manual build
mkdir build
cd build
cmake ..
make
```

Build takes 2-5 minutes on Raspberry Pi 3B.

## Step 5: Test Components

Test each component individually before running the main program.

### Test Audio Capture

```bash
cd build
sudo ./test_audio_capture
```

You should see audio levels changing when you make noise.

**Troubleshooting:**
- If no audio detected, run `arecord -l` to list devices
- Try changing device in code from "default" to "hw:1,0"

### Test Servos

```bash
sudo ./test_servo
```

Servos should move through various patterns.

**Troubleshooting:**
- If servos don't move, check pigpiod: `sudo systemctl status pigpiod`
- Verify GPIO pin numbers match your wiring
- Check power supply is connected and turned on

### Test FFT Analysis

```bash
sudo ./test_fft
```

Play some music and watch the frequency analysis!

## Step 6: Run Main Program

```bash
sudo ./audio_reactive_robot
```

Your robot should now react to audio!

- Servos move based on bass, mid, and treble frequencies
- Press Ctrl+C to stop gracefully

## Step 7: Customize Behavior

Edit `src/main.cpp` to change how servos respond to audio:

```cpp
// Line ~175: Customize servo mapping
targetAngles[0] = mapLevelToAngle(smoothedBass, SENSITIVITY);
targetAngles[1] = mapLevelToAngle(smoothedMid, SENSITIVITY);
targetAngles[2] = mapLevelToAngle(smoothedTreble, SENSITIVITY);

// Try different behaviors:
// - All servos react to bass (good for beat detection)
// - Create wave patterns
// - Add your own custom logic
```

After editing, rebuild:
```bash
cd build
make
sudo ./audio_reactive_robot
```

## Configuration Options

Edit constants in `src/main.cpp`:

```cpp
// Audio settings
const char* AUDIO_DEVICE = "default";     // Change if needed
const unsigned int SAMPLE_RATE = 44100;   // Standard quality
const unsigned int BUFFER_SIZE = 2048;    // FFT size

// Servo pins (BCM numbering)
const std::vector<unsigned int> SERVO_PINS = {17, 18, 27, 22};

// Tuning parameters
const float SMOOTHING_FACTOR = 0.3f;  // 0.0-1.0, lower = smoother
const float SENSITIVITY = 2.0f;       // Higher = more reactive
const unsigned int UPDATE_RATE_MS = 20; // Update frequency
```

## Common Issues

### "Cannot open audio device"
- Check microphone is plugged in: `arecord -l`
- Try recording: `arecord -D default -f S16_LE -r 44100 -d 5 test.wav`

### "Failed to initialize pigpio"
- Start daemon: `sudo systemctl start pigpiod`
- Check status: `sudo systemctl status pigpiod`

### Servos don't move
- Verify pigpiod is running
- Check GPIO pin numbers (BCM not physical)
- Ensure common ground between Pi and power supply
- Test with Python script in HARDWARE_SETUP.md

### Servos jitter or behave erratically
- Power supply may be insufficient (need 2-3A)
- Increase SMOOTHING_FACTOR in code
- Check all ground connections are solid

### Audio lag
- Decrease BUFFER_SIZE (e.g., 1024) for faster response
- Increase SMOOTHING_FACTOR for smoother motion

## Next Steps

1. **Experiment with Frequency Ranges**
   - Modify bass/mid/treble ranges in `FFTAnalyzer.cpp`
   - Add custom frequency bands

2. **Add More Servos**
   - Add GPIO pins to SERVO_PINS array
   - Customize behavior for each servo

3. **Create Patterns**
   - Use `smoothMove()` for choreographed motion
   - Implement beat detection
   - Add timing-based patterns

4. **Integrate LEDs**
   - Use similar GPIO control for LED strips
   - Create audio-reactive light shows

5. **Build a Robot**
   - Mount servos on a chassis
   - Create dancing robots
   - Build kinetic sculptures

## Resources

- **Full Documentation**: See `docs/` folder
  - `CPP_GUIDE.md` - C++ programming concepts
  - `HARDWARE_SETUP.md` - Detailed wiring guide
  - `API_REFERENCE.md` - Complete API documentation

- **Example Programs**: See `examples/` folder
  - Study example code for different use cases

- **Community**:
  - Raspberry Pi Forums: https://forums.raspberrypi.com/
  - Electronics Stack Exchange: https://electronics.stackexchange.com/

## Safety Reminders

- ⚠️ Never power servos from Raspberry Pi 5V pins
- ⚠️ Always use external power supply for servos
- ⚠️ Connect grounds together (Pi GND + power supply GND)
- ⚠️ Check servo voltage ratings before connecting power
- ⚠️ Disconnect power if components get hot

## Get Help

If you encounter issues:

1. Check error messages in terminal
2. Review troubleshooting sections in documentation
3. Test components individually
4. Verify all connections
5. Check logs: `dmesg | tail` for system messages

## Have Fun!

Now you have a working audio reactive robot! Experiment, customize, and create something amazing!

Share your creations and improvements with the community.

---

*Last updated: March 2026*

