# Hardware Setup Guide

This guide helps you set up the physical hardware for your audio reactive robot.

## Components Needed

### Required Components

1. **Raspberry Pi 3B**
   - Any Raspberry Pi model will work, but this project is tested on 3B
   - MicroSD card (8GB minimum, 16GB+ recommended)
   - Power supply (5V 2.5A minimum)

2. **USB Microphone**
   - Any USB microphone compatible with Linux
   - Recommended: Blue Snowball, Fifine K669, or similar
   - Built-in laptop microphones may work but external is better

3. **Servo Motors**
   - Standard hobby servos (e.g., SG90, MG90S, MG996R)
   - Quantity: 1-4 servos
   - Voltage: 4.8V-6V typical
   - Signal: 50Hz PWM (standard)

4. **External Power Supply for Servos**
   - **CRITICAL**: Do NOT power servos from Raspberry Pi's 5V pins!
   - Recommended: 5V 2-3A power supply
   - Battery pack: 4x AA batteries (6V) with regulator
   - Must share ground with Raspberry Pi

5. **Connecting Wires**
   - Female-to-female jumper wires for servo signals
   - Wire for power connections
   - Breadboard (optional, for organization)

### Optional Components

- Servo mounting brackets
- Robot chassis or frame
- LED indicators
- Enclosure for electronics

---

## Raspberry Pi Setup

### 1. Install Raspberry Pi OS

Download and install Raspberry Pi OS (formerly Raspbian):

1. Download Raspberry Pi Imager: https://www.raspberrypi.com/software/
2. Flash Raspberry Pi OS (32-bit or 64-bit) to microSD card
3. Insert card into Pi and boot up
4. Follow initial setup wizard

### 2. Enable SSH (Optional but Recommended)

For remote access:

```bash
sudo raspi-config
# Navigate to: Interface Options -> SSH -> Enable
```

### 3. Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

---

## USB Microphone Setup

### 1. Connect Microphone

Plug USB microphone into any USB port on the Raspberry Pi.

### 2. Verify Detection

```bash
# List audio input devices
arecord -l
```

You should see output like:
```
**** List of CAPTURE Hardware Devices ****
card 1: Device [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

Note the card number (e.g., `card 1`).

### 3. Test Recording

```bash
# Record 5 seconds of audio
arecord -D hw:1,0 -f S16_LE -r 44100 -d 5 test.wav

# Play it back (if you have speakers)
aplay test.wav
```

If this works, your microphone is ready!

### 4. Set as Default (Optional)

Create/edit `~/.asoundrc`:

```bash
nano ~/.asoundrc
```

Add:
```
pcm.!default {
    type hw
    card 1
}
```

---

## Servo Wiring

### Understanding Servo Pins

Standard servo has 3 wires:

| Wire Color | Purpose | Connection |
|------------|---------|------------|
| **Red** or **Orange** | Power (VCC) | External 5V power supply (+) |
| **Brown** or **Black** | Ground (GND) | External power supply (-) AND Pi GND |
| **Yellow** or **Orange** | Signal (PWM) | Raspberry Pi GPIO pin |

### GPIO Pin Layout (BCM Numbering)

```
Raspberry Pi 3B GPIO Pinout (simplified):

        3.3V [ 1] [ 2] 5V
      GPIO 2 [ 3] [ 4] 5V
      GPIO 3 [ 5] [ 6] GND
      GPIO 4 [ 7] [ 8] GPIO 14
         GND [ 9] [10] GPIO 15
     GPIO 17 [11] [12] GPIO 18
     GPIO 27 [13] [14] GND
     GPIO 22 [15] [16] GPIO 23
        3.3V [17] [18] GPIO 24
    ...
```

**Full pinout**: Visit https://pinout.xyz/

### Wiring Diagram

```
┌─────────────────────────────────────┐
│      Raspberry Pi 3B                │
│                                     │
│  GPIO 17 (Pin 11) ──┐               │
│  GPIO 18 (Pin 12) ──┼───────────┐   │
│  GPIO 27 (Pin 13) ──┼───────┐   │   │
│  GPIO 22 (Pin 15) ──┼───┐   │   │   │
│  GND (Pin  6)  ─────┼─┐ │   │   │   │
└─────────────────────┼─┼─┼───┼───┼───┘
                      │ │ │   │   │
                      │ │ │   │   └── Servo 4 Signal (Yellow)
                      │ │ │   └────── Servo 3 Signal (Yellow)
                      │ │ └────────── Servo 2 Signal (Yellow)
                      │ └──────────── Servo 1 Signal (Yellow)
                      │
                      └── Connected to external power GND
                      
┌─────────────────────────────────────┐
│  External 5V Power Supply (2-3A)    │
│                                     │
│  (+) 5V  ───┬───┬───┬───┬──        │
│             │   │   │   │           │
│  (-) GND ───┼───┼───┼───┼── Connected to Pi GND
└─────────────┼───┼───┼───┼───────────┘
              │   │   │   │
        Red   │   │   │   │   Red
        Wire  │   │   │   │   Wire
              │   │   │   │
         ┌────┴─┐ ├───┴─┐ ├───┴─┐ ├───┴─┐
         │Servo1│ │Servo2│ │Servo3│ │Servo4│
         │ SG90 │ │ SG90 │ │ SG90 │ │ SG90 │
         └──────┘ └──────┘ └──────┘ └──────┘
```

### Step-by-Step Wiring

#### For Each Servo:

1. **Power (Red/Orange wire)**:
   - Connect to (+) terminal of external 5V power supply
   - **DO NOT** connect to Pi's 5V pin (can damage Pi!)

2. **Ground (Brown/Black wire)**:
   - Connect to (-) terminal of external power supply

3. **Signal (Yellow/Orange wire)**:
   - Connect to designated GPIO pin on Raspberry Pi
   - Default pins: GPIO 17, 18, 27, 22

#### Critical: Common Ground

The Raspberry Pi GND and power supply GND **MUST** be connected together!

```bash
# Connect one wire from Pi GND (Pin 6, 9, 14, 20, 25, 30, 34, or 39)
# to the (-) terminal of your external power supply
```

This provides a common reference voltage for PWM signals.

### Example with 2 Servos

```
Raspberry Pi:
- GPIO 17 (Pin 11) → Servo 1 Signal (Yellow)
- GPIO 18 (Pin 12) → Servo 2 Signal (Yellow)
- GND (Pin 6) → Power Supply GND

External 5V Power Supply:
- (+) 5V → Servo 1 Red wire AND Servo 2 Red wire
- (-) GND → Servo 1 Black wire AND Servo 2 Black wire AND Pi GND
```

---

## Power Supply Options

### Option 1: AC Adapter (Recommended)

- Use a 5V 2-3A AC adapter with appropriate connector
- Stable, unlimited runtime
- Best for stationary setups

### Option 2: Battery Pack

**AA Batteries (6V)**:
- 4x AA batteries = 6V (may need voltage regulator to 5V)
- Portable but limited runtime
- Use rechargeable for cost savings

**LiPo Battery (7.4V with regulator)**:
- Use a 5V regulator (e.g., LM7805 or buck converter)
- Longer runtime than AA
- Requires charging circuit

### Option 3: USB Power Bank

- 5V USB power bank (10000mAh+)
- Cut a USB cable to expose +5V and GND
- Portable and convenient
- May have auto-shutoff issues with low current draw

---

## Safety Warnings

### ⚠️ CRITICAL SAFETY RULES

1. **NEVER power servos directly from Raspberry Pi 5V pins**
   - Servos can draw several hundred mA each
   - Pi's 5V pins have limited current capacity
   - Can damage Pi or cause unstable operation

2. **Always connect grounds together**
   - Pi GND and power supply GND must be connected
   - Without common ground, PWM signals won't work correctly

3. **Check polarity**
   - Reversing power polarity can damage servos
   - Red = (+), Black/Brown = (-)

4. **Don't exceed voltage ratings**
   - Most hobby servos: 4.8V-6V
   - Check your servo's datasheet

5. **Watch for overheating**
   - If servos or power supply get hot, disconnect immediately
   - May indicate short circuit or wrong voltage

---

## Testing Hardware

### Test 1: Check Microphone

```bash
arecord -l
arecord -D hw:1,0 -f S16_LE -r 44100 -d 5 test.wav
aplay test.wav
```

### Test 2: Check pigpio Daemon

```bash
sudo systemctl start pigpiod
sudo systemctl status pigpiod
```

Should show "active (running)".

### Test 3: Test Single Servo

Create a simple test file:

```bash
nano test_one_servo.py
```

```python
import pigpio
import time

GPIO_PIN = 17
pi = pigpio.pi()

if not pi.connected:
    print("Cannot connect to pigpio daemon")
    exit()

print("Moving servo...")
pi.set_servo_pulsewidth(GPIO_PIN, 1500)  # Center
time.sleep(1)
pi.set_servo_pulsewidth(GPIO_PIN, 1000)  # 0 degrees
time.sleep(1)
pi.set_servo_pulsewidth(GPIO_PIN, 2000)  # 180 degrees
time.sleep(1)
pi.set_servo_pulsewidth(GPIO_PIN, 1500)  # Center
time.sleep(1)
pi.set_servo_pulsewidth(GPIO_PIN, 0)     # Off

pi.stop()
print("Done!")
```

Run:
```bash
sudo python3 test_one_servo.py
```

Servo should move if wired correctly!

---

## Troubleshooting

### Microphone Issues

**Problem**: `arecord -l` doesn't show USB device

- Try different USB port
- Check `dmesg | grep audio` for errors
- Try different microphone

**Problem**: Recording is silent

- Check microphone isn't muted
- Use `alsamixer` to adjust input volume
- Test on another computer to verify microphone works

### Servo Issues

**Problem**: Servo doesn't move

- Check pigpiod is running: `sudo systemctl status pigpiod`
- Verify GPIO pin number is correct (BCM numbering!)
- Check servo power supply is connected and ON
- Verify common ground connection
- Test servo with Python script above

**Problem**: Servo jitters or moves erratically

- Check power supply has enough current (2-3A)
- Verify ground connection is solid
- Try lower PWM frequency (default 50Hz is usually fine)
- Reduce number of servos or use stronger power supply

**Problem**: Raspberry Pi crashes when servos move

- Servos likely drawing power from Pi (BAD!)
- Double-check external power supply is connected
- Verify you're NOT using Pi's 5V pins for servo power

### Power Issues

**Problem**: Servos work individually but not all together

- Power supply doesn't have enough current
- Use higher amperage power supply (3A+)
- Or power servos from separate supplies (maintain common ground)

**Problem**: Raspberry Pi reboots randomly

- Power supply for Pi may be inadequate
- Use quality 2.5A+ power supply for the Pi itself
- Check for short circuits in wiring

---

## Next Steps

Once hardware is set up:

1. Install software dependencies (see main README.md)
2. Build the project (`mkdir build && cd build && cmake .. && make`)
3. Run test programs:
   - `sudo ./examples/test_audio_capture`
   - `sudo ./examples/test_servo`
   - `sudo ./examples/test_fft`
4. Run main program: `sudo ./audio_reactive_robot`

---

## Additional Resources

- Raspberry Pi Pinout: https://pinout.xyz/
- ALSA Documentation: https://www.alsa-project.org/wiki/Documentation
- pigpio Library: http://abyz.me.uk/rpi/pigpio/
- Servo Control Basics: https://learn.adafruit.com/adafruit-arduino-lesson-14-servo-motors

Good luck building your audio reactive robot!

