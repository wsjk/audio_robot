# Project Structure

```
audio-reactive-robot/
│
├── README.md                    # Main project documentation
├── QUICKSTART.md               # Quick start guide for getting up and running
├── LICENSE                     # MIT License
├── CMakeLists.txt             # Build configuration for CMake
├── build.sh                   # Automated build script
├── .gitignore                 # Git ignore file
│
├── include/                   # Header files (class declarations)
│   ├── AudioCapture.h         # Audio capture from USB microphone
│   ├── FFTAnalyzer.h          # FFT audio analysis
│   └── ServoController.h      # Servo motor control via GPIO
│
├── src/                       # Implementation files (actual code)
│   ├── main.cpp               # Main application - ties everything together
│   ├── AudioCapture.cpp       # AudioCapture implementation
│   ├── FFTAnalyzer.cpp        # FFTAnalyzer implementation
│   └── ServoController.cpp    # ServoController implementation
│
├── examples/                  # Example programs demonstrating library usage
│   ├── test_audio_capture.cpp # Test audio capture functionality
│   ├── test_fft.cpp           # Test FFT analysis
│   └── test_servo.cpp         # Test servo control
│
├── docs/                      # Documentation
│   ├── CPP_GUIDE.md           # C++ programming guide for beginners
│   ├── HARDWARE_SETUP.md      # Hardware setup and wiring guide
│   └── API_REFERENCE.md       # Complete API documentation
│
└── build/                     # Build directory (created during compilation)
    ├── audio_reactive_robot   # Main executable
    ├── test_audio_capture     # Test executable
    ├── test_fft              # Test executable
    └── test_servo            # Test executable
```

## File Descriptions

### Root Level Files

- **README.md**: Main project documentation with overview, installation, and usage
- **QUICKSTART.md**: Fast-track guide to get running quickly
- **LICENSE**: MIT License for the project
- **CMakeLists.txt**: CMake build configuration - defines how to compile the project
- **build.sh**: Bash script to automate the build process with dependency checking
- **.gitignore**: Specifies files to ignore in version control

### include/ - Header Files

Header files declare the interface (what a class can do) without implementation details.

- **AudioCapture.h**:
  - Declares AudioCapture class for capturing audio from USB microphone
  - Uses ALSA library
  - Methods: initialize(), capture(), getBuffer()

- **FFTAnalyzer.h**:
  - Declares FFTAnalyzer class for frequency analysis
  - Uses FFTW3 library
  - Methods: analyze(), getBassLevel(), getMidLevel(), getTrebleLevel()

- **ServoController.h**:
  - Declares ServoController and MultiServoController classes
  - Uses pigpio library for GPIO control
  - Methods: setAngle(), setPulseWidth(), smoothMove()

### src/ - Implementation Files

Implementation files contain the actual code that makes everything work.

- **main.cpp**:
  - Main application entry point
  - Initializes all components
  - Contains main control loop
  - Maps audio frequencies to servo movements
  - Handles graceful shutdown

- **AudioCapture.cpp**:
  - Implements audio capture functionality
  - Configures ALSA device
  - Reads audio samples
  - Converts 16-bit integers to normalized floats

- **FFTAnalyzer.cpp**:
  - Implements FFT analysis
  - Converts time-domain audio to frequency-domain
  - Calculates magnitude spectrum
  - Computes frequency band levels

- **ServoController.cpp**:
  - Implements servo control
  - Connects to pigpio daemon
  - Generates PWM signals for servos
  - Provides both single and multi-servo control

### examples/ - Test Programs

Standalone programs that demonstrate how to use each component.

- **test_audio_capture.cpp**:
  - Tests audio capture functionality
  - Displays real-time audio levels with VU meters
  - Good for verifying microphone works

- **test_fft.cpp**:
  - Tests FFT analysis
  - Shows frequency spectrum visualization
  - Displays bass, mid, and treble levels

- **test_servo.cpp**:
  - Tests servo control
  - Runs various movement patterns
  - Tests individual and multi-servo control

### docs/ - Documentation

Comprehensive documentation for understanding and using the project.

- **CPP_GUIDE.md**:
  - Explains C++ concepts used in the project
  - Great for developers not familiar with C++
  - Covers classes, pointers, memory management, etc.

- **HARDWARE_SETUP.md**:
  - Detailed hardware setup instructions
  - Wiring diagrams for servos
  - Safety warnings and best practices
  - Troubleshooting hardware issues

- **API_REFERENCE.md**:
  - Complete API documentation
  - All classes, methods, and parameters
  - Usage examples for each component

### build/ - Build Directory

Created during compilation, contains compiled binaries.

- **audio_reactive_robot**: Main program executable
- **test_***: Test program executables
- **audio_reactive_lib.a**: Static library containing compiled classes
- Various CMake and build artifacts

## Component Dependencies

```
┌─────────────────────────────────────────────────────────┐
│                    main.cpp                              │
│                (Main Application)                        │
└────────┬────────────────┬────────────────┬─────────────┘
         │                │                │
         v                v                v
┌────────────────┐ ┌─────────────┐ ┌────────────────────┐
│ AudioCapture   │ │ FFTAnalyzer │ │ ServoController    │
│                │ │             │ │                    │
│ - Captures     │ │ - Analyzes  │ │ - Controls servos  │
│   audio from   │ │   frequency │ │   via GPIO         │
│   USB mic      │ │   spectrum  │ │                    │
└───────┬────────┘ └──────┬──────┘ └─────────┬──────────┘
        │                 │                   │
        v                 v                   v
┌────────────────┐ ┌─────────────┐ ┌────────────────────┐
│  ALSA Library  │ │FFTW3 Library│ │  pigpio Library    │
│  (libasound)   │ │ (libfftw3f) │ │  (libpigpio)       │
└────────────────┘ └─────────────┘ └────────────────────┘
```

## Data Flow

```
USB Microphone
      │
      │ Audio signal
      v
┌─────────────────┐
│  AudioCapture   │
│  - Captures raw │
│    audio data   │
└────────┬────────┘
         │
         │ Float buffer (-1.0 to 1.0)
         v
┌─────────────────┐
│  FFTAnalyzer    │
│  - Performs FFT │
│  - Extracts     │
│    frequency    │
│    bands        │
└────────┬────────┘
         │
         │ Bass/Mid/Treble levels (0.0-1.0)
         v
┌─────────────────┐
│   main.cpp      │
│  - Maps audio   │
│    to servo     │
│    angles       │
└────────┬────────┘
         │
         │ Angle commands (0-180°)
         v
┌─────────────────┐
│ServoController  │
│  - Converts to  │
│    PWM signals  │
└────────┬────────┘
         │
         │ PWM pulses
         v
    Servo Motors
```

## Library Architecture

The project is designed as a reusable library:

```
audio_reactive_lib.a (Static Library)
├── AudioCapture    (Reusable audio capture)
├── FFTAnalyzer     (Reusable FFT analysis)
└── ServoController (Reusable servo control)

Applications using the library:
├── audio_reactive_robot (Main program)
├── test_audio_capture   (Test audio)
├── test_fft            (Test FFT)
└── test_servo          (Test servos)

Future projects can link against audio_reactive_lib.a
to reuse these components!
```

## Usage in Other Projects

To use this library in your own projects:

1. Copy `include/` and `src/` files to your project
2. Or link against the compiled `audio_reactive_lib.a`
3. Include headers: `#include "AudioCapture.h"`
4. Link libraries: `-lasound -lfftw3f -lpigpio`

Example CMakeLists.txt for new project:
```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

# Include audio reactive robot library
include_directories(/path/to/audio-reactive-robot/include)
link_directories(/path/to/audio-reactive-robot/build)

add_executable(my_app my_app.cpp)
target_link_libraries(my_app 
    audio_reactive_lib
    asound
    fftw3f
    pigpio
    rt
    pthread
)
```

## Modular Design Benefits

1. **Separation of Concerns**: Each class has a single, clear purpose
2. **Reusability**: Classes can be used in other projects
3. **Testability**: Each component can be tested independently
4. **Maintainability**: Easy to update or fix individual components
5. **Extensibility**: Easy to add new features or components

## Build Process

```
Source Files (.cpp, .h)
         │
         v
    Preprocessing (#include, #define)
         │
         v
    Compilation (g++)
         │
         v
    Object Files (.o)
         │
         v
    Linking (ld)
         │
         v
    Executable
```

CMake automates this process by:
1. Detecting system configuration
2. Finding required libraries
3. Generating Makefiles
4. Invoking compiler with correct flags

## Adding New Components

To add a new component (e.g., LED control):

1. Create header: `include/LEDController.h`
2. Create implementation: `src/LEDController.cpp`
3. Add to CMakeLists.txt: `src/LEDController.cpp` in LIB_SOURCES
4. Use in main.cpp: `#include "LEDController.h"`

The modular design makes it easy to extend!

