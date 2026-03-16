# Contributing Guide

Thank you for your interest in improving the Audio Reactive Robot project! This guide will help you understand how to contribute effectively.

## Ways to Contribute

1. **Report Bugs**: Found a bug? Let us know!
2. **Suggest Features**: Have an idea? We'd love to hear it!
3. **Improve Documentation**: Fix typos, clarify explanations
4. **Add Examples**: Create new example programs
5. **Enhance Code**: Optimize performance, add features
6. **Share Projects**: Show us what you built!

## Getting Started

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd audio-reactive-robot

# Install dependencies
sudo apt-get install -y build-essential cmake
sudo apt-get install -y libasound2-dev libfftw3-dev pigpio

# Build the project
./build.sh
```

### 2. Understand the Code

Read these files in order:
1. `README.md` - Project overview
2. `docs/PROJECT_STRUCTURE.md` - Architecture
3. `docs/CPP_GUIDE.md` - C++ concepts used
4. Source code in `src/` and `include/`

## Code Style Guidelines

### Naming Conventions

```cpp
// Classes: PascalCase
class AudioCapture { };

// Functions/Methods: camelCase
void captureAudio();

// Private member variables: camelCase with trailing underscore
int sampleRate_;

// Constants: UPPER_CASE
const int BUFFER_SIZE = 2048;

// Local variables: camelCase
int samplesRead = 0;
```

### Comments

Use descriptive comments:

```cpp
// Good: Explains WHY
// Apply Hann window to reduce spectral leakage
applyWindow(buffer);

// Bad: Just repeats code
// Apply window
applyWindow(buffer);
```

Every function should have a comment block:

```cpp
/**
 * @brief Brief description of what the function does
 * 
 * Longer description if needed, explaining the purpose,
 * approach, or any important details.
 * 
 * @param paramName Description of parameter
 * @return Description of return value
 */
int myFunction(int paramName) {
    // Implementation
}
```

### Error Handling

Always check return values and provide helpful error messages:

```cpp
// Good
if (!audio.initialize()) {
    std::cerr << "ERROR: Failed to initialize audio" << std::endl;
    std::cerr << "Try: arecord -l to list devices" << std::endl;
    return false;
}

// Bad
audio.initialize();  // Ignores errors
```

### Resource Management

Use RAII (Resource Acquisition Is Initialization):

```cpp
class MyClass {
public:
    MyClass() {
        // Acquire resources in constructor
        resource_ = allocateResource();
    }
    
    ~MyClass() {
        // Release resources in destructor
        if (resource_) {
            freeResource(resource_);
        }
    }
    
private:
    Resource* resource_;
};
```

## Adding New Features

### Adding a New Class

Let's say you want to add LED control:

1. **Create Header** (`include/LEDController.h`):

```cpp
#ifndef LED_CONTROLLER_H
#define LED_CONTROLLER_H

/**
 * @brief Controls RGB LED strips via GPIO
 * 
 * Detailed description of the class and its purpose.
 */
class LEDController {
public:
    LEDController(unsigned int redPin, unsigned int greenPin, unsigned int bluePin);
    ~LEDController();
    
    bool initialize();
    void setColor(uint8_t red, uint8_t green, uint8_t blue);
    void cleanup();
    
private:
    unsigned int redPin_;
    unsigned int greenPin_;
    unsigned int bluePin_;
    bool initialized_;
};

#endif // LED_CONTROLLER_H
```

2. **Create Implementation** (`src/LEDController.cpp`):

```cpp
#include "LEDController.h"
#include <iostream>
#include <pigpio.h>

LEDController::LEDController(unsigned int redPin, 
                             unsigned int greenPin, 
                             unsigned int bluePin)
    : redPin_(redPin)
    , greenPin_(greenPin)
    , bluePin_(bluePin)
    , initialized_(false)
{
}

LEDController::~LEDController() {
    cleanup();
}

bool LEDController::initialize() {
    // Implementation here
    return true;
}

void LEDController::setColor(uint8_t red, uint8_t green, uint8_t blue) {
    // Implementation here
}

void LEDController::cleanup() {
    // Implementation here
}
```

3. **Update CMakeLists.txt**:

```cmake
set(LIB_SOURCES
    src/AudioCapture.cpp
    src/FFTAnalyzer.cpp
    src/ServoController.cpp
    src/LEDController.cpp  # Add this line
)
```

4. **Create Test Program** (`examples/test_led.cpp`):

```cpp
#include "LEDController.h"
#include <iostream>

int main() {
    LEDController led(17, 18, 27);
    
    if (!led.initialize()) {
        return 1;
    }
    
    // Test code here
    
    led.cleanup();
    return 0;
}
```

5. **Update CMakeLists.txt** for test:

```cmake
add_executable(test_led examples/test_led.cpp)
target_link_libraries(test_led audio_reactive_lib)
```

6. **Document**: Add to API_REFERENCE.md

### Modifying Existing Classes

When modifying existing code:

1. **Test First**: Make sure existing tests pass
2. **Preserve API**: Don't break existing code using the class
3. **Add Tests**: Create tests for new functionality
4. **Update Docs**: Update relevant documentation

### Testing Changes

Before submitting:

```bash
# Clean build
rm -rf build
./build.sh

# Run all tests
cd build
sudo ./test_audio_capture
sudo ./test_fft
sudo ./test_servo
sudo ./audio_reactive_robot  # Test briefly
```

## Documentation Standards

### Code Comments

- Use `//` for single-line comments
- Use `/* */` for multi-line comments
- Use `/** */` for documentation blocks (Doxygen style)

### Documentation Files

When adding or updating documentation:

1. Use clear, simple language
2. Include code examples
3. Add diagrams where helpful (ASCII art is fine)
4. Explain concepts for beginners
5. Provide troubleshooting tips

Example structure:

```markdown
# Feature Name

Brief description of the feature.

## Usage

```cpp
// Code example
```

## Parameters

- `param1`: Description
- `param2`: Description

## Example

Complete working example...

## Troubleshooting

Common issues and solutions...
```

## Submitting Changes

### For Small Changes

- Fix typos
- Update comments
- Small bug fixes

Just make the change and note what you did.

### For Larger Changes

1. **Describe the Change**: What and why?
2. **Test Thoroughly**: On actual hardware if possible
3. **Update Documentation**: Reflect your changes in docs
4. **Provide Examples**: Show how to use new features

### Git Commit Messages

Use clear, descriptive commit messages:

```
Good:
- "Add smooth movement function to ServoController"
- "Fix audio buffer overflow in AudioCapture"
- "Update HARDWARE_SETUP.md with USB power bank option"

Bad:
- "Update"
- "Fix stuff"
- "Changes"
```

## Project Priorities

When contributing, keep these priorities in mind:

1. **Correctness**: Code must work reliably
2. **Safety**: No hardware damage from incorrect usage
3. **Documentation**: Well-documented is as important as well-coded
4. **Beginner-Friendly**: Maintain accessibility for newcomers
5. **Performance**: Optimize where it matters
6. **Reusability**: Keep components modular

## Feature Ideas

Looking for contribution ideas? Here are some:

### Easy (Good for First Contribution)
- Add more example programs
- Improve error messages
- Add configuration file support
- Create additional documentation

### Medium
- Add beat detection algorithm
- Implement pattern recording/playback
- Create web interface for control
- Add support for different servo types
- Implement noise gate for audio

### Advanced
- Multi-threaded audio processing
- Real-time FFT optimization
- Support for multiple audio sources
- Network synchronization between multiple robots
- Machine learning for music analysis

## Code Review Checklist

Before submitting, check:

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] New features have tests
- [ ] Documentation updated
- [ ] Comments explain WHY, not just WHAT
- [ ] Error handling is present
- [ ] Resources are properly cleaned up
- [ ] Follows project code style
- [ ] Works on actual Raspberry Pi hardware

## Questions?

If you have questions:

1. Check existing documentation
2. Look at similar code in the project
3. Ask on the project's discussion board
4. Open an issue for clarification

## Recognition

Contributors will be:
- Listed in contributors section
- Mentioned in release notes
- Appreciated by the community!

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on what's best for the project
- Assume good intentions
- Give credit where due

## Legal

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Audio Reactive Robot project! Your efforts help make robotics and audio processing more accessible to everyone. 🎵🤖

Happy coding!

