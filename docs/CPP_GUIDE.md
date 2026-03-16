# C++ Programming Guide for Audio Reactive Robot

This guide explains the C++ concepts used in this project for developers who may not be familiar with C++.

## Table of Contents

1. [Basic C++ Concepts](#basic-c-concepts)
2. [Classes and Objects](#classes-and-objects)
3. [Memory Management](#memory-management)
4. [Libraries Used](#libraries-used)
5. [Common Patterns](#common-patterns)

---

## Basic C++ Concepts

### Header Files (.h) vs Implementation Files (.cpp)

C++ splits code into two types of files:

**Header Files (.h)**
- Declare what a class or function can do
- Like a "table of contents" or "interface"
- Example: `AudioCapture.h` declares the AudioCapture class

**Implementation Files (.cpp)**
- Define HOW things work
- Contains the actual code that runs
- Example: `AudioCapture.cpp` implements the methods

### Why Split Files?

```cpp
// AudioCapture.h - Declaration
class AudioCapture {
public:
    bool initialize();  // Declared but not implemented here
};

// AudioCapture.cpp - Implementation
bool AudioCapture::initialize() {
    // Actual code here
    return true;
}
```

This allows:
- Other files to use the class without seeing implementation details
- Faster compilation (only changed files need recompiling)
- Cleaner organization

### Data Types

```cpp
int num = 42;              // Integer number
float pi = 3.14f;          // Floating-point number (single precision)
double precise = 3.14159;  // Double precision float
bool flag = true;          // Boolean (true/false)
char letter = 'A';         // Single character
std::string text = "Hi";   // String (text)
```

### Pointers and References

**Pointers** (`*`) - Store memory addresses:
```cpp
int value = 42;
int* ptr = &value;  // ptr stores the address of value
*ptr = 50;          // Changes value to 50
```

**References** (`&`) - Aliases to variables:
```cpp
int value = 42;
int& ref = value;   // ref is another name for value
ref = 50;           // Changes value to 50
```

**Const References** - Read-only access:
```cpp
void printBuffer(const std::vector<float>& buffer) {
    // Can read buffer but cannot modify it
}
```

---

## Classes and Objects

### What is a Class?

A class is a blueprint for creating objects. Think of it like a template:

```cpp
class AudioCapture {
    // Private: Only accessible within the class
private:
    int sampleRate_;
    
    // Public: Accessible from outside
public:
    AudioCapture(int rate);  // Constructor
    ~AudioCapture();         // Destructor
    void capture();          // Method
};
```

### Constructor and Destructor

**Constructor** - Called when object is created:
```cpp
AudioCapture::AudioCapture(int rate) 
    : sampleRate_(rate)  // Initialize member variables
{
    // Setup code here
}
```

**Destructor** - Called when object is destroyed:
```cpp
AudioCapture::~AudioCapture() {
    // Cleanup code here (close files, free memory, etc.)
}
```

### Object Creation

```cpp
// Create object (constructor called)
AudioCapture audio("default", 44100, 2048);

// Use object
audio.initialize();
audio.capture();

// When audio goes out of scope, destructor is called automatically
```

### Member Variables vs Local Variables

```cpp
class MyClass {
private:
    int memberVar_;  // Belongs to the object, exists until object destroyed
    
public:
    void myMethod() {
        int localVar = 42;  // Only exists during this method call
    }
};
```

Note: Member variables often end with `_` by convention.

---

## Memory Management

### Stack vs Heap

**Stack Allocation** (automatic):
```cpp
void function() {
    AudioCapture audio(...);  // Created on stack
    // Automatically destroyed when function exits
}
```

**Heap Allocation** (manual):
```cpp
AudioCapture* audio = new AudioCapture(...);  // Created on heap
// Must manually delete when done
delete audio;
```

### RAII (Resource Acquisition Is Initialization)

C++ philosophy: Resources are acquired in constructor, released in destructor.

```cpp
class FileHandler {
public:
    FileHandler(const char* filename) {
        file_ = fopen(filename, "r");  // Acquire resource
    }
    
    ~FileHandler() {
        if (file_) fclose(file_);      // Release resource
    }
    
private:
    FILE* file_;
};

// Usage
{
    FileHandler handler("data.txt");
    // Use file...
}  // Destructor automatically called, file closed
```

Our audio reactive robot uses this pattern extensively!

### Smart Pointers (Modern C++)

Instead of raw pointers, use smart pointers that handle cleanup:

```cpp
#include <memory>

std::unique_ptr<AudioCapture> audio = 
    std::make_unique<AudioCapture>(...);
// No need to delete - automatically cleaned up
```

---

## Libraries Used

### Standard Library (std::)

```cpp
#include <vector>   // Dynamic arrays
#include <string>   // Text strings
#include <iostream> // Input/output

std::vector<float> buffer;     // Like a Python list
std::string name = "audio";    // Like a Python string
std::cout << "Hello\n";        // Print to console
```

### ALSA (Advanced Linux Sound Architecture)

Used for audio capture:

```cpp
#include <alsa/asoundlib.h>

snd_pcm_t* handle;                    // Audio device handle
snd_pcm_open(&handle, "default", ...); // Open device
snd_pcm_readi(handle, buffer, size);   // Read samples
```

### FFTW3 (Fastest Fourier Transform in the West)

Used for FFT:

```cpp
#include <fftw3.h>

float* input = fftwf_malloc(...);           // Allocate input
fftwf_complex* output = fftwf_malloc(...);  // Allocate output
fftwf_plan plan = fftwf_plan_dft_r2c_1d(...); // Create plan
fftwf_execute(plan);                        // Execute FFT
```

### pigpio

Used for GPIO control:

```cpp
#include <pigpio.h>

gpioInitialise();           // Connect to daemon
gpioSetMode(pin, OUTPUT);   // Set pin as output
gpioServo(pin, pulseWidth); // Set servo pulse
gpioTerminate();            // Disconnect
```

---

## Common Patterns

### Initialization Pattern

```cpp
class MyClass {
public:
    MyClass() : initialized_(false) {}
    
    bool initialize() {
        if (initialized_) return true;  // Already initialized
        
        // Setup code...
        
        initialized_ = true;
        return true;
    }
    
private:
    bool initialized_;
};
```

Why? Some setup (like hardware initialization) might fail, so we separate construction from initialization.

### Error Handling

```cpp
bool MyClass::doSomething() {
    int result = someFunction();
    if (result < 0) {
        std::cerr << "ERROR: Operation failed\n";
        return false;  // Indicate failure
    }
    return true;  // Indicate success
}

// Usage
if (!obj.doSomething()) {
    // Handle error
}
```

### Const Correctness

```cpp
class MyClass {
public:
    // This method doesn't modify the object
    int getValue() const {
        return value_;
    }
    
    // This method modifies the object
    void setValue(int v) {
        value_ = v;
    }
    
private:
    int value_;
};
```

`const` after method signature means "this method won't change the object".

### Vector Usage

```cpp
#include <vector>

// Create vector
std::vector<float> data;

// Add elements
data.push_back(1.5f);
data.push_back(2.5f);

// Access elements
float first = data[0];
float second = data.at(1);  // Bounds-checked

// Get size
size_t count = data.size();

// Iterate
for (size_t i = 0; i < data.size(); ++i) {
    std::cout << data[i] << std::endl;
}

// Modern range-based for loop
for (float value : data) {
    std::cout << value << std::endl;
}
```

### Namespace Usage

```cpp
// Standard library uses std namespace
std::cout << "Hello\n";
std::vector<int> numbers;

// Or use namespace (not recommended in headers)
using namespace std;
cout << "Hello\n";
vector<int> numbers;
```

### Templates

Templates allow generic programming:

```cpp
// Template in std::vector
std::vector<int> integers;     // Vector of ints
std::vector<float> floats;     // Vector of floats
std::vector<std::string> text; // Vector of strings
```

### Type Casting

```cpp
// C-style cast (old, not recommended)
float f = (float)42;

// C++ style casts (preferred)
float f = static_cast<float>(42);     // Convert types
const float* p = const_cast<float*>(cp); // Remove const (dangerous!)
```

---

## Compilation Process

### How C++ Code Becomes a Program

1. **Preprocessing** - Handle `#include`, `#define`, etc.
2. **Compilation** - Convert .cpp files to object files (.o)
3. **Linking** - Combine object files and libraries into executable

### CMake Build System

Instead of running compiler manually, we use CMake:

```bash
mkdir build
cd build
cmake ..     # Configure build
make         # Compile
```

CMake reads `CMakeLists.txt` and generates Makefiles.

---

## Best Practices Used in This Project

1. **RAII** - Resources acquired in constructor, released in destructor
2. **Const correctness** - Mark read-only methods as `const`
3. **Error handling** - Check return values, report errors clearly
4. **Initialization pattern** - Separate construction from initialization
5. **Clear naming** - Use descriptive names for variables and functions
6. **Comments** - Explain WHY, not just WHAT
7. **Modular design** - Each class has a single, clear purpose

---

## Common Gotchas

### 1. Forgetting to Initialize

```cpp
int value;           // Uninitialized! Contains garbage
int value = 0;       // Correct
```

### 2. Array Index Out of Bounds

```cpp
std::vector<int> v(10);
v[10] = 5;          // ERROR! Valid indices are 0-9
```

### 3. Dereferencing Null Pointer

```cpp
int* ptr = nullptr;
*ptr = 5;           // CRASH! Cannot dereference null
```

### 4. Memory Leaks

```cpp
int* ptr = new int;
// Forgot to delete ptr;  // MEMORY LEAK
```

Use smart pointers or RAII to avoid this!

### 5. Comparing Floats

```cpp
float a = 0.1f + 0.2f;
if (a == 0.3f) { }  // May fail due to floating-point precision!

// Instead:
const float EPSILON = 0.0001f;
if (fabs(a - 0.3f) < EPSILON) { }  // Correct
```

---

## Further Learning

### Recommended Resources

- **C++ Reference**: https://en.cppreference.com/
- **LearnCpp**: https://www.learncpp.com/
- **C++ Core Guidelines**: https://isocpp.github.io/CppCoreGuidelines/

### Practice Exercises

1. Modify `main.cpp` to add a new frequency band
2. Create a new servo movement pattern in `test_servo.cpp`
3. Add a peak detection feature to `FFTAnalyzer`
4. Implement audio recording to file in `AudioCapture`

---

## Questions?

If you have questions about specific C++ concepts used in this project, check:

1. Comments in the source files
2. This guide
3. C++ reference documentation
4. Ask on C++ forums or communities

Good luck and happy coding!

