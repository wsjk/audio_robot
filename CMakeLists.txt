cmake_minimum_required(VERSION 3.10)
project(AudioReactiveRobot VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

# Find required packages
find_package(Threads REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files for the library
set(LIB_SOURCES
    src/AudioCapture.cpp
    src/FFTAnalyzer.cpp
    src/ServoController.cpp
)

# Create static library
add_library(audio_reactive_lib STATIC ${LIB_SOURCES})

# Link libraries to the static library
target_link_libraries(audio_reactive_lib
    asound          # ALSA library for audio capture
    fftw3f          # FFTW3 library for FFT (single precision)
    pigpio          # pigpio library for GPIO control
    rt              # Real-time library
    Threads::Threads
)

# Main executable
add_executable(audio_reactive_robot src/main.cpp)
target_link_libraries(audio_reactive_robot audio_reactive_lib)

# Example executables
add_executable(test_audio_capture examples/test_audio_capture.cpp)
target_link_libraries(test_audio_capture audio_reactive_lib)

add_executable(test_fft examples/test_fft.cpp)
target_link_libraries(test_fft audio_reactive_lib)

add_executable(test_servo examples/test_servo.cpp)
target_link_libraries(test_servo audio_reactive_lib)

# Install targets
install(TARGETS audio_reactive_robot
        RUNTIME DESTINATION bin)

install(TARGETS audio_reactive_lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/
        DESTINATION include/audio_reactive_robot)

# Print configuration summary
message(STATUS "")
message(STATUS "====================================")
message(STATUS "Audio Reactive Robot Configuration")
message(STATUS "====================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "====================================")
message(STATUS "")

