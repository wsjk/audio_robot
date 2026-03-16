#!/bin/bash

# Build script for Audio Reactive Robot
# This script automates the build process on Raspberry Pi

set -e  # Exit on error

echo "=========================================="
echo "  Audio Reactive Robot Build Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi (optional)
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model)
    echo -e "${GREEN}Detected: $MODEL${NC}"
    echo ""
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "Checking for required tools..."

if ! command_exists g++; then
    echo -e "${RED}ERROR: g++ not found${NC}"
    echo "Install with: sudo apt-get install build-essential"
    exit 1
fi

if ! command_exists cmake; then
    echo -e "${RED}ERROR: cmake not found${NC}"
    echo "Install with: sudo apt-get install cmake"
    exit 1
fi

echo -e "${GREEN}✓ Build tools found${NC}"
echo ""

# Check for required libraries
echo "Checking for required libraries..."

MISSING_LIBS=0

if ! ldconfig -p | grep -q libasound; then
    echo -e "${RED}✗ ALSA library not found${NC}"
    echo "  Install with: sudo apt-get install libasound2-dev"
    MISSING_LIBS=1
fi

if ! ldconfig -p | grep -q libfftw3f; then
    echo -e "${RED}✗ FFTW3 library not found${NC}"
    echo "  Install with: sudo apt-get install libfftw3-dev"
    MISSING_LIBS=1
fi

if ! command_exists pigpiod; then
    echo -e "${RED}✗ pigpio not found${NC}"
    echo "  Install with: sudo apt-get install pigpio python3-pigpio"
    MISSING_LIBS=1
fi

if [ $MISSING_LIBS -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Would you like to install missing dependencies? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Installing dependencies..."
        sudo apt-get update
        sudo apt-get install -y libasound2-dev libfftw3-dev pigpio python3-pigpio
        echo -e "${GREEN}Dependencies installed${NC}"
    else
        echo -e "${RED}Cannot build without dependencies${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ All libraries found${NC}"
echo ""

# Check if pigpiod is running
if ! systemctl is-active --quiet pigpiod; then
    echo -e "${YELLOW}pigpiod daemon is not running${NC}"
    echo -e "${YELLOW}Would you like to start it? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Starting pigpiod..."
        sudo systemctl start pigpiod
        sudo systemctl enable pigpiod
        echo -e "${GREEN}pigpiod started and enabled${NC}"
    fi
else
    echo -e "${GREEN}✓ pigpiod daemon is running${NC}"
fi
echo ""

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Run CMake
echo "Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building project..."
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=========================================="
    echo "  Build Successful!"
    echo "==========================================${NC}"
    echo ""
    echo "Executables created:"
    echo "  - audio_reactive_robot    (Main program)"
    echo "  - test_audio_capture      (Test audio)"
    echo "  - test_fft                (Test FFT)"
    echo "  - test_servo              (Test servos)"
    echo ""
    echo "Run with: sudo ./audio_reactive_robot"
    echo ""
else
    echo ""
    echo -e "${RED}=========================================="
    echo "  Build Failed!"
    echo "==========================================${NC}"
    echo ""
    echo "Check the error messages above."
    exit 1
fi

