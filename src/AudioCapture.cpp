/**
 * @file AudioCapture.cpp
 * @brief Implementation of audio capture from USB microphone using ALSA
 *
 * This file contains all the actual code that makes the AudioCapture class work.
 * The header file (AudioCapture.h) declares what the class can do,
 * and this file implements HOW it does it.
 */

#include "AudioCapture.h"
#include <iostream>
#include <cstring>
#include <stdexcept>

/**
 * Constructor: Sets up the object with configuration values
 * but doesn't actually open the audio device yet (that's done in initialize())
 */
AudioCapture::AudioCapture(const std::string& deviceName,
                           unsigned int sampleRate,
                           unsigned int bufferSize)
    : deviceName_(deviceName)
    , sampleRate_(sampleRate)
    , bufferSize_(bufferSize)
    , initialized_(false)
    , captureHandle_(nullptr)
{
    // Pre-allocate memory for our buffers to avoid doing it later
    // This makes audio capture more efficient
    rawBuffer_.resize(bufferSize);     // 16-bit integer samples from hardware
    audioBuffer_.resize(bufferSize);   // Converted floating-point samples
}

/**
 * Destructor: Called automatically when the object is destroyed
 * Makes sure we clean up any resources we're using
 */
AudioCapture::~AudioCapture() {
    cleanup();
}

/**
 * Initialize the audio capture device
 * This is where we do all the ALSA setup to prepare for recording
 */
bool AudioCapture::initialize() {
    // If already initialized, don't do it again
    if (initialized_) {
        return true;
    }

    int err;

    // Step 1: Open the PCM device for recording (capture mode)
    // snd_pcm_open returns 0 on success, negative error code on failure
    err = snd_pcm_open(&captureHandle_, deviceName_.c_str(),
                       SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
        std::cerr << "ERROR: Cannot open audio device '" << deviceName_
                  << "': " << snd_strerror(err) << std::endl;
        std::cerr << "TIP: Try 'arecord -l' to list available devices" << std::endl;
        return false;
    }

    // Step 2: Allocate a hardware parameters object
    // This object holds all the settings for the audio device
    snd_pcm_hw_params_t* hwParams;
    snd_pcm_hw_params_alloca(&hwParams);

    // Step 3: Fill the parameters object with default values
    err = snd_pcm_hw_params_any(captureHandle_, hwParams);
    if (err < 0) {
        std::cerr << "ERROR: Cannot initialize hardware parameters: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    // Step 4: Set the access type (how we read samples)
    // SND_PCM_ACCESS_RW_INTERLEAVED means we read samples in sequential order
    err = snd_pcm_hw_params_set_access(captureHandle_, hwParams,
                                       SND_PCM_ACCESS_RW_INTERLEAVED);
    if (err < 0) {
        std::cerr << "ERROR: Cannot set access type: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    // Step 5: Set the sample format to 16-bit signed little-endian
    // This is the most common format for audio (CD quality uses this)
    err = snd_pcm_hw_params_set_format(captureHandle_, hwParams,
                                       SND_PCM_FORMAT_S16_LE);
    if (err < 0) {
        std::cerr << "ERROR: Cannot set sample format: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    // Step 6: Set the number of channels to 1 (mono)
    // Mono is sufficient for audio analysis and uses less CPU
    // If you need stereo, change this to 2
    err = snd_pcm_hw_params_set_channels(captureHandle_, hwParams, 1);
    if (err < 0) {
        std::cerr << "ERROR: Cannot set channel count: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    // Step 7: Set the sample rate (e.g., 44100 Hz)
    // The actual rate may be slightly different, so we get the real value
    unsigned int actualRate = sampleRate_;
    err = snd_pcm_hw_params_set_rate_near(captureHandle_, hwParams,
                                          &actualRate, 0);
    if (err < 0) {
        std::cerr << "ERROR: Cannot set sample rate: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    // Warn if the actual rate is different from what we asked for
    if (actualRate != sampleRate_) {
        std::cerr << "WARNING: Requested rate " << sampleRate_
                  << " Hz, got " << actualRate << " Hz" << std::endl;
        sampleRate_ = actualRate;
    }

    // Step 8: Set the buffer size (how many samples per capture)
    snd_pcm_uframes_t frames = bufferSize_;
    err = snd_pcm_hw_params_set_period_size_near(captureHandle_, hwParams,
                                                  &frames, 0);
    if (err < 0) {
        std::cerr << "ERROR: Cannot set period size: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    // Step 9: Apply all these settings to the device
    err = snd_pcm_hw_params(captureHandle_, hwParams);
    if (err < 0) {
        std::cerr << "ERROR: Cannot set hardware parameters: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    // Step 10: Prepare the device for use
    err = snd_pcm_prepare(captureHandle_);
    if (err < 0) {
        std::cerr << "ERROR: Cannot prepare audio interface: "
                  << snd_strerror(err) << std::endl;
        cleanup();
        return false;
    }

    initialized_ = true;
    std::cout << "Audio capture initialized successfully:" << std::endl;
    std::cout << "  Device: " << deviceName_ << std::endl;
    std::cout << "  Sample Rate: " << sampleRate_ << " Hz" << std::endl;
    std::cout << "  Buffer Size: " << bufferSize_ << " samples" << std::endl;

    return true;
}

/**
 * Capture audio samples from the microphone
 * This function blocks (waits) until samples are available
 */
int AudioCapture::capture() {
    if (!initialized_) {
        std::cerr << "ERROR: Audio capture not initialized" << std::endl;
        return -1;
    }

    // Read samples from the audio device into our raw buffer
    // snd_pcm_readi() is the ALSA function for reading interleaved samples
    int samplesRead = snd_pcm_readi(captureHandle_, rawBuffer_.data(),
                                     bufferSize_);

    // Check for errors
    if (samplesRead < 0) {
        // Handle buffer overrun (xrun) - this means we didn't read fast enough
        if (samplesRead == -EPIPE) {
            std::cerr << "WARNING: Buffer overrun occurred, recovering..." << std::endl;
            snd_pcm_prepare(captureHandle_);  // Reset the device
            return 0;
        } else {
            std::cerr << "ERROR: Read from audio interface failed: "
                      << snd_strerror(samplesRead) << std::endl;
            return samplesRead;
        }
    }

    // If we got fewer samples than expected, warn the user
    if (samplesRead != static_cast<int>(bufferSize_)) {
        std::cerr << "WARNING: Short read (expected " << bufferSize_
                  << ", got " << samplesRead << ")" << std::endl;
    }

    // Convert the raw 16-bit samples to normalized float samples
    convertToFloat();

    return samplesRead;
}

/**
 * Convert raw 16-bit integer samples to normalized floating-point samples
 *
 * Why do we do this?
 * - 16-bit integers range from -32768 to 32767 (awkward to work with)
 * - Floats from -1.0 to 1.0 are much easier for audio processing
 * - Most audio algorithms expect normalized float samples
 */
void AudioCapture::convertToFloat() {
    // The maximum value for a 16-bit signed integer
    const float maxValue = 32768.0f;

    // Convert each sample
    for (size_t i = 0; i < bufferSize_; ++i) {
        // Divide by maxValue to normalize to -1.0 to 1.0 range
        audioBuffer_[i] = static_cast<float>(rawBuffer_[i]) / maxValue;
    }
}

/**
 * Get the buffer containing the most recent audio samples
 */
const std::vector<float>& AudioCapture::getBuffer() const {
    return audioBuffer_;
}

/**
 * Get the sample rate being used
 */
unsigned int AudioCapture::getSampleRate() const {
    return sampleRate_;
}

/**
 * Get the buffer size being used
 */
unsigned int AudioCapture::getBufferSize() const {
    return bufferSize_;
}

/**
 * Check if the audio device is initialized
 */
bool AudioCapture::isInitialized() const {
    return initialized_;
}

/**
 * Clean up and release resources
 */
void AudioCapture::cleanup() {
    if (captureHandle_ != nullptr) {
        // Drain any remaining samples
        snd_pcm_drain(captureHandle_);
        // Close the device
        snd_pcm_close(captureHandle_);
        captureHandle_ = nullptr;
    }
    initialized_ = false;
}

