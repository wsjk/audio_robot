/**
 * @file AudioCapture.h
 * @brief Header file for audio capture from USB microphone using ALSA
 *
 * This class provides an interface to capture audio from a USB microphone
 * connected to the Raspberry Pi. It uses the ALSA (Advanced Linux Sound
 * Architecture) library to interface with audio hardware.
 *
 * Key Concepts for Beginners:
 * - ALSA: Linux sound system that allows programs to use audio devices
 * - Sample Rate: Number of audio samples captured per second (e.g., 44100 Hz)
 * - Buffer: Temporary storage for audio data before processing
 * - PCM: Pulse Code Modulation, the standard way digital audio is represented
 *
 * @author Audio Reactive Robot Project
 * @date 2026
 */

#ifndef AUDIO_CAPTURE_H
#define AUDIO_CAPTURE_H

#include <string>
#include <vector>
#include <memory>
#include <alsa/asoundlib.h>

/**
 * @class AudioCapture
 * @brief Captures audio from a USB microphone using ALSA
 *
 * This class handles all the low-level details of setting up and reading
 * from an audio input device. You create an AudioCapture object, initialize
 * it, and then repeatedly call capture() to get audio samples.
 *
 * Example usage:
 * @code
 *   AudioCapture audio("default", 44100, 2048);
 *   if (audio.initialize()) {
 *       while (running) {
 *           int samples = audio.capture();
 *           const auto& buffer = audio.getBuffer();
 *           // Process the audio buffer...
 *       }
 *   }
 * @endcode
 */
class AudioCapture {
public:
    /**
     * @brief Constructor for AudioCapture
     *
     * @param deviceName Name of the ALSA audio device (e.g., "default", "hw:1,0")
     *                   "default" uses the system default microphone
     *                   "hw:X,Y" specifies card X, device Y (find with "arecord -l")
     * @param sampleRate Number of samples per second (typical: 44100 or 48000)
     * @param bufferSize Number of samples to capture at once (power of 2 recommended)
     *                   Larger = more latency but more efficient
     *                   Smaller = less latency but more CPU overhead
     */
    AudioCapture(const std::string& deviceName, unsigned int sampleRate, unsigned int bufferSize);

    /**
     * @brief Destructor - cleans up ALSA resources
     */
    ~AudioCapture();

    /**
     * @brief Initialize the audio capture device
     *
     * This must be called before capture(). It opens the audio device,
     * sets up the audio format (16-bit mono), and prepares it for recording.
     *
     * @return true if initialization successful, false otherwise
     */
    bool initialize();

    /**
     * @brief Capture audio samples from the microphone
     *
     * This function blocks (waits) until enough audio samples are available.
     * The captured samples are stored in an internal buffer that you can
     * access with getBuffer().
     *
     * @return Number of samples captured, or negative value on error
     */
    int capture();

    /**
     * @brief Get the buffer containing the most recent audio samples
     *
     * The buffer contains floating-point values typically ranging from
     * -1.0 to 1.0, representing the audio waveform amplitude.
     *
     * @return Const reference to the audio buffer (read-only)
     */
    const std::vector<float>& getBuffer() const;

    /**
     * @brief Get the sample rate being used
     * @return Sample rate in Hz (e.g., 44100)
     */
    unsigned int getSampleRate() const;

    /**
     * @brief Get the buffer size being used
     * @return Number of samples in each buffer
     */
    unsigned int getBufferSize() const;

    /**
     * @brief Check if audio device is initialized and ready
     * @return true if ready to capture audio
     */
    bool isInitialized() const;

    /**
     * @brief Close the audio device and free resources
     */
    void cleanup();

private:
    std::string deviceName_;           ///< ALSA device name (e.g., "default")
    unsigned int sampleRate_;          ///< Sample rate in Hz
    unsigned int bufferSize_;          ///< Number of samples per capture
    bool initialized_;                 ///< Whether device is initialized

    snd_pcm_t* captureHandle_;        ///< ALSA PCM device handle (pointer to audio device)
    std::vector<short> rawBuffer_;    ///< Raw 16-bit integer samples from ALSA
    std::vector<float> audioBuffer_;  ///< Normalized floating-point samples (-1.0 to 1.0)

    /**
     * @brief Convert raw 16-bit samples to normalized float samples
     *
     * ALSA gives us 16-bit integers (range: -32768 to 32767).
     * We convert these to floats (range: -1.0 to 1.0) for easier processing.
     */
    void convertToFloat();
};

#endif // AUDIO_CAPTURE_H

