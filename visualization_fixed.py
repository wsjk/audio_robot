from __future__ import print_function, division
import time
import numpy as np
from scipy.ndimage import gaussian_filter1d
import config
import microphone
import dsp
import sys

# Global variables
_time_prev = time.time() * 1000.0
_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)

# Filters for visualization effects
r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2), alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2), alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2), alpha_decay=0.1, alpha_rise=0.5)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.1, alpha_rise=0.99)
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)

# Audio processing filters
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.5, alpha_rise=0.99)

# Global state
p = np.tile(1.0, (3, config.N_PIXELS // 2))
_prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)
y_roll = np.random.rand(config.N_ROLLING_HISTORY, int(config.MIC_RATE / config.FPS)) / 1e16
prev_fps_update = time.time()

def frames_per_second():
    """Return the estimated frames per second"""
    global _time_prev, _fps
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)

def interpolate(y, new_length):
    """Resize array using linear interpolation"""
    if len(y) == new_length:
        return y
    x_old = np.linspace(0, 1, len(y))
    x_new = np.linspace(0, 1, new_length)
    return np.interp(x_new, x_old, y)

def visualize_scroll(y):
    """Effect that scrolls outwards from center"""
    global p
    y = y**2.0
    gain.update(y)
    y /= gain.value
    y *= 255.0

    # Get RGB values from frequency bands
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))

    # Scroll effect
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)

    # Add new color at center
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b

    return np.concatenate((p[:, ::-1], p), axis=1)

def visualize_energy(y):
    """Effect that expands from center with sound energy"""
    global p
    gain.update(y)
    y /= gain.value
    y *= float((config.N_PIXELS // 2) - 1)

    # Map energy to RGB
    scale = 0.9
    r = int(np.mean(y[:len(y) // 3]**scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3]**scale))
    b = int(np.mean(y[2 * len(y) // 3:]**scale))

    # Set LED colors based on energy
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0

    p_filt.update(p)
    p = np.round(p_filt.value)

    # Apply blur for smooth edges
    for i in range(3):
        p[i, :] = gaussian_filter1d(p[i, :], sigma=4.0)

    return np.concatenate((p[:, ::-1], p), axis=1)

def visualize_spectrum(y):
    """Map Mel filterbank frequencies to LED strip"""
    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS // 2))

    # Calculate color channels
    r = r_filt.update(y)
    g = np.abs(y - _prev_spectrum)
    b = b_filt.update(np.copy(y))
    _prev_spectrum = np.copy(y)

    # Mirror for symmetric output
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))

    return np.array([r, g, b]) * 255

def microphone_update(audio_samples):
    """Process audio input and update visualization"""
    global y_roll, prev_fps_update

    # Normalize and roll audio samples
    y = audio_samples / 2.0**15
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)

    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        return

    # FFT and Mel filterbank processing
    N = len(y_data)
    N_zeros = 2**int(np.ceil(np.log2(N))) - N
    y_padded = np.pad(y_data * np.hamming(N), (0, N_zeros), mode='constant')
    YS = np.abs(np.fft.rfft(y_padded)[:N // 2])

    # Create Mel filterbank
    mel = np.sum(np.atleast_2d(YS).T * dsp.mel_y.T, axis=0)
    mel = mel**2.0

    # Normalize and smooth
    mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
    mel /= mel_gain.value
    mel = mel_smoothing.update(mel)

    # Apply visualization effect
    output = visualization_effect(mel)

    # Update GUI if enabled
    if config.USE_GUI:
        x = np.linspace(config.MIN_FREQUENCY, config.MAX_FREQUENCY, len(mel))
        mel_curve.setData(x=x, y=mel)
        app.processEvents()

    # Display FPS
    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if time.time() - 0.5 > prev_fps_update:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))

# Set visualization type from command line argument
visualization_effects = {
    "spectrum": visualize_spectrum,
    "energy": visualize_energy,
    "scroll": visualize_scroll
}

# Default to spectrum if no argument provided or invalid argument
if len(sys.argv) > 1:
    visualization_effect = visualization_effects.get(sys.argv[1], visualize_spectrum)
    if sys.argv[1] not in visualization_effects:
        print(f"Unknown visualization effect '{sys.argv[1]}'. Using default: spectrum")
        print("Available effects: spectrum, energy, scroll")
else:
    visualization_effect = visualize_spectrum
    print("No visualization effect specified. Using default: spectrum")
    print("Usage: python visualization_simplified.py [spectrum|energy|scroll]")

if __name__ == '__main__':
    if config.USE_GUI:
        from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
        import pyqtgraph as pg

        # Create simple GUI
        app = QtWidgets.QApplication([])
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(100,100,100))
        view.setCentralItem(layout)
        view.show()
        view.setWindowTitle('Audio Visualization')
        view.resize(800, 400)

        # Mel filterbank plot
        fft_plot = layout.addPlot(title='Audio Spectrum')
        fft_plot.setRange(yRange=[-0.1, 1.2])
        mel_curve = pg.PlotCurveItem()
        mel_curve.setData(x=np.array(range(1, config.N_FFT_BINS + 1)), y=np.zeros(config.N_FFT_BINS))
        fft_plot.addItem(mel_curve)

    # Start audio processing
    microphone.start_stream(microphone_update)
