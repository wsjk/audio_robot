from __future__ import print_function
from __future__ import division
import time
import numpy as np
# Guarded gaussian_filter1d import with fallback
try:  # pragma: no cover
    from scipy.ndimage import gaussian_filter1d  # type: ignore
except Exception:  # pragma: no cover
    def gaussian_filter1d(arr, sigma):
        if sigma <= 0:
            return arr
        win = int(max(1, round(sigma * 3)))
        if win <= 1:
            return arr
        kernel = np.ones(win) / win
        if arr.ndim == 1:
            return np.convolve(arr, kernel, mode='same')
        return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), -1, arr)
import config  # type: ignore
import microphone  # type: ignore
import dsp  # type: ignore
import led  # type: ignore
import sys
import argparse
from typing import Callable, Dict, Any, Optional, Protocol, cast

class Curve(Protocol):  # Lightweight protocol for pyqtgraph curves
    def setData(self, *args, **kwargs) -> Any: ...

_time_prev = time.time() * 1000.0
"""The previous time that the frames_per_second() function was called"""

_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)
"""The low-pass filter used to estimate frames-per-second"""

# Runtime-tunable globals
fps_print_interval: float = 0.5
brightness_scale: float = 1.0
app: Optional[Any] = None
mel_curve: Optional[Curve] = None
r_curve: Optional[Curve] = None
g_curve: Optional[Curve] = None
b_curve: Optional[Curve] = None  # GUI curves initialized later


def frames_per_second() -> float:
    """Return the estimated frames per second.
    The FPS estimate is low-pass filtered to reduce noise.
    """
    global _time_prev, _fps  # removed duplicate global statement
    time_now = time.time() * 1000.0
    dt = time_now - _time_prev
    _time_prev = time_now
    if dt == 0.0:
        return _fps.value
    return _fps.update(1000.0 / dt)


def memoize(function: Callable) -> Callable:
    """Provides a decorator for memoizing functions"""
    from functools import wraps
    memo: Dict[tuple, Any] = {}

    @wraps(function)
    def wrapper(*args):
        if args in memo:
            return memo[args]
        rv = function(*args)
        memo[args] = rv
        return rv
    return wrapper


@memoize
def _normalized_linspace(size: int) -> np.ndarray:
    return np.linspace(0, 1, size)


def interpolate(y: np.ndarray, new_length: int) -> np.ndarray:
    """Resize array by linear interpolation"""
    if len(y) == new_length:
        return y
    x_old = _normalized_linspace(len(y))
    x_new = _normalized_linspace(new_length)
    return np.interp(x_new, x_old, y)


# Filters & state
r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2), alpha_decay=0.2, alpha_rise=0.99)
g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2), alpha_decay=0.05, alpha_rise=0.3)
b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2), alpha_decay=0.1, alpha_rise=0.5)
common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2), alpha_decay=0.99, alpha_rise=0.01)
p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)), alpha_decay=0.1, alpha_rise=0.99)
p = np.tile(1.0, (3, config.N_PIXELS // 2))
gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS), alpha_decay=0.001, alpha_rise=0.99)


def visualize_scroll(y: np.ndarray) -> np.ndarray:
    """Effect that originates in the center and scrolls outwards"""
    global p
    y = y ** 2.0
    gain.update(y)
    # Avoid divide-by-zero
    if np.any(gain.value == 0):
        return np.concatenate((p[:, ::-1], p), axis=1)
    y /= gain.value
    y *= 255.0
    r = int(np.max(y[:len(y) // 3]))
    g = int(np.max(y[len(y) // 3: 2 * len(y) // 3]))
    b = int(np.max(y[2 * len(y) // 3:]))
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    p[0, 0] = r
    p[1, 0] = g
    p[2, 0] = b
    return np.concatenate((p[:, ::-1], p), axis=1)


def visualize_energy(y: np.ndarray) -> np.ndarray:
    """Effect that expands from the center with increasing sound energy"""
    global p
    y = np.copy(y)
    gain.update(y)
    if np.any(gain.value == 0):
        return np.concatenate((p[:, ::-1], p), axis=1)
    y /= gain.value
    y *= float((config.N_PIXELS // 2) - 1)
    scale = 0.9
    r = int(np.mean(y[:len(y) // 3] ** scale))
    g = int(np.mean(y[len(y) // 3: 2 * len(y) // 3] ** scale))
    b = int(np.mean(y[2 * len(y) // 3:] ** scale))
    p[0, :r] = 255.0
    p[0, r:] = 0.0
    p[1, :g] = 255.0
    p[1, g:] = 0.0
    p[2, :b] = 255.0
    p[2, b:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    for ch in range(3):
        p[ch, :] = gaussian_filter1d(p[ch, :], sigma=4.0)
    return np.concatenate((p[:, ::-1], p), axis=1)


_prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)


def visualize_spectrum(y: np.ndarray) -> np.ndarray:
    """Effect that maps the Mel filterbank frequencies onto the LED strip"""
    global _prev_spectrum
    y = np.copy(interpolate(y, config.N_PIXELS // 2))
    common_mode.update(y)
    diff = y - _prev_spectrum
    _prev_spectrum = np.copy(y)
    r = r_filt.update(y - common_mode.value)
    g = np.abs(diff)
    b = b_filt.update(np.copy(y))
    r = np.concatenate((r[::-1], r))
    g = np.concatenate((g[::-1], g))
    b = np.concatenate((b[::-1], b))
    return np.array([r, g, b]) * 255

# Effect mapping & selection
_effects: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def register_effect(name: str, func: Callable[[np.ndarray], np.ndarray]) -> None:
    """Register a new visualization effect at runtime."""
    if not name or not callable(func):
        raise ValueError('Invalid effect registration: {}'.format(name))
    _effects[name] = func

# Register built-in effects
register_effect('spectrum', lambda y: visualize_spectrum(y))
register_effect('energy', lambda y: visualize_energy(y))
register_effect('scroll', lambda y: visualize_scroll(y))

visualization_effect: Callable[[np.ndarray], np.ndarray] = visualize_spectrum

# Plot-related filters
fft_plot_filter = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.5, alpha_rise=0.99)
mel_gain = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(np.tile(1e-1, config.N_FFT_BINS), alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(config.MIN_VOLUME_THRESHOLD, alpha_decay=0.02, alpha_rise=0.02)
fft_window = np.hamming(int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()


def microphone_update(audio_samples: np.ndarray) -> None:
    global y_roll, prev_fps_update, visualization_effect, mel_curve, r_curve, g_curve, b_curve, app, brightness_scale
    # Normalize samples between -1 and 1
    y = audio_samples / 2.0 ** 15
    # Rolling window
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(y)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)

    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        # Silence handling
        led.pixels = np.zeros((3, config.N_PIXELS))
        led.update()
    else:
        N = len(y_data)
        N_zeros = 2 ** int(np.ceil(np.log2(N))) - N
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, N_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:N // 2])
        mel = np.atleast_2d(YS).T * dsp.mel_y.T
        mel = np.sum(mel, axis=0)
        mel = mel ** 2.0
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        # Fix ambiguous truth-value error when mel_gain.value is an array
        if np.isscalar(mel_gain.value):
            if mel_gain.value == 0:
                mel_gain.value = 1e-6
        else:
            if np.any(mel_gain.value == 0):
                mel_gain.value[mel_gain.value == 0] = 1e-6
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        output = visualization_effect(mel)
        # Brightness scaling
        output = np.clip(output * brightness_scale, 0, 255)
        led.pixels = output
        led.update()
        if config.USE_GUI and all(c is not None for c in (mel_curve, r_curve, g_curve, b_curve)) and app is not None:
            x = np.linspace(config.MIN_FREQUENCY, config.MAX_FREQUENCY, len(mel))
            cast(Curve, mel_curve).setData(x=x, y=fft_plot_filter.update(mel))
            cast(Curve, r_curve).setData(y=led.pixels[0])
            cast(Curve, g_curve).setData(y=led.pixels[1])
            cast(Curve, b_curve).setData(y=led.pixels[2])
    if config.USE_GUI and app is not None:
        app.processEvents()

    if config.DISPLAY_FPS:
        fps = frames_per_second()
        if (time.time() - prev_fps_update) > fps_print_interval:
            prev_fps_update = time.time()
            print('FPS {:.0f} / {:.0f}'.format(fps, config.FPS))


# Number of audio samples per frame
samples_per_frame = int(config.MIC_RATE / config.FPS)
# Rolling window initialization with tiny values
y_roll = np.random.rand(config.N_ROLLING_HISTORY, samples_per_frame) / 1e16


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Audio LED visualization')
    parser.add_argument('--effect', '-e', choices=sorted(_effects.keys()), default='spectrum', help='Select visualization effect')
    parser.add_argument('--list-effects', action='store_true', help='List available effects and exit')
    parser.add_argument('--fps-interval', type=float, default=0.5, help='Seconds between FPS prints when DISPLAY_FPS enabled (min 0.1)')
    parser.add_argument('--brightness', type=float, default=1.0, help='Brightness scaling factor (0.01 - 5.0)')
    return parser.parse_args(argv)


def set_effect(name: str) -> None:
    global visualization_effect
    visualization_effect = _effects.get(name, visualize_spectrum)


def main(argv=None) -> int:
    global visualization_effect, fps_print_interval, brightness_scale, app, mel_curve, r_curve, g_curve, b_curve
    try:
        args = parse_args(argv)
    except SystemExit:
        return 1
    if args.list_effects:
        print('Available effects:')
        for k in sorted(_effects.keys()):
            print(' -', k)
        return 0
    fps_print_interval = max(0.1, args.fps_interval)
    brightness_scale = min(5.0, max(0.01, args.brightness))
    set_effect(args.effect)

    if config.USE_GUI:
        try:
            import pyqtgraph as pg  # type: ignore
            from pyqtgraph.Qt import QtWidgets  # type: ignore
        except Exception as e:
            print('GUI disabled (pyqtgraph not available):', e)
            config.USE_GUI = False
    if config.USE_GUI:
        app = QtWidgets.QApplication([])
        view = pg.GraphicsView()
        layout = pg.GraphicsLayout(border=(100, 100, 100))
        view.setCentralItem(layout)
        view.show()
        view.setWindowTitle('Visualization')
        view.resize(800, 600)
        fft_plot = layout.addPlot(title='Filterbank Output', colspan=3)
        fft_plot.setRange(yRange=[-0.1, 1.2])
        fft_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        x_data = np.array(range(1, config.N_FFT_BINS + 1))
        mel_curve = cast(Curve, pg.PlotCurveItem())
        mel_curve.setData(x=x_data, y=x_data * 0)
        fft_plot.addItem(mel_curve)
        layout.nextRow()
        led_plot = layout.addPlot(title='Visualization Output', colspan=3)
        led_plot.setRange(yRange=[-5, 260])
        led_plot.disableAutoRange(axis=pg.ViewBox.YAxis)
        r_pen = pg.mkPen((255, 30, 30, 200), width=4)
        g_pen = pg.mkPen((30, 255, 30, 200), width=4)
        b_pen = pg.mkPen((30, 30, 255, 200), width=4)
        r_curve = cast(Curve, pg.PlotCurveItem(pen=r_pen))
        g_curve = cast(Curve, pg.PlotCurveItem(pen=g_pen))
        b_curve = cast(Curve, pg.PlotCurveItem(pen=b_pen))
        x_led = np.array(range(1, config.N_PIXELS + 1))
        r_curve.setData(x=x_led, y=x_led * 0)
        g_curve.setData(x=x_led, y=x_led * 0)
        b_curve.setData(x=x_led, y=x_led * 0)
        led_plot.addItem(r_curve)
        led_plot.addItem(g_curve)
        led_plot.addItem(b_curve)
        freq_label = pg.LabelItem('')

        def freq_slider_change(tick):
            minf = freq_slider.tickValue(0) ** 2.0 * (config.MIC_RATE / 2.0)
            maxf = freq_slider.tickValue(1) ** 2.0 * (config.MIC_RATE / 2.0)
            freq_label.setText('Frequency range: {:.0f} - {:.0f} Hz'.format(minf, maxf))
            config.MIN_FREQUENCY = minf
            config.MAX_FREQUENCY = maxf
            dsp.create_mel_bank()

        freq_slider = pg.TickSliderItem(orientation='bottom', allowAdd=False)
        freq_slider.addTick((config.MIN_FREQUENCY / (config.MIC_RATE / 2.0)) ** 0.5)
        freq_slider.addTick((config.MAX_FREQUENCY / (config.MIC_RATE / 2.0)) ** 0.5)
        freq_slider.tickMoveFinished = freq_slider_change
        freq_label.setText('Frequency range: {} - {} Hz'.format(
            config.MIN_FREQUENCY,
            config.MAX_FREQUENCY))

        active_color = '#16dbeb'
        inactive_color = '#FFFFFF'

        def energy_click(x):
            set_effect('energy')
            pg.LabelItem.setText(energy_label, 'Energy', color=active_color)
            pg.LabelItem.setText(scroll_label, 'Scroll', color=inactive_color)
            pg.LabelItem.setText(spectrum_label, 'Spectrum', color=inactive_color)

        def scroll_click(x):
            set_effect('scroll')
            pg.LabelItem.setText(energy_label, 'Energy', color=inactive_color)
            pg.LabelItem.setText(scroll_label, 'Scroll', color=active_color)
            pg.LabelItem.setText(spectrum_label, 'Spectrum', color=inactive_color)

        def spectrum_click(x):
            set_effect('spectrum')
            pg.LabelItem.setText(energy_label, 'Energy', color=inactive_color)
            pg.LabelItem.setText(scroll_label, 'Scroll', color=inactive_color)
            pg.LabelItem.setText(spectrum_label, 'Spectrum', color=active_color)

        energy_label = pg.LabelItem('Energy')
        scroll_label = pg.LabelItem('Scroll')
        spectrum_label = pg.LabelItem('Spectrum')
        energy_label.mousePressEvent = energy_click
        scroll_label.mousePressEvent = scroll_click
        spectrum_label.mousePressEvent = spectrum_click
        if args.effect == 'energy':
            energy_click(0)
        elif args.effect == 'scroll':
            scroll_click(0)
        else:
            spectrum_click(0)

        layout.nextRow()
        layout.addItem(freq_label, colspan=3)
        layout.nextRow()
        layout.addItem(freq_slider, colspan=3)
        layout.nextRow()
        layout.addItem(energy_label)
        layout.addItem(scroll_label)
        layout.addItem(spectrum_label)
    # Initialize LEDs
    led.update()
    try:
        microphone.start_stream(microphone_update)
    except KeyboardInterrupt:
        print('Stopping visualization...')
        led.pixels = np.zeros((3, config.N_PIXELS))
        led.update()
    return 0

# Replace original __main__ block with main()
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
