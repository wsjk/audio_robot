import numpy as np
import config
import melbank


class ExpFilter:
    """Exponential smoothing filter"""
    def __init__(self, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
        assert 0.0 < alpha_decay < 1.0, 'Invalid decay factor'
        assert 0.0 < alpha_rise < 1.0, 'Invalid rise factor'
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val

    def update(self, value):
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = np.where(value > self.value, self.alpha_rise, self.alpha_decay)
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1 - alpha) * self.value
        return self.value


def rfft(data, window=None):
    window = window(len(data)) if window else 1.0
    ys = np.abs(np.fft.rfft(data * window))
    xs = np.fft.rfftfreq(len(data), 1 / config.MIC_RATE)
    return xs, ys


def fft(data, window=None):
    window = window(len(data)) if window else 1.0
    ys = np.fft.fft(data * window)
    xs = np.fft.fftfreq(len(data), 1 / config.MIC_RATE)
    return xs, ys


def create_mel_bank():
    global samples, mel_y, mel_x
    samples = int(config.MIC_RATE * config.N_ROLLING_HISTORY / (2 * config.FPS))
    mel_y, (_, mel_x) = melbank.compute_melmat(
        num_mel_bands=config.N_FFT_BINS,
        freq_min=config.MIN_FREQUENCY,
        freq_max=config.MAX_FREQUENCY,
        num_fft_bands=samples,
        sample_rate=config.MIC_RATE
    )


samples, mel_y, mel_x = None, None, None
create_mel_bank()
