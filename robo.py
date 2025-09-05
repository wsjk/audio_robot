import pyaudio
import numpy as np
import time
from .servo import *
from .audio import *

alpha = 0.3     # smoothing
decay = 0.9     # falling speed
smooth_energy = {name: 0 for name in BANDS}
servo_level = {name: 0 for name in BANDS}  # store -1..1 position

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("Servo VU meter with gpiozero running... (Ctrl+C to stop)")

def band_energy(magnitude, freqs, low, high):
    idx = np.where((freqs >= low) & (freqs <= high))
    return np.sum(magnitude[idx])

def log_scale(value, base=10):
    return np.log1p(value) / np.log(base)

try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)

        fft = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(fft), 1.0 / RATE)
        magnitude = np.abs(fft)

        for name, (low, high) in BANDS.items():
            energy = band_energy(magnitude, freqs, low, high)

            # Smooth energy
            smooth_energy[name] = alpha * energy + (1 - alpha) * smooth_energy[name]

            # Log scaling + normalization
            scaled = log_scale(smooth_energy[name])
            norm = scaled / log_scale(THRESHOLDS[name])

            # Target level mapped to -1..+1 (servo range)
            target_level = np.clip(norm, 0, 1) * 2 - 1

            # Bounce effect: rise instantly, fall gradually
            if target_level > servo_level[name]:
                servo_level[name] = target_level
            else:
                servo_level[name] *= decay

            # Send to servo
            servos[name].value = servo_level[name]

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping...")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    for s in servos.values():
        s.detach()