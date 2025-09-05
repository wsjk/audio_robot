from gpiozero import Servo

# --- Servo Setup ---
# Adjust min/max pulse_width values for your specific servo
servos = {
    "bass": Servo(18, min_pulse_width=0.0005, max_pulse_width=0.0025),
    "mid": Servo(23, min_pulse_width=0.0005, max_pulse_width=0.0025),
    "treble": Servo(24, min_pulse_width=0.0005, max_pulse_width=0.0025),
}

