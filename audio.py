# --- Audio Setup ---
CHUNK = 1024
RATE = 44100

BANDS = {
    "bass": (20, 250),
    "mid": (251, 2000),
    "treble": (2001, 8000)
}

THRESHOLDS = {
    "bass": 8e5,
    "mid": 6e5,
    "treble": 4e5
}

alpha = 0.3     # smoothing
decay = 0.9     # falling speed