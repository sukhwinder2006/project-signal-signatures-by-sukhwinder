import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# Load Audio File
audio_file = r"C:\Users\sukhv\Downloads\song.mp3"

audio, fs = librosa.load(audio_file, sr=None, mono=True)

# Compute Spectrogram with a Short Window
frequencies, times, Sxx = spectrogram(
    audio,
    fs=fs,
    window='hann',
    nperseg=256,      # Short window
    noverlap=128      # 50% overlap
)

# Convert Power to dB
Sxx_db = 10 * np.log10(Sxx + 1e-10)

# Plot Spectrogram
plt.figure(figsize=(12,6))
plt.pcolormesh(
    times,
    frequencies,
    Sxx_db,
    shading='gouraud',
    cmap='viridis'
)

plt.title("Spectrogram with Short Window (256 Samples)")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")
plt.ylim(0, 8000)

plt.tight_layout()
plt.show()
