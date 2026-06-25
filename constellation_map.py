import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter

# ===========================================
# Load Audio
# ===========================================
audio_file = r"C:\Users\sukhv\Downloads\song.mp3"

audio, fs = librosa.load(audio_file, sr=None, mono=True)

# ===========================================
# Compute Spectrogram
# ===========================================
frequencies, times, Sxx = spectrogram(
    audio,
    fs=fs,
    window='hann',
    nperseg=1024,
    noverlap=512
)

# Convert to dB
Sxx_db = 10 * np.log10(Sxx + 1e-10)

# ===========================================
# Detect Strong Local Maxima
# ===========================================

# Local maxima within a 20×20 neighborhood
local_max = maximum_filter(Sxx_db, size=(20,20))

# Keep only strongest peaks (top 2%)
threshold = np.percentile(Sxx_db, 98)

peaks = (Sxx_db == local_max) & (Sxx_db >= threshold)

peak_freq, peak_time = np.where(peaks)

# ===========================================
# Plot Spectrogram + Constellation Map
# ===========================================

plt.figure(figsize=(14,6))

plt.imshow(
    Sxx_db,
    origin='lower',
    aspect='auto',
    extent=[times.min(), times.max(),
            frequencies.min(), frequencies.max()],
    cmap='magma'
)

plt.scatter(
    times[peak_time],
    frequencies[peak_freq],
    facecolors='none',
    edgecolors='cyan',
    s=25,
    linewidths=1
)

plt.title("Constellation Map (Detected Strong Peaks)")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.ylim(0,3500)

plt.tight_layout()
plt.show()

# ===========================================
# Generate Fingerprints
# ===========================================

fingerprints = []

fan_value = 5

for i in range(len(peak_time)):
    for j in range(1, fan_value):

        if i + j >= len(peak_time):
            break

        f1 = int(frequencies[peak_freq[i]])
        f2 = int(frequencies[peak_freq[i+j]])

        t1 = times[peak_time[i]]
        t2 = times[peak_time[i+j]]

        dt = round(t2 - t1, 2)

        if dt > 0:
            fingerprints.append((f1, f2, dt))

print("\nTotal Fingerprints Generated:", len(fingerprints))

print("\nFirst 20 Fingerprints:\n")

for fp in fingerprints[:20]:
    print(fp)
