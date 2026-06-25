import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# Full path to your MP3 file
audio_file = r"C:\Users\sukhv\Downloads\song.mp3"

# Check if file exists
if not os.path.exists(audio_file):
    print("Error: File not found!")
    exit()

# Load the audio
audio, fs = librosa.load(audio_file, sr=None, mono=True)

print(f"Sampling Frequency: {fs} Hz")
print(f"Number of Samples: {len(audio)}")
print(f"Duration: {len(audio)/fs:.2f} seconds")

# Plot waveform
time = np.arange(len(audio)) / fs

plt.figure(figsize=(12,4))
plt.plot(time, audio)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Compute FFT
X = np.fft.fft(audio)

# Frequency axis
freq = np.fft.fftfreq(len(audio), d=1/fs)

# Magnitude spectrum
magnitude = np.abs(X)

# Plot FFT
plt.figure(figsize=(12,5))
plt.plot(freq[:len(freq)//2], magnitude[:len(magnitude)//2])
plt.title("DFT Magnitude Spectrum of Entire Song")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
