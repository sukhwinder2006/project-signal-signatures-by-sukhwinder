import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter

# ============================================================
# SONG FOLDER
# ============================================================

SONG_FOLDER = r"C:\Users\sukhv\Downloads\songs"

# ============================================================
# PARAMETERS
# ============================================================

WINDOW_SIZE = 1024
OVERLAP = 512

PEAK_PERCENTILE = 98
FAN_VALUE = 5

database = []

# ============================================================
# GET ALL SONGS
# ============================================================

song_files = [f for f in os.listdir(SONG_FOLDER) if f.endswith(".mp3")]

print(f"\nTotal Songs Found : {len(song_files)}\n")

# ============================================================
# PROCESS EVERY SONG
# ============================================================

for count, filename in enumerate(song_files, start=1):

    print(f"[{count}/{len(song_files)}] Processing : {filename}")

    filepath = os.path.join(SONG_FOLDER, filename)

    # --------------------------------------------------------
    # Load Audio
    # --------------------------------------------------------

    audio, fs = librosa.load(filepath, sr=None, mono=True)

    # --------------------------------------------------------
    # Spectrogram
    # --------------------------------------------------------

    frequencies, times, Sxx = spectrogram(
        audio,
        fs=fs,
        window="hann",
        nperseg=WINDOW_SIZE,
        noverlap=OVERLAP
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # --------------------------------------------------------
    # Peak Detection
    # --------------------------------------------------------

    local_max = maximum_filter(Sxx_db, size=(20, 20))

    threshold = np.percentile(Sxx_db, PEAK_PERCENTILE)

    peaks = (Sxx_db == local_max) & (Sxx_db >= threshold)

    peak_freq_idx, peak_time_idx = np.where(peaks)

    # --------------------------------------------------------
    # Sort peaks according to time
    # --------------------------------------------------------

    peak_points = sorted(zip(peak_time_idx, peak_freq_idx))

    peak_time_idx = [p[0] for p in peak_points]
    peak_freq_idx = [p[1] for p in peak_points]

    # --------------------------------------------------------
    # Fingerprint Generation
    # --------------------------------------------------------

    for i in range(len(peak_time_idx)):

        for j in range(1, FAN_VALUE + 1):

            if i + j >= len(peak_time_idx):
                break

            f1 = int(frequencies[peak_freq_idx[i]])
            f2 = int(frequencies[peak_freq_idx[i + j]])

            t1 = float(times[peak_time_idx[i]])
            t2 = float(times[peak_time_idx[i + j]])

            dt = t2 - t1

            if dt <= 0:
                continue

            dt_ms = int(round(dt * 1000))

            fingerprint_hash = f"{f1}_{f2}_{dt_ms}"

            database.append({

                "Hash": fingerprint_hash,
                "Song": os.path.splitext(filename)[0],
                "Frequency1": f1,
                "Frequency2": f2,
                "TimeDifference": round(dt, 3),
                "AnchorTime": round(t1, 3)

            })

# ============================================================
# SAVE DATABASE
# ============================================================

df = pd.DataFrame(database)

df.to_csv("fingerprint_database.csv", index=False)

print("\n====================================")
print("DATABASE CREATED SUCCESSFULLY")
print("====================================")
print("Songs Processed :", len(song_files))
print("Fingerprints    :", len(df))
print("Saved As        : fingerprint_database.csv")
print("====================================")

print("\nSample Entries:\n")
print(df.head())
