import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter

# ============================================================
# CHANGE THIS PATH TO YOUR SONG FOLDER
# ============================================================

SONG_FOLDER = r"C:\Users\sukhv\Downloads\songs"

# ============================================================
# PARAMETERS
# ============================================================

WINDOW_SIZE = 1024
OVERLAP = 512

PEAK_PERCENTILE = 98      # Keep strongest 2% peaks
FAN_VALUE = 5             # Number of neighbouring peaks

database = []

# ============================================================
# PROCESS EACH SONG
# ============================================================

song_files = [f for f in os.listdir(SONG_FOLDER) if f.endswith(".mp3")]

print(f"\nTotal Songs Found : {len(song_files)}\n")

for count, filename in enumerate(song_files, start=1):

    print(f"[{count}/{len(song_files)}] Processing : {filename}")

    filepath = os.path.join(SONG_FOLDER, filename)

    # --------------------------------------------------------
    # Load Song
    # --------------------------------------------------------

    audio, fs = librosa.load(filepath, sr=None, mono=True)

    # --------------------------------------------------------
    # Compute Spectrogram
    # --------------------------------------------------------

    frequencies, times, Sxx = spectrogram(
        audio,
        fs=fs,
        window='hann',
        nperseg=WINDOW_SIZE,
        noverlap=OVERLAP
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # --------------------------------------------------------
    # Detect Strong Peaks
    # --------------------------------------------------------

    local_max = maximum_filter(Sxx_db, size=(20,20))

    threshold = np.percentile(Sxx_db, PEAK_PERCENTILE)

    peaks = (Sxx_db == local_max) & (Sxx_db >= threshold)

    peak_freq, peak_time = np.where(peaks)

    # --------------------------------------------------------
    # Generate Fingerprints
    # --------------------------------------------------------

    for i in range(len(peak_time)):

        for j in range(1, FAN_VALUE + 1):

            if i + j >= len(peak_time):
                break

            f1 = int(frequencies[peak_freq[i]])
            f2 = int(frequencies[peak_freq[i + j]])

            t1 = float(times[peak_time[i]])
            t2 = float(times[peak_time[i + j]])

            dt = round(t2 - t1, 3)

            if dt <= 0:
                continue

            # Create Hash
            fingerprint_hash = f"{f1}_{f2}_{dt}"

            database.append({

                "Hash": fingerprint_hash,
                "Song": os.path.splitext(filename)[0],
                "Frequency1": f1,
                "Frequency2": f2,
                "TimeDifference": dt,
                "AnchorTime": round(t1,3)

            })

# ============================================================
# SAVE DATABASE
# ============================================================

df = pd.DataFrame(database)

output_file = "fingerprint_database.csv"

df.to_csv(output_file, index=False)

print("\n======================================")
print("DATABASE CREATED SUCCESSFULLY")
print("======================================")
print(f"Songs Processed      : {len(song_files)}")
print(f"Total Fingerprints   : {len(df)}")
print(f"Database Saved As    : {output_file}")
print("======================================")

print("\nFirst 10 Database Entries:\n")

print(df.head(10))
