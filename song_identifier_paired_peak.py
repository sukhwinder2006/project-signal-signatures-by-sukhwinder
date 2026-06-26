import librosa
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict

# ============================================================
# ENTER QUERY SONG PATH
# ============================================================

QUERY_SONG = r"C:\Users\sukhv\Downloads\query.mp3"

# ============================================================
# PARAMETERS (SAME AS DATABASE)
# ============================================================

WINDOW_SIZE = 1024
OVERLAP = 512
PEAK_PERCENTILE = 98
FAN_VALUE = 5

print("\nLoading Query Song...\n")

# ============================================================
# LOAD QUERY AUDIO
# ============================================================

audio, fs = librosa.load(QUERY_SONG, sr=None, mono=True)

print("Song Loaded Successfully!")

# ============================================================
# CREATE SPECTROGRAM
# ============================================================

frequencies, times, Sxx = spectrogram(
    audio,
    fs=fs,
    window='hann',
    nperseg=WINDOW_SIZE,
    noverlap=OVERLAP
)

Sxx_db = 10 * np.log10(Sxx + 1e-10)

print("Spectrogram Generated!")

# ============================================================
# FIND LOCAL PEAKS
# ============================================================

local_max = maximum_filter(Sxx_db, size=(20,20))

threshold = np.percentile(Sxx_db, PEAK_PERCENTILE)

peaks = (Sxx_db == local_max) & (Sxx_db >= threshold)

peak_freq_idx, peak_time_idx = np.where(peaks)

# ============================================================
# SORT PEAKS BY TIME
# ============================================================

peak_points = sorted(zip(peak_time_idx, peak_freq_idx))

peak_time_idx = [p[0] for p in peak_points]
peak_freq_idx = [p[1] for p in peak_points]

print("Total Peaks :", len(peak_time_idx))

# ============================================================
# GENERATE QUERY FINGERPRINTS
# ============================================================

query_fingerprints = []

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

        query_fingerprints.append((fingerprint_hash, t1))

print("Query Fingerprints Generated :", len(query_fingerprints))
# ============================================================
# LOAD DATABASE
# ============================================================

print("\nLoading Fingerprint Database...\n")

database = pd.read_csv("fingerprint_database.csv")

print("Database Loaded Successfully!")
print("Total Database Fingerprints :", len(database))

# ============================================================
# CREATE HASH LOOKUP TABLE
# ============================================================

print("\nCreating Hash Lookup Table...")

hash_table = defaultdict(list)

for _, row in database.iterrows():

    hash_table[row["Hash"]].append(
        (
            row["Song"],
            row["AnchorTime"]
        )
    )

print("Hash Table Created!")

# ============================================================
# MATCH QUERY WITH DATABASE
# ============================================================

print("\nSearching for Matching Fingerprints...\n")

matches = []

for fingerprint_hash, query_time in query_fingerprints:

    if fingerprint_hash in hash_table:

        for song_name, song_time in hash_table[fingerprint_hash]:

            offset = round(song_time - query_time, 2)

            matches.append(
                (
                    song_name,
                    offset
                )
            )

print("Total Matching Fingerprints :", len(matches))
from collections import Counter

# ============================================================
# FIND THE BEST MATCH
# ============================================================

print("\nCalculating Best Match...\n")

vote_counter = Counter(matches)

if len(vote_counter) == 0:
    print("No matching song found!")
    exit()

# Count votes for each (Song, Offset)
best_match, best_votes = vote_counter.most_common(1)[0]

matched_song = best_match[0]
best_offset = best_match[1]

print("========================================")
print("SONG IDENTIFICATION RESULT")
print("========================================")
print(f"Matched Song          : {matched_song}")
print(f"Most Common Offset    : {best_offset:.2f} seconds")
print(f"Matching Fingerprints : {best_votes}")
print("========================================")
# ==========================================
# OFFSET HISTOGRAM
# ==========================================

import matplotlib.pyplot as plt

matched_offsets = []

for song, offset in matches:
    if song == matched_song:
        matched_offsets.append(offset)

plt.figure(figsize=(10,5))
plt.hist(matched_offsets, bins=40, edgecolor="black")

plt.title(f"Offset Histogram - {matched_song}")
plt.xlabel("Offset (seconds)")
plt.ylabel("Number of Matching Fingerprints")
plt.grid(True)

plt.show()

# ============================================================
# OPTIONAL : SHOW TOP 5 MATCHES
# ============================================================

print("\nTop 5 Candidate Songs:\n")

song_counter = Counter()

for song, offset in matches:
    song_counter[song] += 1

for song, votes in song_counter.most_common(5):
    print(f"{song:35} {votes} matches")

print("\nRecognition Completed Successfully!")
