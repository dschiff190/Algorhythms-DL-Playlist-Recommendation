"""Preprocess [FINAL - 4 Dataset, Batched Playlists].py"""

import kagglehub
import os
import pandas as pd
import ast
import json
import re
import tensorflow as tf
import gc
from pathlib import Path

# -----------------------------
#  PREPARE HELPER-FUNCTIONS
# -----------------------------

# Core numeric features we care about
core_features = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms", "time_signature",
]

# Full canonical column list
canonical_cols = [
    "track_id", "track_name", "artist_name",
    "genre", "year", "popularity", "explicit",
] + core_features


def find_first_csv(root_path: str):
    """Return the first .csv path under a Kaggle dataset folder."""
    for r, dirs, files in os.walk(root_path):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(r, f)
    return None


def normalize_artists(x):
    """
    Normalize various artist formats into 'A;B;C':
      - "['A', 'B']" -> "A;B"
      - "A,B" or "A;B" -> "A;B"
    """
    if pd.isna(x):
        return None
    s = str(x).strip()

    # Only try literal_eval if it looks like a list/tuple
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                parts = [str(a).strip() for a in parsed if a]
                return ";".join(parts) if parts else None
        except (ValueError, SyntaxError):
            pass

    # Fallback: split on ';' and ',' and re-join with ';' and no spaces
    parts = [p.strip() for chunk in s.split(";") for p in chunk.split(",")]
    parts = [p for p in parts if p]
    return ";".join(parts) if parts else None


def align_to_canonical(df):
    """Ensure all canonical columns exist and are ordered."""
    for col in canonical_cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df[canonical_cols]

# -----------------------------
# DOWNLOAD & EXTRACT SONGS
# -----------------------------

# 1) 12M songs
path_12m = kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs")
print("12M path:", path_12m)

# 2) 114k tracks
path_114k = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
print("114k path:", path_114k)

# 3) 600k tracks
path_600k = kagglehub.dataset_download("yamaerenay/spotify-dataset-19212020-600k-tracks")
print("600k path:", path_600k)

# 4) 1M tracks
path_1m = kagglehub.dataset_download("amitanshjoshi/spotify-1million-tracks")
print("1M path:", path_1m)

# Known / discovered CSV paths
csv_12m = os.path.join(path_12m, "tracks_features.csv")  # known name
csv_114k = find_first_csv(path_114k)                     # dataset.csv
csv_600k = os.path.join(path_600k, "tracks.csv")         # known name
csv_1m   = find_first_csv(path_1m)                       # spotify_data.csv

print("\nCSV paths:")
print("12M :", csv_12m)
print("114k:", csv_114k)
print("600k:", csv_600k)
print("1M  :", csv_1m)

# -----------------------------------------------
# STANDARDIZE SONGS & FILL IN OPTIONAL COLUMNS
# -----------------------------------------------

# ---- Dataset 1: 12M ----
df1 = pd.read_csv(csv_12m)

df1 = df1.rename(columns={"id": "track_id", "name": "track_name"})
df1["track_id"] = df1["track_id"].astype(str)

# year already in df1; if not, fallback from release_date
if "year" not in df1.columns and "release_date" in df1.columns:
    df1["year"] = pd.to_datetime(df1["release_date"], errors="coerce").dt.year

# no genre / popularity in this dataset
df1["genre"] = pd.NA
df1["popularity"] = pd.NA

# normalize artists -> artist_name
df1["artist_name"] = df1["artists"].apply(normalize_artists)

cols_df1 = [
    "track_id", "track_name", "artist_name",
    "genre", "year", "popularity", "explicit",
] + core_features
df1 = df1[[c for c in cols_df1 if c in df1.columns]]
df1 = align_to_canonical(df1)

# ---- Dataset 2: 114k ----
df2 = pd.read_csv(csv_114k)

df2["track_id"] = df2["track_id"].astype(str)

# track_genre -> genre
if "track_genre" in df2.columns:
    df2["genre"] = df2["track_genre"]

# no year column
df2["year"] = pd.NA

# normalize artists -> artist_name
df2["artist_name"] = df2["artists"].apply(normalize_artists)

cols_df2 = [
    "track_id", "track_name", "artist_name",
    "genre", "year", "popularity", "explicit",
] + core_features
df2 = df2[[c for c in cols_df2 if c in df2.columns]]
df2 = align_to_canonical(df2)

# ---- Dataset 3: 600k ----
df3 = pd.read_csv(csv_600k)

df3 = df3.rename(columns={"id": "track_id", "name": "track_name"})
df3["track_id"] = df3["track_id"].astype(str)

# derive year from release_date
if "release_date" in df3.columns:
    df3["year"] = pd.to_datetime(df3["release_date"], errors="coerce").dt.year
else:
    df3["year"] = pd.NA

# no genre column
df3["genre"] = pd.NA

# normalize artists -> artist_name
df3["artist_name"] = df3["artists"].apply(normalize_artists)

cols_df3 = [
    "track_id", "track_name", "artist_name",
    "genre", "year", "popularity", "explicit",
] + core_features
df3 = df3[[c for c in cols_df3 if c in df3.columns]]
df3 = align_to_canonical(df3)

# ---- Dataset 4: 1M ----
df4 = pd.read_csv(csv_1m)

df4["track_id"] = df4["track_id"].astype(str)

# 1M has genre, year, popularity
if "genre" not in df4.columns:
    df4["genre"] = pd.NA
if "year" not in df4.columns:
    df4["year"] = pd.NA
if "popularity" not in df4.columns:
    df4["popularity"] = pd.NA

# 1M has no explicit -> set NaN (optional feature)
if "explicit" not in df4.columns:
    df4["explicit"] = pd.NA

# normalize artist_name
df4["artist_name"] = df4["artist_name"].apply(normalize_artists)

cols_df4 = [
    "track_id", "track_name", "artist_name",
    "genre", "year", "popularity", "explicit",
] + core_features
df4 = df4[[c for c in cols_df4 if c in df4.columns]]
df4 = align_to_canonical(df4)

print("Shapes after standardization:")
print("12M :", df1.shape)
print("114k:", df2.shape)
print("600k:", df3.shape)
print("1M  :", df4.shape)

# -----------------------------------------------
# MERGE ALL SONG DATASETS INTO master_tracks
# -----------------------------------------------

# Prefer 1M > 114k > 600k > 12M when track_id collides
master_tracks = pd.concat([df4, df2, df3, df1], ignore_index=True)

print("Raw combined rows:", len(master_tracks))
master_tracks.drop_duplicates(subset=["track_id"], inplace=True)
print("Unique track_ids in master_tracks:", len(master_tracks))

# ---------------------------------------------------
# ENFORCE SPOTIFY API CONSTRAINTS (pre-imputation)
# ---------------------------------------------------
def mask_range(df, col, lower=None, upper=None, allowed_values=None):
    """Helper: mask out-of-range values to NaN."""
    if col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce")
    if allowed_values is not None:
        s = s.mask(~s.isin(allowed_values))
    else:
        if lower is not None:
            # Special case: lower==0 → mask <= 0 (used for tempo/duration)
            s = s.mask(s <= lower) if lower == 0 else s.mask(s < lower)
        if upper is not None:
            s = s.mask(s > upper)
    df[col] = s

# loudness: typical range [-60, 0] dB
mask_range(master_tracks, "loudness", lower=-60.0, upper=0.0)

# tempo: anything > 0 is fine, just kill zeros / negatives
mask_range(master_tracks, "tempo", lower=0.0)  # masks tempo <= 0

# duration_ms: anything > 0 is fine, just kill zeros / negatives
mask_range(master_tracks, "duration_ms", lower=0)  # masks duration_ms <= 0

# year: clamp anything < 1900 up to 1900
if "year" in master_tracks.columns:
    year_numeric = pd.to_numeric(master_tracks["year"], errors="coerce")

    # Clamp all non-missing years below 1900 (including 0, negatives) to 1900
    year_clamped = year_numeric.copy()
    year_clamped.loc[year_clamped.notna() & (year_clamped < 1900)] = 1900

    master_tracks["year"] = year_clamped

# time signature - filter to 3 and 7 strictly
if "time_signature" in master_tracks.columns:
    ts = pd.to_numeric(master_tracks["time_signature"], errors="coerce")

    # For valid values (non-NaN), clamp:
    # values < 3 → 3, values > 7 → 7
    ts_clamped = ts.copy()
    ts_clamped.loc[ts_clamped < 3] = 3
    ts_clamped.loc[ts_clamped > 7] = 7

    master_tracks["time_signature"] = ts_clamped

print("\nMin/max AFTER masking (skip NaNs):")
for col in ["time_signature", "loudness", "tempo", "duration_ms", "year"]:
    if col in master_tracks.columns:
        s = master_tracks[col]
        print(
            f"  {col}: "
            f"min={s.min(skipna=True)}, "
            f"max={s.max(skipna=True)}, "
            f"NaNs={s.isna().sum()}"
        )
        
# ----------------------------------------------------------------------
# NUMERIC CLEANUP + OPTIONAL FEATURE IS_MISSING COLS + GENRE ONE-HOT
# ----------------------------------------------------------------------

numeric_optional = ["year", "popularity", "explicit"]

# 1) Normalize explicit to 0/1/NaN
if "explicit" in master_tracks.columns:
    exp = master_tracks["explicit"]
    master_tracks["explicit"] = (
        exp.map({True: 1, False: 0})   # handle booleans
           .fillna(exp)               # keep existing 0/1 if already numeric
    )
    master_tracks["explicit"] = pd.to_numeric(master_tracks["explicit"], errors="coerce")

# 2) Cast ALL numeric-ish features to float
numeric_base = core_features + numeric_optional
for col in numeric_base:
    if col in master_tracks.columns:
        master_tracks[col] = pd.to_numeric(
            master_tracks[col],
            errors="coerce"
        ).astype("float64")

# 3) For OPTIONAL numeric features: create *_filled + *_is_missing
for col in numeric_optional:
    if col not in master_tracks.columns:
        continue

    col_float = master_tracks[col]

    # Missing mask: 0 = real value, 1 = missing
    master_tracks[col + "_is_missing"] = col_float.isna().astype(int)

    # Fill strategy:
    if col == "explicit":
        # For explicit, use fixed default: assume not explicit (=0.0)
        fill_val = 0.0
    else:
        # For year & popularity, use mean imputation
        fill_val = col_float.mean(skipna=True)

    master_tracks[col + "_filled"] = col_float.fillna(fill_val)

# 4) One-hot encode genre; NaN → all zeros in dummy columns, as floats
genre_dummies = pd.get_dummies(
    master_tracks["genre"],
    prefix="genre",
    dummy_na=False
).astype("float64")

master_tracks = pd.concat([master_tracks, genre_dummies], axis=1)

# 5) Drop raw optional columns + raw genre so we only keep *_filled + *_is_missing
cols_to_drop = [c for c in ["year", "popularity", "explicit", "genre"] if c in master_tracks.columns]
master_tracks = master_tracks.drop(columns=cols_to_drop)

print("master_tracks shape after feature engineering:", master_tracks.shape)

# -----------------------------
# TRACK-LEVEL MIN–MAX NORMALIZATION
# -----------------------------

# Continuous numeric features to normalize
optional_numeric_filled = [
    "year_filled",
    "popularity_filled",
]

numeric_to_normalize = core_features + optional_numeric_filled

scale_stats = {}
for col in numeric_to_normalize:
    if col not in master_tracks.columns:
        continue
    col_vals = master_tracks[col].astype("float32")
    c_min = col_vals.min()
    c_max = col_vals.max()
    if c_max == c_min:
        master_tracks[col] = 0.0
    else:
        master_tracks[col] = (col_vals - c_min) / (c_max - c_min)
    scale_stats[col] = (float(c_min), float(c_max))

print("Min–max normalized (track-level):", numeric_to_normalize)

# Masks & genre one-hots (stay as 0/1, no normalization)
optional_masks = [
    "year_is_missing",
    "popularity_is_missing",
    "explicit_is_missing",
]
genre_one_hot_cols = [c for c in master_tracks.columns if c.startswith("genre_")]

# ----------------------------------------------------------------------
# DOWNLOAD 1-MILLION PLAYLIST & SORT SLICE FILES NUMERICALLY
# ----------------------------------------------------------------------

mpl_path = kagglehub.dataset_download("himanshuwagh/spotify-million")
print("Million playlist dataset path:", mpl_path)
mpl_path = Path(mpl_path)

# Grab all json files; then filter down to just the mpd slices
json_files = list(mpl_path.rglob("*.json"))
slice_files = [p for p in json_files if "mpd.slice." in p.name]

print("Number of slice files before sort:", len(slice_files))

# Regex to pull the numeric start of the range: mpd.slice.1000-1999.json -> 1000
pattern = re.compile(r"mpd\.slice\.(\d+)-(\d+)\.json")

def slice_key(path):
    m = pattern.match(path.name)
    if m:
        return int(m.group(1))  # sort by the starting index
    # fallback if something weird slips in
    return float("inf")

# Sort numerically by the slice start index
slice_files = sorted(slice_files, key=slice_key)

print("First 5 slice files after numeric sort:")
for p in slice_files[:5]:
    print(p.name)

print("Last 5 slice files after numeric sort:")
for p in slice_files[-5:]:
    print(p.name)

# Use ALL slices (1000)
slice_files_small = slice_files
print("Total slice files used:", len(slice_files_small))

# ----------------------------------------------------------------------
# BATCHED FLATTEN + JOIN + FILTER + SAVE
# ----------------------------------------------------------------------

# Features we REQUIRE to be present (for training)
feature_cols_required = core_features + [
    "year_filled",
    "popularity_filled",
    "explicit_filled"
]

BATCH_SIZE = 20  # number of slice files to process at once
out_path = "playlist_song_features_FINAL_FULL.csv"
first_batch = True
total_rows_written = 0

for start_idx in range(0, len(slice_files_small), BATCH_SIZE):
    batch_paths = slice_files_small[start_idx:start_idx + BATCH_SIZE]
    print(f"\nProcessing slice batch {start_idx}–{start_idx + len(batch_paths) - 1} ...")

    # 1) Build a local playlist–track DataFrame for this batch
    rows = []
    for path in batch_paths:
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)

        for pl in blob["playlists"]:
            pid = pl["pid"]
            title = pl.get("name", "")

            for tr in pl["tracks"]:
                track_uri = tr["track_uri"]          # e.g. 'spotify:track:4qPNDBW1i3p13qLCt0Ki3A'
                track_id = track_uri.split(":")[-1]  # '4qPNDBW1i3p13qLCt0Ki3A'

                rows.append({
                    "playlist_id": pid,
                    "playlist_name": title,
                    "track_id": track_id,
                })

    pl_batch = pd.DataFrame(rows)
    pl_batch["track_id"] = pl_batch["track_id"].astype(str)
    print("  batch playlist–track rows:", len(pl_batch))

    # 2) Join with normalized master_tracks
    full_batch = pl_batch.merge(
        master_tracks,
        how="left",
        on="track_id"
    )
    print("  joined rows:", len(full_batch))

    # 3) Drop rows with missing required features
    filtered_batch = full_batch.dropna(subset=feature_cols_required).copy()
    print("  kept rows after dropna:", len(filtered_batch))

    # 4) Reorder / keep only the columns you care about for the model
    cols_for_output = (
        ["playlist_id", "playlist_name", "track_id", "track_name", "artist_name"] +
        feature_cols_required +
        optional_masks +
        genre_one_hot_cols
    )
    cols_for_output = [c for c in cols_for_output if c in filtered_batch.columns]
    filtered_batch = filtered_batch[cols_for_output]

    # 5) Append to CSV on disk
    filtered_batch.to_csv(
        out_path,
        mode="w" if first_batch else "a",
        header=first_batch,
        index=False,
    )
    first_batch = False
    total_rows_written += len(filtered_batch)
    print(f"  wrote {len(filtered_batch)} rows (total so far: {total_rows_written})")

    # 6) Free memory
    del rows, pl_batch, full_batch, filtered_batch
    gc.collect()


print("\nDone! Final dataset written to:", out_path)
print("Total rows written:", total_rows_written)