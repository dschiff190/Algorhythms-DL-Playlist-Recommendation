import pandas as pd
import tensorflow as tf
import numpy as np
import os
import datetime
import time
import pickle

from src.model import PlaylistModel, build_song_encoder
from src.train import encode_title


# ====================================================
# 2. DATA RECONSTRUCTION (The Heavy Part)
# ====================================================

print("⏳ Reading CSV (this may take a while)...")
start = time.perf_counter()
playlists = pd.read_csv("playlist_song_features_FINAL_FULL.csv", nrows=10000, engine="pyarrow")
print(f'read_csv took {start - time.perf_counter()}')

# 2a. Fix NaNs (must match training)
playlists["playlist_name"] = playlists["playlist_name"].fillna("").astype(str)

# ====================================================
# 2a.5 REBUILD FILLED COLUMNS (Required!)
# Must match training preprocessing for *_filled columns
# ====================================================

for col in playlists.columns:
    if col.endswith("_filled"):
        base = col.replace("_filled", "")
        if base in playlists.columns:
            # Fill using the exact same logic training used
            playlists[col] = playlists[base].fillna(0.0)
        else:
            # If base column does not exist, leave it as-is
            playlists[col] = playlists[col].fillna(0.0)

# ====================================================
# 2b. LOAD TRAINING VOCAB (unchanged from original)
# ====================================================
print("⏳ Reconstructing Vocabulary...")
K = 30000 
UNK_TOKEN = '<UNK>' 
UNK_ID = 0          

track_counts = playlists["track_id"].value_counts()
top_k_track_ids = track_counts.head(K).index.tolist()

# MUST match training order exactly
vocab_list = [UNK_TOKEN] + sorted(top_k_track_ids)
track2int = {tid: i for i, tid in enumerate(vocab_list)}
int2track = {i: tid for tid, i in track2int.items()}

VOCAB_SIZE = len(track2int)
print(f"✅ Vocab Reconstructed. Size: {VOCAB_SIZE}")

# ====================================================
# 2c. LOAD FEATURE COLS FROM TRAINING
# ====================================================
print("⏳ Loading feature column list used during training...")

import pickle
with open("data/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)

print(f"✅ Loaded {len(feature_cols)} feature columns (must match training exactly).")

# ====================================================
# 2d. REBUILD FEATURE MATRIX USING *THE SAME COLUMNS*
# ====================================================
print("⏳ Reconstructing Feature Dictionary with TRAINING columns...")

# Ensure all required columns exist (sanity check)
missing_cols = [c for c in feature_cols if c not in playlists.columns]
if missing_cols:
    raise ValueError(f"❌ Missing required feature columns: {missing_cols}")

# Select only the columns in the exact original order
track_features_df = (
    playlists
    .drop_duplicates(subset=['track_id'])
    .set_index('track_id')[feature_cols]
)

track2feat = {
    tid: vals.astype("float32")
    for tid, vals in zip(track_features_df.index, track_features_df.values)
}

FEATURE_DIM = len(feature_cols)
print(f"✅ Features Loaded. FEATURE_DIM = {FEATURE_DIM}")

# ====================================================
# 2e. Metadata lookup
# ====================================================
meta_df = playlists.drop_duplicates(subset=['track_id']).set_index('track_id')[['track_name', 'artist_name']]
track_meta = {
    tid: f"{row['track_name']} by {row['artist_name']}"
    for tid, row in meta_df.iterrows()
}

print("✅ Metadata reconstructed.")

# ====================================================
# 3. LOAD MODEL
# ====================================================

print("⏳ Loading Universal Sentence Encoder...")
USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def encode_title(title_texts):
    return USE(title_texts)

print("⏳ Initializing Model...")
EMB_DIM = 128
model = PlaylistModel(
    num_songs=VOCAB_SIZE,
    song_feat_dim=FEATURE_DIM,
    emb_dim=EMB_DIM
)

# Dummy call to build graph
dummy_feat = tf.zeros((1, 5, FEATURE_DIM))
dummy_title = encode_title(["dummy"])
dummy_mask = tf.ones((1, 5))
_ = model(dummy_feat, dummy_title, dummy_mask)

# Load Checkpoint
print("⏳ Restoring Weights...")
checkpoint_dir = "./playlist_model_ckpts"
checkpoint = tf.train.Checkpoint(optimizer=tf.keras.optimizers.Adam(3e-4), model=model)
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

if latest_checkpoint:
    # Expect partial because we don't care about optimizer state for inference
    status = checkpoint.restore(latest_checkpoint).expect_partial()
    print(f"✅ Model weights restored from {latest_checkpoint}")
else:
    print("❌ No checkpoint found. Check your directory path.")

# ====================================================
# 4. INFERENCE LOGIC
# ====================================================

def build_manual_batch(track_ids, track2feat, unk_id=0):
    valid_ids = [tid for tid in track_ids if (tid in track2int) and (tid in track2feat)]
    
    if not valid_ids:
        print("🚨 Error: None of the provided track IDs were found in the training vocabulary.")
        return None, None, None, None

    token_ids = [track2int.get(tid, unk_id) for tid in valid_ids]
    seq_len = len(token_ids)
    
    # Stack features
    feat_mat = np.stack([track2feat[tid] for tid in valid_ids], axis=0)
    
    song_features = tf.constant(feat_mat[None, :, :], dtype=tf.float32)
    mask = tf.ones((1, seq_len), dtype=tf.int32)
    playlist_song_ids = tf.constant(token_ids, dtype=tf.int64)
    
    return song_features, mask, playlist_song_ids

def autoregressive_recommend(
    model,
    song_features,
    title_text,
    mask,
    playlist_song_ids,
    k=3,
    num_generated=5,
    exclude_original=True  # NEW
):
    sf = tf.identity(song_features)
    m   = tf.identity(mask)
    ids = tf.identity(playlist_song_ids)

    # Keep a separate handle on the original playlist length
    orig_len = playlist_song_ids.shape[0]  # Python int in eager mode

    generated_ids = []
    
    print(f"\nGenerative Process for '{title_text}':")
    
    for step in range(num_generated):
        title_emb = encode_title([title_text])
        logits = model(sf, title_emb, m, training=False)[0]

        # -------------------------
        # Build the "seen" set
        # -------------------------
        if exclude_original:
            # Current behavior: block original + generated
            seen = ids[ids >= 0]
        else:
            # New behavior: allow original playlist songs,
            # only block songs we have generated in this loop.
            # First `orig_len` entries of `ids` are the original playlist;
            # entries after that are generated tokens.
            seen = ids[orig_len:]  # slice off the original songs

        # If there is nothing to exclude yet (first step and allow-original mode),
        # just use logits as-is.
        if tf.size(seen) > 0:
            exclude_mask = tf.scatter_nd(
                tf.expand_dims(seen, 1),
                tf.ones_like(seen, dtype=tf.float32),
                shape=(logits.shape[0],)
            )
            filtered_logits = logits - 1e9 * exclude_mask
        else:
            filtered_logits = logits

        # Top-k sampling
        _, top_k_indices = tf.math.top_k(filtered_logits, k=k)
        chosen_token = np.random.choice(top_k_indices.numpy())
        
        generated_ids.append(chosen_token)
        chosen_tid = int2track.get(chosen_token, "UNKNOWN")

        # Update tracking of ids (used for masking & features)
        ids = tf.concat([ids, [chosen_token]], axis=0)
        
        # Add new feature row for the chosen song
        if chosen_tid in track2feat:
            new_feat = tf.constant(track2feat[chosen_tid], dtype=tf.float32)
            new_feat = tf.reshape(new_feat, (1, 1, -1))
            sf = tf.concat([sf, new_feat], axis=1)
            m  = tf.concat([m, [[1]]], axis=1)
            
            track_name = track_meta.get(chosen_tid, "Unknown Title")
            print(f"  Step {step+1}: Added → {track_name}")
        else:
            print(f"  Step {step+1}: Added Token {chosen_token} (Feature lookup failed)")

    return generated_ids

# ====================================================
# 5. MANUAL "HUMAN TEST" PLAYLISTS
#     Each run with:
#       - ALLOW originals
#       - EXCLUDE originals
# ====================================================

tracks_df = playlists  # alias for readability

def pretty_print_playlist(playlist_song_ids, title_label):
    print("\n======================================")
    print("=== MANUAL HUMAN TEST PLAYLIST ===")
    print("Title:", title_label)

    original_token_ids = playlist_song_ids.numpy().tolist()
    original_track_ids = [int2track[int(tok)] for tok in original_token_ids]

    print("\nSongs already in playlist:")

    seen = set()  # optional: avoid printing duplicates if the seed list itself repeats
    for tid in original_track_ids:
        if tid in seen:
            continue
        seen.add(tid)

        if tid in meta_df.index:
            row = meta_df.loc[tid]
            print(f" • {row['track_name']} → {row['artist_name']}")
        else:
            print(f" • {tid} (metadata not found)")


def run_human_test(track_ids, title_text):
    song_feats, mask, playlist_song_ids = build_manual_batch(
        track_ids=track_ids,
        track2feat=track2feat,
        unk_id=UNK_ID,
    )

    if song_feats is None:
        print(f"Skipping playlist '{title_text}' — no valid tracks.")
        return

    pretty_print_playlist(playlist_song_ids, title_text)

    # ===== ORIGINAL TOKEN IDS (as integers) =====
    original_token_ids = set(playlist_song_ids.numpy().tolist())

    # ----------------------------------------------------
    # MODE A: ALLOW ORIGINALS
    # ----------------------------------------------------
    print("\n-- Initial Recommendations (ALLOW originals) --")
    rec_ids_allow = autoregressive_recommend(
        model=model,
        song_features=song_feats,
        title_text=title_text,
        mask=mask,
        playlist_song_ids=playlist_song_ids,
        k=20,
        num_generated=10,
        exclude_original=False,
    )

    for rid in rec_ids_allow:
        tid = int2track.get(int(rid), "UNKNOWN")
        print(" ➜", track_meta.get(tid, tid))

    # ----------------------------------------------------
    # FILTER OUT REPEATS
    # ----------------------------------------------------
    unique_recs = [rid for rid in rec_ids_allow if rid not in original_token_ids]
    num_unique = len(unique_recs)
    num_missing = 10 - num_unique

    if num_missing == 0:
        print("\n(No repeats — already 10 unique recommendations.)")
        final_recs = unique_recs
    else:
        print(f"\nFound {10 - num_unique} repeat(s). Generating {num_missing} replacement(s)...")

        # ----------------------------------------------------
        # MODE B: EXCLUDE ORIGINALS (generate only missing ones)
        # ----------------------------------------------------
        extra_recs = autoregressive_recommend(
            model=model,
            song_features=song_feats,
            title_text=title_text,
            mask=mask,
            playlist_song_ids=playlist_song_ids,
            k=20,
            num_generated=num_missing,
            exclude_original=True,
        )

        final_recs = unique_recs + extra_recs

    # Safety: ensure final list has exactly 10
    final_recs = final_recs[:10]

    print("\n====== FINAL 10 NON-REPEATED RECOMMENDATIONS ======")
    for rid in final_recs:
        tid = int2track.get(int(rid), "UNKNOWN")
        print(" ➜", track_meta.get(tid, tid))

# ====== Pop PLAYLIST ======
pop_ids = [
    "5Q0Nhxo0l2bP3pNjpGJwV1",
    "1Slwb6dOYkBlWal1PGtnNg",
    "32OlwWuMpZ6b0aN2RZOeMS",
    "3gbBpTdY8lnQwqxNCcf795",
    "7qiZfU4dY1lWllzX7mPBI3",
    "4wCmqSrbyCgxEXROQE6vtV",
    "1mXVgsBdtIVeCLJnSnmtdV",
    "1XGmzt0PVuFgQYYnV2It7A",
    "2vwlzO0Qp8kfEtzTsCXfyE",
    "2YlZnw2ikdb837oKMKjBkW",
    "4gbVRS8gloEluzf0GzDOFc",
    "7ySUcLPVX7KudhnmNcgY2D",
    "2kSb3wYSOV996xA2NSmpck",
    "67T6l4q3zVjC5nZZPXByU8",
    "4E5P1XyAFtrjpiIxkydly4",
    "1oHNvJVbFkexQc0BpQp7Y4",
    "3x2YIvkLcxvZQsA5x6xyIR",
    "4rHZZAmHpZrA3iH5zx8frV",
    "5g7sDjBhZ4I3gcFIpkrLuI",
    "4kbj5MwxO1bq9wjT5g9HaA",
    "6gj08XDlv9Duc2fPOxUmVD",
    "4356Typ82hUiFAynbLYbPn",
    "2ekn2ttSfGqwhhate0LSR0",
    "5bgwqaRSS3M8WHWruHgSL5",
    "0azC730Exh71aQlOt9Zj3y",
    "378iszndTZAR4dH8kwsLC6",
    "2oENJa1T33GJ0w8dC167G4",
    "4lLtanYk6tkMvooU0tWzG8",
    "0O45fw2L5vsWpdsOdXwNAR",
    "5i66xrvSh1MjjyDd6zcwgj",
    "1XGmzt0PVuFgQYYnV2It7A",
    "494OU6M7NOf4ICYb4zWCf5",
    "2dOTkLZFbpNXrhc24CnTFd",
    "37f4ITSlgPX81ad2EvmVQr",
    "6vSforRhuzsA0D0SO9fG1S",
    "62LHRv9uwSNlBmByQF5jdE",
    "6lV2MSQmRIkycDScNtrBXO",
    "1rKBOL9kJfX1Y4C3QaOvRH",
    "3oHNJECGN3bBoGXejlw2b1",
    "1mXuMM6zjPgjL4asbBsgnt",
    "69kOkLUCkxIZYexIgSG8rq",
    "6ECp64rv50XVz93WvxXMGF",
    "0KKkJNfGyhkQ5aFogxQAPU",
    "6b8Be6ljOzmkOmFslEb23P",
    "16pwlVsypm4aDKMhXdOuXg",
    "22mek4IiqubGD9ctzxc69s",
]

run_human_test(pop_ids, "Pop")

# ====== Rap PLAYLIST ======
rap_ids = [
    "3XVozq1aeqsJwpXrEZrDJ9",
    "3bidbhpOYeV4knp8AIu8Xn",
    "3ZLyt2ndLFBh148XRYjYYZ",
    "7KOlJ92bu51cltsD9KU5I7",
    "2bjwRfXMk4uRgOD9IBYl9h",
    "1f5cbQtDrykjarZVrShaDI",
    "6p8NuHm8uCGnn2Dtbtf7zE",
    "439TlnnznSiBbQbgXiBqAd",
    "2771LMNxwf62FTAdpJMQfM",
    "3muBQDekYAg7jm6hDu6R0Z",
    "6LxSe8YmdPxy095Ux6znaQ",
    "42zd6DYQ4o4SECmTITrM1U",
    "0eEXcw3JLVXcRxYrVYMy68",
    "7GX5flRQZVHRAGd6B4TmDO",
    "2KpCpk6HjXXLb7nnXoXA5O",
    "6HZILIRieu8S0iqY8kIKhj",
    "7KXjTSCq5nL1LoYtL7XAwS",
    "0VgkVdmE4gld66l8iyGjgx",
    "0z5ZPs57J2KERwM1tBM2GF",
    "1kMuU3TNQvHbqvXCWBodmP",
    "2gZUPNdnz5Y45eiGxpHGSc",
    "23SZWX2IaDnxmhFsSLvkG2",
    "7yNK27ZTpHew0c55VvIJgm",
    "4uhvMW7ly7tJil31YYscAN",
    "22L7bfCiAkJo5xGSQgmiIO",
    "1XRgIKC5TPwo7nWGyKqgG0",
    "4jTiyLlOJVJj3mCr7yfPQD",
    "4Km5HrUvYTaSUfiSGPJeQR",
    "0SGkqnVQo9KPytSri1H6cF",
    "4Kz4RdRCceaA9VgTqBhBfa",
    "3a1lNhkSLSkpJE4MSHpDu9",
    "3yfqSUWxFvZELEM4PmlwIR",
    "7KwZNVEaqikRSBSpyhXK2j",
    "1JClFT74TYSXlzpagbmj0S",
    "3IrkbGQCoEPAkzJ0Tkv8nm",
    "2kQuhkFX7uSVepCD3h29g5",
    "355RUwXtdecClzxv1zU98n",
    "0Puj4YlTm6xNzDDADXHMI9",
    "0N3W5peJUQtI4eyR6GJT5O",
    "6SwRhMLwNqEi6alNPVG00n",
    "6MreoQf8oixWI2xRcw3Fv1",
    "20oenBXlmwIfK0F3fQIjhM",
    "62vpWI1CHwFy7tMIcSStl8",
    "6GnhWMhgJb7uyiiPEiEkDA",
    "0v9Wz8o0BT8DU38R4ddjeH",
    "19a3JfW8BQwqHWUMbcqSx8",
    "5qxChyzKLEyoPJ5qGrdurN",
    "5uZm7EFtP5aoTJvx5gv9Xf",
    "5uQOauh47VFt3B2kV9kRXw",
]

run_human_test(rap_ids, "Rap")



# ====== Country PLAYLIST ======
country_ids = [
    "4BP3uh0hFLFRb5cjsgLqDh",
    "3fqwjXwUGN6vbzIwvyFMhx",
    "0ZUo4YjG4saFnEJhdWp9Bt",
    "0jHSdjxn9LfzNL0WkDu93W",
    "2SpEHTbUuebeLkgs9QB7Ue",
    "5KqldkCunQ2rWxruMEtGh0",
    "54eZmuggBFJbV7k248bTTt",
    "11EX5yhxr9Ihl3IN1asrfK",
    "2LawezPeJhN4AWuSB0GtAU",
    "12TE7Vt592RcM1G3EaaZ0f",
    "1mvl4McrgpDUwusRRrn7UU",
    "5N3S6ppQkYlWLd0WuOJUYz",
    "3yrmIOL4ImKaRxOXiVNGEv",
    "1PoGWZbJPGmViVi7CYbDUK",
    "3b7CDTKB0SRTmQ6ytYi5vZ",
    "5vCgOg9VqRaAUbnflCO6P3",
    "3MFV4DgrAOXz6KURPQxRj9",
    "5HGibWoxnkYSkl6mHmAlOE",
    "0dbzWSYpMcRtwjI1S7Pkql",
    "6ZOPiKQeibCn7fP8dncucL",
    "5VJRSXqHca3yAsCEISymlc",
    "61hmCqoIlTJjcVMMLhcH5n",
    "5Oxgt1m5SMpwM17zByC11n",
    "1M9qgq0SaZ5OuAeU0GKXif",
    "3esPcn43N0CytAtcY9V30C",
    "5ytc87gcx6dZ4nWRNykinV",
    "1KmquVzXEoIHUZuZqKGRvs",
    "12utEDJIkWakIdxCqAedaf",
    "3XmGUn730lnJnYMUAMhlHk",
    "0H04yVa3DJxoXbLBpAb7iV",
    "2j79NtwxoWDmkiH4MGwdLq",
    "5n2yia2AwxIAe9qMtlvYQt",
    "28Sc21lrWcZD9Ov4rRkPJq",
    "08zgiAP2zezRPHdrWaRDbO",
    "0c4ICGb0jvszKj3KPR59JU",
    "1eFiehZXXGbxfkZx9mraUj",
    "5yIiXdLRE85OBiQmCaUenq",
    "1B5FYFidz18CrtMttAVMgw",
    "15NGPktDl1wh3kY4rMS9TH",
    "2qYsSHsYkihWx043HVJQRV",
    "3gVRILe7XCyNakb6sy5umX",
    "7mu21H4g5sGUX7n3tGsGig",
    "4cDWuDOQPJsnS88XKrkAFy",
    "539dFCMA8lfEElDPgGpcIB",
    "49bZZaVzzyhX0Sd4aWuonJ",
    "2SpEHTbUuebeLkgs9QB7Ue",
]

run_human_test(country_ids, "American country yeehaw")

# ====== Jazz PLAYLIST ======
jazz_ids = [
    "0aWMVrwxPNYkKmFthzmpRi",
    "1kKLWkqyZEnrOd5tBYYCUn",
    "1ko2lVN0vKGUl9zrU0qSlT",
    "4l9hml2UCnxoNI3yCdL1BW",
    "01hJnhpAmjzg85Etnz2ECH",
    "3ts3nBCGF7K9JvSjWVQfX8",
    "1muGsbcN9ykkhvYiiasp71",
    "6pY1AWqj42B5pngwtgyTJ4",
    "0F845nujLVqCb0XMZCh5Pc",
    "2Cakl6tRkLxccrnqJdvGpu",
    "6RjvMuSa4ZCqPueIIO4IBs",
    "1YQWosTIljIvxAgHWTp7KP",
    "4IbOPxstIn2KbdlWf5xRZ0",
]

run_human_test(jazz_ids, "Jazz")


# ====== Workout PLAYLIST ======

workout_ids = [
    "0WqIKmW4BTrj3eJFmnCKMv",
    "3qK8x4GZcIkzTz9JEqvIF5",
    "22L7bfCiAkJo5xGSQgmiIO",
    "5A6OHHy73AR5tLxgTc98zz",
    "2PpruBYCo4H7WOBJ7Q2EwM",
    "0k2GOhqsrxDTAbFFSdNJjT",
    "1Ser4X0TKttOvo8bgdytTP",
    "57bgtoPSgt236HzfBOd8kj",
    "7LRMbd3LEoV5wZJvXT1Lwb",
    "3yfqSUWxFvZELEM4PmlwIR",
    "2kQuhkFX7uSVepCD3h29g5",
    "2MvIMgtWyK88OiPi0J8Dg3",
    "2KpCpk6HjXXLb7nnXoXA5O",
    "7KXjTSCq5nL1LoYtL7XAwS",
    "0VgkVdmE4gld66l8iyGjgx",
    "4DLH2fr8pWX1iksMrk47Kw",
    "39sUeHQoIGY6BEIcgTMRXW",
    "65eohvrL4ttjA7EfFkQOhX",
    "6taWR0qCkEAZZNjoW8KccZ",
    "1dFnMC9NlHVUE4rlT5vr83",
    "5VYbLzAplnqPaGNpcf8qhZ",
    "3rpnfXSECgapxeGeRgUYqy",
    "7GVqIMb6OdMY5mG7fUrtOq",
    "1FPSkRkDlthbAceSIIWc4C",
    "5zinAGcI5aZ63vmqIrUXkj",
    "6I9VzXrHxO9rA9A5euc8Ak",
    "4fixebDZAVToLbUCuEloa2",
    "3MjUtNVVq3C8Fn0MP3zhXa",
    "6naxalmIoLFWR0siv8dnQQ",
    "2d8JP84HNLKhmd6IYOoupQ",
    "62LHRv9uwSNlBmByQF5jdE",
    "7AQjiRtIpr33P8UT98iveh",
    "5W3cjX2J3tjhG8zb6u0qHn",
    "0dhxE0JTayqMycMva3D3rM",
    "0oZiK5HCMPn241PbUCslkF",
    "6M3lqzrkzSKrBehkeAxTXc",
    "5vRE2lyh7sE5Mn1dMF2Mps",
    "4Km5HrUvYTaSUfiSGPJeQR",
    "2D1hlMwWWXpkc3CZJ5U351",
    "7iL6o9tox1zgHpKUfh9vuC",
    "3LHWJ7C6P8fGzpdoiuoI3A",
    "0Puj4YlTm6xNzDDADXHMI9",
    "5NLuC70kZQv8q34QyQa1DP",
    "5qxChyzKLEyoPJ5qGrdurN",
    "3IOQZRcEkplCXg6LofKqE9",
    "6gBFPUFcJLzWGx4lenP6h2",
    "0c7wqpBLOTFr1yb70LHGFM",
    "5B37ocpk2zxeZL1lq5F6ui",
]

run_human_test(workout_ids, "Workout")


# ====== Chill PLAYLIST ======
chill_ids = [
    "0q6LuUqGLUiCPP1cbdwFs3",
    "3eze1OsZ1rqeXkKStNfTmi",
    "1G391cbiT3v3Cywg8T7DM1",
    "2ZWlPOoWh0626oTaHrnl2a",
    "4uhvMW7ly7tJil31YYscAN",
    "2eAZfqOm4EnOF9VvN50Tyc",
    "0BgbobvykXxEvxo2HhCuvM",
    "7DfFc7a6Rwfi3YQMRbDMau",
    "3Ti0GdlrotgwsAVBBugv0I",
    "3sNVsP50132BTNlImLx70i",
    "1H4idkmruFoJBg1DvUv2tY",
    "6yLIqXX9edg1x0HZS7cZEv",
    "5Z3GHaZ6ec9bsiI5BenrbY",
    "1TPLsNVlofwX1txcE9gZZF",
    "6PGoSes0D9eUDeeAafB2As",
    "152lZdxL1OR0ZMW6KquMif",
    "3h5AZDf5z7D18plaLtHTfi",
    "0725YWm6Z0TpZ6wrNk64Eb",
    "3O4s2m47MFhnGqmpkjoKYk",
    "5WOLghNew2YKZO96gYoM7p",
    "4U45aEWtQhrm8A5mxPaFZ7",
    "12jjuxN1gxlm29cqL5M6MW",
    "3Try68sPCVOlmX0B5o57ZT",
    "5fCxl609046LL8q6Ib5xaY",
    "6wntNrWu9bXBNPmujdwqhL",
    "6eT7xZZlB2mwyzJ2sUKG6w",
    "71GvlH0VdeClloLIkHrAVu",
    "2Yf0HjCklkx54C4wQzhvzH",
    "12q3V8ShACq2PSWINMc2rC",
    "3Umg8CDhO8dOSj7yBTInYb",
    "1ITQbrueGLl581a25XXm9c",
    "65vdMBskhx3akkG9vQlSH1",
    "5SUlhldQJtOhUr2GzH5RI7",
    "5XmetMMUFNXClbiYnGdVmP",
    "4L7jMAP8UcIe309yQmkdcO",
    "35xSkNIXi504fcEwz9USRB",
    "2VQc9orzwE6a5qFfy54P6e",
    "3xKsf9qdS1CyvXSMEid6g8",
    "4bEcoz1OcfMgUbp2ft8ieQ",
    "4xR3MAscflQ262kMeiKshQ",
    "0djZ2ndRfAL69WYNra5jRC",
    "1eN42Q7IWRzRBq8eW2Y2TE",
    "0b1NUCAYfEOuPx9nELBBfX",
    "4Acofe9hICRvyBTP5hFNk0",
    "7fOwfp4u4UD9REy0SOXIGO",
    "4OBZT9EnhYIV17t4pGw7ig",
]

run_human_test(chill_ids, "Chill slow vibes")


# ====== 70's Hard Rock PLAYLIST ======

hard_rock_ids = [
    "4f3RDq9nYPBeR1yMSgnmBm",
    "6N0AnkdDFZUetw8KAGHV7e",
    "0L7zm6afBEtrNKo6C6Gj08",
    "2RaA6kIcvomt77qlIgGhCT",
    "0IXpUl1fn2QZcBavfuq0H4",
    "3w9zHVKI28aUPZTxq8oDC6",
    "5axOkQnmQmwtjr4bv1Xt7i",
    "36WNQJFVjYhDRZ6vJhYXEh",
    "7w6PJe5KBPyvuRYxFkPssC",
    "3VPcxdwwo9TcjAOYZa5ZrP",
    "2E7W1X4maFFcjHrVrFA7Vs",
    "0i1RTnH2Lj5gTDRU5wtyT2",
    "1anIy4FRJbu7N8ASed1bcE",
    "0gXMYs6RDg5muWWoi2q3mb",
    "594ONMYoHlJnqHPEx98t08",
    "2zYzyRzz6pRmhPzyfMEC8s",
    "08mG3Y1vljYA6bvDt4Wqkj",
    "2SiXAy7TuUkycRVbbWDEpo",
    "57bgtoPSgt236HzfBOd8kj",
    "7LRMbd3LEoV5wZJvXT1Lwb",
    "0C80GCp0mMuBzLf3EAXqxv",
    "69QHm3pustz01CJRwdo20z",
    "2d4e45fmUnguxh6yqC7gNT",
    "2j0zExWFB0PowLOeoZosjK",
    "6Wn3pdFtAcnYJyJVITwt7N",
    "7zscdQe9CjzXnqT3P1Ey7K",
    "753KutoAy00apPsplMRetG",
    "3KhF2YiNpJvGpfiCW45R6D",
    "05RgAMGypEvqhNs5hPCbMS",
    "6QDbGdbJ57Mtkflsg42WV5",
    "6fybp4N6eW3bsFAvARxyVe",
    "4RS9PmtHQe7I0o5fEeweOY",
    "0upLyFR8Rr52ZpMp5esQoq",
    "2tAeN2TKlQLOoSPXtARzBV",
    "26fZwf1ImE4aUJ4XaqOkUg",
    "0YpmF3aZXOIuyi8itZbpkp",
    "7o2CTH4ctstm8TNelqjb51",
    "3YBZIN3rekqsKxbJc9FZko",
    "0bVtevEgtDIeRjCJbK3Lmv",
    "2vNw57KPaYDzkyPxXYUORX",
    "2dyfo7lqKI7NtSAhUZwnoJ",
    "6v5VsfCYKdrkQBTMhAlkDr",
    "78lgmZwycJ3nzsdgmPPGNx",
    "3MODES4TNtygekLl146Dxd",
    "3qT4bUD1MaWpGrTwcvguhb",
    "6zGDIDjfDkPyNxrEERO3XG",
    "1UBQ5GK8JaQjm5VbkBZY66",
    "0OBwxFLu6Yj61s2OagYbgY",
    "4VfWY8hydsKIXtKQ2AT9oO",
    "3YdhuJjqZaJKC4xWKUUyYl",
    "5SZ6zX4rOrEQferfFC2MfP",
    "6gQUbFwwdYXlKdmqRoWKJe",
    "2KCJYwlBWxdlwyIYckIf6V",
    "28clONjZmul6FjfO6tZQDE",
    "2rd9ETlulTbz6BYZcdvIE1",
    "1Y373MqadDRtclJNdnUXVc",
    "3IOQZRcEkplCXg6LofKqE9",
    "6EPRKhUOdiFSQwGBRBbvsZ",
    "3QZ7uX97s82HFYSmQUAN1D",
    "0K6yUnIKNsFtfIpTgGtcHm",
]

run_human_test(hard_rock_ids, "70s Hard Rock")

# ====== Latin PLAYLIST ======

latin_ids = [
    "3FdHgoJbH3DXNtGLh56pFu",
    "2a1o6ZejUi8U3wzzOtCOYw",
    "0Ph6L4l8dYUuXFmb71Ajnd",
    "1A5yplwEk6cJbAL63L6bkp",
    "7JIjUx3GsL0upxmNJacmtz",
    "23TUteFlo4GRVKy2zRvtMC",
    "0o9Vr0BGIQPh32staLHcuU",
    "3ZFTkvIE7kyPt6Nu3PEa7V",
    "2Cd9iWfcOpGDHLz6tVA3G4",
    "3QHMxEOAGD51PDlbFPHLyJ",
    "2HbmLkHkkI15eES8kpWRuI",
    "3fwqOw3lzCle2zxamHMcQ2",
    "07U5izMEcWaETGct9nhhAg",
]

run_human_test(latin_ids, "Latin")


# ====== Christmas PLAYLIST ======

christmas_ids = [
    "4so0Wek9Ig1p6CRCHuINwW",
    "2FRnf9qhLbvw8fu4IBXx78",
    "2FPfeYlrbSBR8PwCU0zaqq",
    "5xlS0QkVrSH7ssEbBgBzbM",
    "6Lh85Pe18pqvNdDoWgTVUC",
    "0lLdorYw7lVrJydTINhWdI",
    "4z8sz6E4YyFuEkv5o7IJni",
    "2U9kDk5mlHYunC7PvbZ8KX",
    "09OojFvtrM9YRzRjnXqJjA",
    "1foCxQtxBweJtZmdxhEHVO",
    "0bYg9bo50gSsH3LtXe2SQn",
    "6a1hzBiTkgqTsDA0Xgfh9z",
    "7taXf5odg9xCAZERYfyOkS",
    "4ricyQVd20UQde1jpXCSuJ",
]

run_human_test(christmas_ids, "Christmas")

# ====== Taylor Swift PLAYLIST ======

taylor_swift_ids = [
    "5yEPktRqvIhko5QFF3aBhQ",
    "1NmVZsG18CzCAtw7rnV3yA",
    "4XMP3zVxrnr58T0tjIHvpR",
    "4NAmRvqSITAAzKWnC8yRq3",
    "10eBRyImhfqVvkiVEGf0N0",
    "3fVnlF4pGqWI9flVENcT28",
    "72jCZdH0Lhg93z6Z4hBjgj",
    "5gRYrtvyVyaCRvLt56OfuV",
    "0cITLOYn1Sv4q27zZPqlNK",
    "1vrd6UOGamcKNGnSHJQlSt",
    "3GCL1PydwsLodcpv0Ll1ch",
    "28M2gifMU282QBM3fKajIS",
    "6d9IiDcFxtFVIvt9pCqyGH",
    "2ULNeSomDxVNmdDy8VxEBU",
]
run_human_test(taylor_swift_ids, "Taylor Swift")
