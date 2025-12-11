"""
Data Loading Utilities for Playlist Recommendation

This module provides functions to load and prepare TFRecord datasets for training,
including dataset creation, parsing, batching, and augmentation with masking.
"""
import os
import time
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


from .preprocessing import (
    serialize_playlist,
    parse_playlist,
    prepare_batched_dataset,
    create_input_target_with_dropout
)

def load_and_prepare_data(csv_path):
    """
    Complete data loading and preparation pipeline.
    
    Orchestrates the entire data loading workflow including CSV reading, track
    vocabulary creation, train/val/test splitting, and TFRecord serialization.
    
    Args:
        csv_path (str): Path to CSV file containing playlist data
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, vocab_info) where vocab_info
               contains VOCAB_SIZE and mapping dictionaries
    """
    print("reading csv...")
    start = time.perf_counter()
    playlists = pd.read_csv(csv_path, nrows=10000)
    print(f'read_csv took {start - time.perf_counter()}')

    print("\nChecking playlist_name column for issues...")

    # Count NaNs
    num_nans = playlists["playlist_name"].isna().sum()
    print(f"Number of NaN playlist names: {num_nans}")

    # Count non-string types (e.g., floats)
    non_string_mask = ~playlists["playlist_name"].apply(lambda x: isinstance(x, str))
    num_non_strings = non_string_mask.sum()
    print(f"Number of NON-string playlist names: {num_non_strings}")

    # Fix missing / non-string playlist names
    playlists["playlist_name"] = playlists["playlist_name"].fillna("")
    playlists["playlist_name"] = playlists["playlist_name"].astype(str)

    track_columns = [
        'track_id','year_filled','popularity_filled', 'explicit_filled', 'year_is_missing','popularity_is_missing', 'explicit_is_missing',
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'duration_ms', 'time_signature'
    ]


    # Add all the genre columns, which start from 'genre_acoustic'
    genre_columns = [col for col in playlists.columns if col.startswith('genre_')]
    all_track_features = track_columns + genre_columns

    # Create a new DataFrame containing only track information
    tracks_df = playlists[all_track_features].copy()

    # Drop duplicates to ensure each track ID appears only once
    tracks_df.drop_duplicates(subset=['track_id'], inplace=True)

    print(f"\nTotal unique tracks extracted: {len(tracks_df)}")
    print(tracks_df.head())

    """The code block below is to pull the information from the dataset"""

    # Create a sorted list of all unique track IDs present in the dataset
    unique_ids = sorted(playlists["track_id"].unique().tolist())

    # Most frequent songs to keep as part of vocab
    K = 30000 

    # Special token for unknown tracks
    UNK_TOKEN = '<UNK>' 

    # Reserve the 0 index for the UNK token
    UNK_ID = 0          

    # Calculate the frequency of each track ID
    track_counts = playlists["track_id"].value_counts()
    print(f"Total unique tracks found: {len(track_counts)}")

    # Select the Top-K most frequent track IDs
    top_k_track_ids = track_counts.head(K).index.tolist()
    print(f"Top {K} tracks selected for vocabulary.")

    # Create vocabulary mappings
    vocab_list = [UNK_TOKEN] + sorted(top_k_track_ids)
    track2int = {tid: i for i, tid in enumerate(vocab_list)}
    # Map each integer index back to the track ID
    int2track = {i: tid for tid, i in track2int.items()}

    # Define vocab size 
    VOCAB_SIZE = K
    print(f"Total Unique Tracks (VOCAB_SIZE): {VOCAB_SIZE}")

    # Map track IDs to integer tokens
    playlists["track_token"] = playlists["track_id"].apply(
        lambda tid: track2int.get(tid, UNK_ID)
    )

    # Filter out playlists with too few valid songs (< 5 songs)
    MIN_NON_UNK_SONGS = 5
    is_not_unk = playlists["track_token"] != UNK_ID
    non_unk_counts = is_not_unk.groupby(playlists["playlist_id"]).sum()
    valid_playlist_ids = non_unk_counts[non_unk_counts >= MIN_NON_UNK_SONGS].index

    # Filter playlists
    initial_track_count = len(playlists)
    initial_playlist_count = playlists["playlist_id"].nunique()
    playlists = playlists[playlists["playlist_id"].isin(valid_playlist_ids)].copy()
    final_track_count = len(playlists)
    final_playlist_count = playlists["playlist_id"].nunique()

    print("\n--- Playlist Filtering Summary ---")
    print(f"Playlists before filtering: {initial_playlist_count} (Total Tracks: {initial_track_count})")
    print(f"Playlists retained (with >= {MIN_NON_UNK_SONGS} non-UNK songs): {final_playlist_count} (Total Tracks: {final_track_count})")

    # List of the standard audio feature columns
    feature_cols = [
        "danceability","energy","key","loudness","mode", "year_filled","popularity_filled","explicit_filled", "year_is_missing","popularity_is_missing", "explicit_is_missing", # ------------- removed popularity -------------------
        "speechiness","acousticness","instrumentalness","liveness",
        "valence","tempo","duration_ms","time_signature", 
    ] + [c for c in playlists.columns if c.startswith("genre_")] # Add all genre columns

    # Extract the feature values as a NumPy array (used for model input)
    merged_features = playlists[feature_cols].values.astype("float32")

    FEATURE_DIM = merged_features.shape[1]
    print(f"Total Features (FEATURE_DIM): {FEATURE_DIM}")

    # Build dictionary: raw track_id → feature vector
    track2feat = {
        tid: feat
        for tid, feat in zip(
            playlists.groupby("track_id")[feature_cols].first().index,
            playlists.groupby("track_id")[feature_cols].first().values.astype("float32")
        )
    }

    """This next block is to group the playlists by id"""

    # Group 1: Features
    grouped_features = (
        playlists.groupby("playlist_id")
        .apply(lambda df: df[feature_cols].values, include_groups=False) 
    )

    # Group 2: Track Tokens
    grouped_track_tokens = (
        playlists.groupby("playlist_id")
        .apply(lambda df: df["track_token"].values, include_groups=False)
    )

    # Group 3: Playlist Titles
    grouped_titles = (
        playlists.groupby("playlist_id")
        .apply(lambda df: df["playlist_name"].iloc[0], include_groups=False) 
    )

    """This next block changes the data into smaller segments"""

    
    tfrecord_path = "playlists.tfrecord"

    all_playlist_ids = grouped_features.index.tolist()

    # Split: 80% Train, 20% Temp (Test + Val)
    train_ids, temp_ids = train_test_split(all_playlist_ids, test_size=0.2, random_state=42)

    # Second split: Split the 20% Temp into 50/50 Validation and Test (10% total each)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    def write_tfrecord(filename, playlist_ids):
        with tf.io.TFRecordWriter(filename) as writer:
            for pid in playlist_ids:
                feats = grouped_features[pid]
                ids = grouped_track_tokens[pid]
                title = grouped_titles[pid]

                serialized = serialize_playlist(feats, ids, title)
                writer.write(serialized)
            print(f"Finished writing {filename}")

    write_tfrecord("train.tfrecord", train_ids)
    write_tfrecord("val.tfrecord", val_ids)
    write_tfrecord("test.tfrecord", test_ids)

    """The next block"""

    train_dataset = tf.data.TFRecordDataset("train.tfrecord")
    val_dataset = tf.data.TFRecordDataset("val.tfrecord")
    test_dataset = tf.data.TFRecordDataset("test.tfrecord")
    train_dataset = train_dataset.map(parse_playlist, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(parse_playlist, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.map(parse_playlist, num_parallel_calls=tf.data.AUTOTUNE)

    # Usage Example:
    train_dataset = prepare_batched_dataset(train_dataset, FEATURE_DIM)
    val_dataset   = prepare_batched_dataset(val_dataset, FEATURE_DIM)
    test_dataset  = prepare_batched_dataset(test_dataset, FEATURE_DIM)
    print("Datasets prepared successfully (Order Preserved)!")

    # Apply the denoising map function to measure reconstruction capability
    train_dataset_final = train_dataset.map(create_input_target_with_dropout, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = train_dataset_final.prefetch(tf.data.AUTOTUNE)

    val_dataset_final = val_dataset.map(create_input_target_with_dropout, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset_final.prefetch(tf.data.AUTOTUNE)

    test_dataset_final = test_dataset.map(create_input_target_with_dropout, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset_final.prefetch(tf.data.AUTOTUNE)

    print("✅ Pipeline updated with Denoising Autoencoder strategy.")

    vocab_info = {
        'VOCAB_SIZE': VOCAB_SIZE,
        'FEATURE_DIM': FEATURE_DIM,
        'track2int': track2int,
        'int2track': int2track
    }

    for inputs, labels in dataset.take(1):
        batch_song_feats = inputs["features"]
        batch_title_texts = inputs["title"]
        batch_mask = inputs["mask"]

        print("--- Checking Single Batch ---")
        print(f"Song Features Shape: {batch_song_feats.shape}")
        print(f"Title Texts Shape:   {batch_title_texts.shape}")
        print(f"Mask Shape:          {batch_mask.shape}")
        print(f"Labels Shape:        {labels.shape}")
        print(f"Titles: {batch_title_texts}")
        print(f"Playlist: {labels[0]}")
        print(f"Mask: {batch_mask[0]}")
        # Check for NaNs/Infs in the input data
        # NaNs/Infs can cause loss to become None or NaN downstream
        if tf.reduce_any(tf.math.is_nan(batch_song_feats)):
            print("🚨 **WARNING: NaN found in Song Features!**")
        break # Only check the first batch

    with open("../datafeature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    return dataset, val_dataset, test_dataset, vocab_info

