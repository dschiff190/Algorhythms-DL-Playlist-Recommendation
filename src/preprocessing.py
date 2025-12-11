# -*- coding: utf-8 -*-
"""MYModel.ipynb
# Preparing Batches with Data

Tokenize first
"""
import tensorflow as tf
import pickle as pickle


"""This next block changes the data into smaller segments"""

def serialize_playlist(features_array, track_id_array, title):
    playlist_length, feature_dim = features_array.shape

    example = tf.train.Example(features=tf.train.Features(feature={
        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[playlist_length])),
        "feature_dim": tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_dim])),

        # store flattened features
        "sequence": tf.train.Feature(
            float_list=tf.train.FloatList(value=features_array.flatten().tolist())
        ), # <--- MISSING COMMA ADDED HERE

        # store track ids
        "track_ids": tf.train.Feature(
            int64_list=tf.train.Int64List(value=track_id_array.tolist())
        ), # <--- COMMA ADDED HERE (though optional, it's good practice)

        # store title
        "title": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[title.encode("utf-8")])
        ),
    }))
    return example.SerializeToString()


"""The next block"""

def parse_playlist(example_proto):
    # 1. Define Features
    features = {
        "length": tf.io.FixedLenFeature([], tf.int64),
        "feature_dim": tf.io.FixedLenFeature([], tf.int64),
        "title": tf.io.FixedLenFeature([], tf.string),

        # NOTE: When writing, you flattened the array into a single list,
        # so you must use FixedLenFeature here with a default value of []
        # as the shape is not fixed across examples.
        # Since the TFRecord doesn't store the shape, you must use VarLen for the value list.
        # Correction: Use VarLenFeature as the number of elements varies per playlist.
        "sequence": tf.io.VarLenFeature(tf.float32),
        "track_ids": tf.io.VarLenFeature(tf.int64),
    }

    # 2. Parse the Example
    parsed = tf.io.parse_single_example(example_proto, features)

    # 3. Extract Dimensions
    length = parsed["length"]
    feature_dim = parsed["feature_dim"]

    # 4. Convert Sparse Tensors to Dense and Reshape

    # Dense float sequence (features)
    seq = tf.sparse.to_dense(parsed["sequence"])
    # Reshape features back to (length, feature_dim)
    features_seq = tf.reshape(seq, (length, feature_dim))

    # Dense track IDs (tokens)
    track_tokens = tf.sparse.to_dense(parsed["track_ids"])

    # 5. Define Input and Labels (Target) for Sequence Modeling
    # Typically, for sequence prediction:
    # INPUT is the sequence up to the second-to-last item.
    # LABELS are the sequence from the second item to the end (shifted by one).
    input_tokens = track_tokens

    labels = track_tokens # This is the token we want the model to predict next

    # We also need to split the features sequence in the same way if used as input
    input_features = features_seq

    # The length is now reduced by one since we shifted the sequence.
    new_length = tf.shape(input_tokens)[0]

    # 6. Create Masks (Needed for padding in subsequent steps like .padded_batch)
    # Masking is simpler here, just defining a full-length mask for the new sequence.
    input_mask = tf.ones((new_length,), dtype=tf.int32)

    # 7. Return Tuple
    # Output structure: (Input Dictionary, Label Tensor)
    # The input dictionary should contain all the data your model needs.
    return {
        "track_tokens": input_tokens,
        "features": input_features,
        "title": parsed["title"],

        "mask": input_mask,
    }, labels



def prepare_batched_dataset(dataset, feature_dim):
    """
    Takes a TFRecord dataset, shuffles the order of playlists,
    and batches by sequence length.

    NOTE: This version preserves the original song order within each playlist.
    """

    # ====================================================
    # 0. SAFETY CHECK: Handle Re-Runs
    # ====================================================
    spec = dataset.element_spec

    # Check if spec is a tuple (inputs, labels) and inputs is a dict
    if isinstance(spec, tuple) and isinstance(spec[0], dict):
        features_shape = spec[0]["features"].shape
        # If rank is 3 (Batch, Sequence, Features), it is already batched.
        if len(features_shape) == 3:
            print(">> Detected already batched dataset (Rank 3). Unbatching...")
            dataset = dataset.unbatch()

    # Fallback check
    spec = dataset.element_spec
    if isinstance(spec, tuple) and isinstance(spec[0], dict):
         if len(spec[0]["features"].shape) == 3:
             print(">> Dataset is still batched! Unbatching again...")
             dataset = dataset.unbatch()

    # ====================================================
    # Helper Functions
    # ====================================================

    # Helper: Calculate Sequence Length
    def _element_length_func(input_dict, labels):
        return tf.shape(input_dict["track_tokens"])[0]

    # ====================================================
    # Main Pipeline
    # ====================================================

    # 1. Shuffle the Playlists (Dataset level shuffle)
    #    We still shuffle which playlist comes first/second, but NOT the content inside.
    # (REMOVED: The step that shuffled items within the playlist is gone)

    # 2. Define Bucket Configs
    bucket_boundaries = [20, 40, 60, 80, 120, 200]
    bucket_batch_sizes = [32, 32, 32, 32, 16, 8, 4]

    # 3. Batch using bucket_by_sequence_length
    dataset = dataset.bucket_by_sequence_length(
        element_length_func=_element_length_func,
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,

        padded_shapes=(
            # Input Dictionary Shapes
            {
                "track_tokens": [None],              # Sequence (will be padded)
                "features":     [None, feature_dim], # Sequence (will be padded)
                "title":        [],                  # Scalar (no padding needed)
                "mask":         [None],              # Sequence (will be padded)
            },
            # Labels Tensor Shape
            [None]                                   # Sequence (will be padded)
        ),

        padding_values=(
            # Input Dictionary Padding Values
            {
                "track_tokens": tf.constant(-1, dtype=tf.int64),
                "features":     tf.constant(0.0, dtype=tf.float32),
                "title":        tf.constant("", dtype=tf.string),
                "mask":         tf.constant(0, dtype=tf.int32),
            },
            # Labels Tensor Padding Value
            tf.constant(-1, dtype=tf.int64)
        )
    )

    return dataset


def create_input_target_with_dropout(input_dict, labels):
    """
    Prepares inputs and targets for a Denoising Autoencoder / Recommender task.

    1. INPUTS: Randomly masks 10-50% of the songs in the playlist.
       - Masked tokens become 0 (UNK_ID).
       - Masked features become 0.0.
       - The 'mask' tensor is updated so the Transformer ignores these slots in attention.

    2. LABELS: Remains the ORIGINAL, full playlist.
       - This forces the model to predict the songs we hid.
    """

    # 1. Unpack existing tensors
    # input_dict["features"] shape: (Batch, Seq_Len, Feature_Dim)
    # labels shape: (Batch, Seq_Len)
    features = input_dict["features"]
    tokens = input_dict["track_tokens"]
    title = input_dict["title"]
    original_mask = input_dict["mask"] # 1 for real song, 0 for padding

    # 2. Generate Dropout Mask
    # We want to keep 50% to 90% of songs visible (drop 10% to 50%)
    keep_prob = tf.random.uniform([], minval=0.5, maxval=0.9)

    # Generate random probabilities matching the token shape
    random_probs = tf.random.uniform(tf.shape(tokens))

    # Create boolean mask: True = Keep, False = Drop
    # We also ensure we NEVER drop padding (keep where original_mask is 0)
    # essentially: keep if (prob < keep_prob) OR (it is padding)
    # But simpler: calculate drop mask only on valid items
    is_kept = random_probs < keep_prob

    # 3. Apply Masking to Features
    # Expand dims for broadcasting: (Batch, Seq, 1) to match (Batch, Seq, Feat)
    feature_mask = tf.cast(is_kept, tf.float32)[:, :, tf.newaxis]

    # Zero out features where is_kept is False
    masked_features = features * feature_mask

    # 4. Apply Masking to Tokens (Input IDs)
    # UNK_ID is 0. We replace dropped tokens with 0.
    # Note: If padding is also 0 or -1, ensure we don't confuse model.
    # Here we assume dropped song = UNK (0).
    masked_tokens = tf.where(is_kept, tokens, tf.zeros_like(tokens))

    # 5. Update the Attention Mask
    # The Set Transformer should strictly attend to VISIBLE songs.
    # So we multiply the original mask (padding mask) by our keep mask.
    new_mask = tf.cast(original_mask, tf.int32) * tf.cast(is_kept, tf.int32)

    # 6. Pack inputs
    model_inputs = {
        "features": masked_features,  # Corrupted input
        "title": title,
        "mask": new_mask,             # Attention only on kept songs
        "track_tokens": masked_tokens
    }

    # 7. Targets stay pure!
    # We want to predict the original tokens (labels) given the corrupted input.
    return model_inputs, labels

