"""
Data Preprocessing and Batch Preparation for Playlist Recommendation

This module handles serialization/deserialization of playlist data to TFRecords,
batching with sequence length bucketing, and application of masking for the
denoising autoencoder training objective.
"""
import tensorflow as tf
import pickle as pickle

def serialize_playlist(features_array, track_id_array, title):
    """
    Serialize a single playlist to a TFRecord Example format.
    
    Converts playlist features, track IDs, and title into a serialized format
    suitable for efficient storage and loading from TFRecords.
    
    Args:
        features_array (np.ndarray): Song audio features of shape (seq_len, feature_dim)
        track_id_array (np.ndarray): Song IDs for this playlist of shape (seq_len,)
        title (str): Playlist title/name
        
    Returns:
        bytes: Serialized TFRecord Example
    """
    playlist_length, feature_dim = features_array.shape

    example = tf.train.Example(features=tf.train.Features(feature={
        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[playlist_length])),
        "feature_dim": tf.train.Feature(int64_list=tf.train.Int64List(value=[feature_dim])),

        # store flattened features
        "sequence": tf.train.Feature(
            float_list=tf.train.FloatList(value=features_array.flatten().tolist())
        ), 

        # store track ids
        "track_ids": tf.train.Feature(
            int64_list=tf.train.Int64List(value=track_id_array.tolist())
        ), 

        # store title
        "title": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[title.encode("utf-8")])
        ),
    }))
    return example.SerializeToString()


def parse_playlist(example_proto):
    """
    Parse a serialized TFRecord Example back into playlist components.
    
    Deserializes a playlist from TFRecord format and extracts features, track IDs,
    title, and generates attention masks. The parsed data is structured for model input.
    
    Args:
        example_proto (bytes): Serialized TFRecord Example
        
    Returns:
        tuple: (input_dict, labels) where input_dict contains features, track tokens,
               title, and mask; labels contains the target track IDs
    """
    # Define feature schema for parsing
    features = {
        "length": tf.io.FixedLenFeature([], tf.int64),
        "feature_dim": tf.io.FixedLenFeature([], tf.int64),
        "title": tf.io.FixedLenFeature([], tf.string),
        "sequence": tf.io.VarLenFeature(tf.float32),
        "track_ids": tf.io.VarLenFeature(tf.int64),
    }

    # Parse the Example
    parsed = tf.io.parse_single_example(example_proto, features)

    length = parsed["length"]
    feature_dim = parsed["feature_dim"]

    # Convert sparse tensors to dense
    seq = tf.sparse.to_dense(parsed["sequence"])

    # Reshape features back to (length, feature_dim)
    features_seq = tf.reshape(seq, (length, feature_dim))

    # Dense track IDs (tokens)
    track_tokens = tf.sparse.to_dense(parsed["track_ids"])

    # Define Input and Labels (Target) for Sequence Modeling
    input_tokens = track_tokens

    # This is the token we want the model to predict next
    labels = track_tokens 

    # Split the features sequence in the same way if used as input
    input_features = features_seq
    new_length = tf.shape(input_tokens)[0]

    # Create Masks (Needed for padding in subsequent steps like .padded_batch)
    input_mask = tf.ones((new_length,), dtype=tf.int32)

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

    # Get sequence length for bucketing
    def _element_length_func(input_dict, labels):
        return tf.shape(input_dict["track_tokens"])[0]

    # Define Bucket Configs
    bucket_boundaries = [20, 40, 60, 80, 120, 200]
    bucket_batch_sizes = [32, 32, 32, 32, 16, 8, 4]

    # Batch using bucket_by_sequence_length
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

    # Unpack tensors (1 for real song, 0 for padding)
    features = input_dict["features"]
    tokens = input_dict["track_tokens"]
    title = input_dict["title"]
    original_mask = input_dict["mask"] # 

    # Random dropout: keep 50-90% of songs
    keep_prob = tf.random.uniform([], minval=0.5, maxval=0.9)
    random_probs = tf.random.uniform(tf.shape(tokens))

    # Create boolean mask: True = Keep, False = Drop
    is_kept = random_probs < keep_prob

    # Mask features
    feature_mask = tf.cast(is_kept, tf.float32)[:, :, tf.newaxis]

    # Zero out features where is_kept is False
    masked_features = features * feature_mask

    # Mask tokens (set dropped to 0/UNK)
    masked_tokens = tf.where(is_kept, tokens, tf.zeros_like(tokens))

    # Update attention mask to only attend to visible songs! 
    new_mask = tf.cast(original_mask, tf.int32) * tf.cast(is_kept, tf.int32)

    # Return corrupted input with original labels
    model_inputs = {
        "features": masked_features,
        "title": title,
        "mask": new_mask,
        "track_tokens": masked_tokens
    }

    return model_inputs, labels

