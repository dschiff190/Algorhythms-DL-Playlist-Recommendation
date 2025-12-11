"""
Loss functions and metrics for the playlist recommendation model.
Includes: WARP loss, negative sampling, and R-Precision metrics.
"""
import tensorflow as tf
import numpy as np

"""## Loss Functions"""

# Hyperparameters for the Hybrid Loss
SEEN_WEIGHT = 0.5   # Lower weight for reconstruction (Autoencoder task)
HIDDEN_WEIGHT = 1.0 # Higher weight for prediction (Recommender task)

def sample_negative(batch_playlist_ids, num_songs, num_neg_samples):
    """
    batch_playlist_ids: (batch, seq_len) int32, padded with -1
    num_songs: int
    num_neg_samples: int

    Returns: (batch, num_neg_samples)
    """

    batch_size = tf.shape(batch_playlist_ids)[0]

    # Build range of all song IDs
    all_ids = tf.range(num_songs, dtype=tf.int64)  # Changed dtype to tf.int64

    # 1) Build a mask of which songs are present in each playlist
    # playlist_mask[b, id] = True if id is in playlist b
    # -------------------------------------------------------------
    # Expand dims to broadcast:
    # batch_playlist_ids       -> (batch, seq_len, 1)
    # all_ids                  -> (1, 1, num_songs)
    equal_matrix = tf.equal(
        tf.expand_dims(batch_playlist_ids, axis=2),
        tf.reshape(all_ids, (1, 1, num_songs))
    )   # (batch, seq_len, num_songs)

    # Ignore padded IDs (-1)
    valid_mask = tf.expand_dims(batch_playlist_ids >= 0, axis=2)
    equal_matrix = equal_matrix & valid_mask

    # Now collapse seq_len to get a final (batch, num_songs) mask
    playlist_contains = tf.reduce_any(equal_matrix, axis=1)   # (batch, num_songs)

    # 2) Allowed songs mask = NOT in playlist
    allowed_mask = ~playlist_contains   # (batch, num_songs)

    # 3) We now need to sample num_neg_samples per batch row.
    # For this, generate random scores for each song and pick top-k among allowed songs.
    # -------------------------------------------------------------
    random_scores = tf.random.uniform((batch_size, num_songs))

    # Set disallowed items to -inf so they are never chosen
    neg_inf = tf.constant(-1e9, dtype=random_scores.dtype)
    masked_scores = tf.where(allowed_mask, random_scores, neg_inf)

    # 4) For each batch row, take the top-k items as negatives.
    # top_k.values ignored; we care about indices
    _, neg_indices = tf.math.top_k(masked_scores, k=num_neg_samples)

    # Shape (batch, num_neg_samples)
    return neg_indices

def warp_loss(logits, batch_playlist_ids, input_ids, num_neg_samples=50, margin=1.0):
    """
    WARP loss that automatically weights samples based on Input vs Target.

    logits: (Batch, Vocab)
    batch_playlist_ids: (Batch, Seq_Len) - The Target (Full Playlist)
    input_ids: (Batch, Seq_Len) - The Input (Masked Playlist)
    """
    batch_size = tf.shape(logits)[0]
    num_songs = tf.shape(logits)[1]

    # 1. Identify Valid Targets (Ignore Padding)
    # Assumes padding is -1. Change to > 0 if using 0 as padding.
    valid_mask = batch_playlist_ids >= 0

    # 2. Calculate Sample Weights internally
    # Case A: Input matches Target => It was SEEN (Autoencoder task)
    is_seen = tf.equal(batch_playlist_ids, input_ids)

    # Case B: Input does NOT match Target => It was HIDDEN (Recommender task)
    # (Since we masked it with 0 in the input)
    is_hidden = tf.not_equal(batch_playlist_ids, input_ids)

    # Create weight tensor
    sample_weights = tf.zeros_like(batch_playlist_ids, dtype=tf.float32)
    sample_weights = tf.where(is_seen, SEEN_WEIGHT, sample_weights)
    sample_weights = tf.where(is_hidden, HIDDEN_WEIGHT, sample_weights)

    # 3. Apply Masking (Remove Padding) to Logits and Weights
    # Get values only for valid songs
    safe_ids = tf.where(valid_mask, batch_playlist_ids, tf.zeros_like(batch_playlist_ids))
    gathered_logits = tf.gather(logits, safe_ids, axis=1, batch_dims=1)

    # Use ragged tensor to collapse padding
    pos_scores_ragged = tf.ragged.boolean_mask(gathered_logits, valid_mask)
    weights_ragged = tf.ragged.boolean_mask(sample_weights, valid_mask)

    # Convert back to dense for map_fn (padded with 0)
    pos_scores_dense = pos_scores_ragged.to_tensor(default_value=0.0)
    weights_dense = weights_ragged.to_tensor(default_value=0.0)
    pos_lengths = pos_scores_ragged.row_lengths()

    # 4. Sample Negatives
    neg_samples = sample_negative(batch_playlist_ids, num_songs, num_neg_samples)

    # 5. Compute Loss per Playlist
    def _compute_single_playlist_loss(args):
        scores_i, weights_i, len_i, neg_idxs_i, logits_i = args

        # Slice to actual valid items
        real_scores = scores_i[:len_i]
        real_weights = weights_i[:len_i]

        def single_item_loss(pair):
            pos_score, weight = pair

            # Get negative scores for this playlist
            neg_scores = tf.gather(logits_i, neg_idxs_i)

            # Check for WARP violations
            violations = neg_scores > (pos_score - margin)
            violation_indices = tf.where(violations)

            def has_violation():
                first_idx = violation_indices[0][0]
                neg_score = neg_scores[first_idx]

                # WARP Rank Approximation
                N = tf.cast(num_songs, tf.float32)
                attempts = tf.cast(first_idx + 1, tf.float32)
                rank = tf.maximum(1.0, (N - 1.0) / attempts)

                L = tf.reduce_sum(1.0 / tf.range(1.0, rank + 1.0))
                loss = L * (margin + neg_score - pos_score)

                # Apply the specific weight (Seen vs Hidden)
                return loss * weight

            return tf.cond(tf.size(violation_indices) > 0, has_violation, lambda: 0.0)

        # Sum loss over all positive items in this playlist
        item_losses = tf.map_fn(single_item_loss, (real_scores, real_weights), fn_output_signature=tf.float32)
        total_loss = tf.reduce_sum(item_losses)

        total_weight = tf.maximum(tf.reduce_sum(real_weights),1)
        return total_loss/ total_weight
    # Map over batch
    batch_losses = tf.map_fn(
        _compute_single_playlist_loss,
        (pos_scores_dense, weights_dense, pos_lengths, neg_samples, logits),
        fn_output_signature=tf.float32,
        parallel_iterations=32
    )

    return tf.reduce_mean(batch_losses)

"""Batch Metrics

"""

def calculate_batch_metrics(logits, target_ids_padded):
    """
    Calculates evaluation metrics for a batch of playlist predictions.

    Args:
        logits: (batch_size, vocab_size) - Raw score predictions from the model.
        target_ids_padded: (batch_size, max_seq_len) - Ground truth song IDs (int), padded with -1.

    Returns:
        dict: containing 'r_precision'
    """
    # Ensure inputs are numpy arrays
    if hasattr(logits, 'numpy'):
        logits = logits.numpy()
    if hasattr(target_ids_padded, 'numpy'):
        target_ids_padded = target_ids_padded.numpy()

    batch_size = logits.shape[0]

    r_precisions = []

    for i in range(batch_size):
        # 1. Extract valid Ground Truth IDs (Remove padding -1)
        true_ids = target_ids_padded[i]
        true_ids = true_ids[true_ids >= 0]

        # If no targets (e.g., empty playlist end), skip
        if len(true_ids) == 0:
            continue

        R = len(true_ids)

        # --- Metric: R-Precision ---
        # "Proportion of relevant tracks in top R recommendations"
        # We need the indices of the top R highest scores in logits

        # Optimization: Use argpartition to find top R without full sort
        # (Negative R because we want the largest values)
        if R >= logits.shape[1]:
            top_r_indices = np.arange(logits.shape[1])
        else:
            # Indices of the top R scores
            top_r_indices = np.argpartition(logits[i], -R)[-R:]

        # Intersection between Truth and Predictions
        relevant_matches = np.intersect1d(true_ids, top_r_indices)

        # R-Precision = (Intersection Count) / R
        r_precisions.append(len(relevant_matches) / R)

    # 2. Aggregate Batch Results
    avg_r_precision = np.mean(r_precisions) if r_precisions else 0.0

    return avg_r_precision

def calculate_hidden_r_precision(logits, target_ids, input_ids):
    """
    Calculates R-Precision ONLY for the songs that were masked (hidden) in the input.

    Args:
        logits: (batch, vocab_size)
        target_ids: (batch, seq_len) - Original full playlist (ground truth)
        input_ids: (batch, seq_len)  - Masked playlist (input with 0s)

    Returns:
        float: Average Hidden R-Precision for the batch
    """
    if hasattr(logits, 'numpy'): logits = logits.numpy()
    if hasattr(target_ids, 'numpy'): target_ids = target_ids.numpy()
    if hasattr(input_ids, 'numpy'): input_ids = input_ids.numpy()

    batch_size = logits.shape[0]
    r_precisions = []

    for i in range(batch_size):
        # 1. Identify Hidden Songs
        # A song is hidden if the Target does not equal Input AND Target is not padding (-1)
        # Note: In your masking logic, masked tokens became 0.
        original = target_ids[i]
        masked = input_ids[i]

        # Filter out padding (-1)
        valid_indices = original >= 0
        original = original[valid_indices]
        masked = masked[valid_indices]

        # Identify which specific IDs were hidden
        # (Where original != masked)
        hidden_mask = (original != masked)
        hidden_ids = original[hidden_mask]
        
        # Identify which specific IDs were SEEN (kept)
        # We want to prevent the model from just predicting what it already sees
        seen_ids = masked[masked > 0] # indices that are not UNK/0

        R = len(hidden_ids)

        # If no songs were hidden for this example, skip
        if R == 0:
            continue

        # 2. Modify Logits to ignore SEEN songs
        # We want to see if the model predicts the MISSING songs, not the visible ones.
        # Create a copy so we don't affect gradients or other calcs
        row_logits = logits[i].copy()
        
        # Set seen songs to -infinity so they are never picked in Top-R
        # (This makes the metric stricter and more accurate for recommendation)
        row_logits[seen_ids] = -float('inf')

        # 3. Get Top R Predictions
        # Use argpartition for efficiency (O(n)) vs argsort (O(n log n))
        if R >= row_logits.shape[0]:
            top_r_indices = np.arange(row_logits.shape[0])
        else:
            # We want indices of the largest R values
            top_r_indices = np.argpartition(row_logits, -R)[-R:]

        # 4. Calculate Intersection
        relevant_matches = np.intersect1d(hidden_ids, top_r_indices)
        
        r_precisions.append(len(relevant_matches) / R)

    if not r_precisions:
        return 0.0

    return np.mean(r_precisions)