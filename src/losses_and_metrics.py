"""
Loss Functions and Evaluation Metrics for Playlist Recommendation

This module implements the WARP (Weighted Approximate-Rank Pairwise) loss function
for training and R-Precision metrics for evaluating playlist recommendation performance.
"""
import tensorflow as tf
import numpy as np

# Hyperparameters for the Hybrid Loss

# Lower weight for reconstruction (Autoencoder task)
SEEN_WEIGHT = 0.5   

# Higher weight for prediction (Recommender task)
HIDDEN_WEIGHT = 1.0 

def sample_negative(batch_playlist_ids, num_songs, num_neg_samples):
    """
    Sample negative examples for WARP loss computation.
    
    Randomly selects songs that are NOT in the playlist to serve as negative samples.
    Sampling is done per playlist, ensuring no negative samples are songs already
    in the target playlist.
    
    Args:
        batch_playlist_ids (tf.Tensor): Target song IDs of shape (batch, seq_len), padded with -1
        num_songs (int): Total vocabulary size
        num_neg_samples (int): Number of negative samples to draw per playlist
        
    Returns:
        tf.Tensor: Negative sample indices of shape (batch, num_neg_samples)
    """

    batch_size = tf.shape(batch_playlist_ids)[0]

    # Build range of all song IDs -> (1, 1, num_songs))
    all_ids = tf.range(num_songs, dtype=tf.int64) 

    # batch_playlist_ids -> (batch, seq_len, 1)
    equal_matrix = tf.equal(
        tf.expand_dims(batch_playlist_ids, axis=2),
        tf.reshape(all_ids, (1, 1, num_songs))
    )   

    # Exclude padding from the mask
    valid_mask = tf.expand_dims(batch_playlist_ids >= 0, axis=2)
    equal_matrix = equal_matrix & valid_mask

    # Collapse to (batch, num_songs)
    playlist_contains = tf.reduce_any(equal_matrix, axis=1)
    allowed_mask = ~playlist_contains

    # Sample negatives by choosing top-k unallowed songs by random score
    random_scores = tf.random.uniform((batch_size, num_songs))

    # Set disallowed items to -inf so they are never chosen
    neg_inf = tf.constant(-1e9, dtype=random_scores.dtype)
    masked_scores = tf.where(allowed_mask, random_scores, neg_inf)

    # For each batch row, take the top-k items as negatives.
    _, neg_indices = tf.math.top_k(masked_scores, k=num_neg_samples)
    return neg_indices

def warp_loss(logits, batch_playlist_ids, input_ids, num_neg_samples=50, margin=1.0):
    """
    WARP loss that automatically weights samples based on Input vs Target.

    Computes Weighted Approximate-Rank Pairwise loss that weights samples based
    on whether they were visible (SEEN) or hidden (HIDDEN) in the input.
    
    Args:
        logits (tf.Tensor): Model predictions of shape (batch, vocab_size)
        batch_playlist_ids (tf.Tensor): Target full playlist of shape (batch, seq_len), padded with -1
        input_ids (tf.Tensor): Input playlist with masking of shape (batch, seq_len), padded with -1
        num_neg_samples (int): Number of negative samples per positive (default: 50)
        margin (float): Margin for WARP ranking violation (default: 1.0)
        
    Returns:
        tf.Tensor: Scalar loss value
    """
    batch_size = tf.shape(logits)[0]
    num_songs = tf.shape(logits)[1]

    # Identify valid targets (skip padding)
    valid_mask = batch_playlist_ids >= 0

    # Weight samples by visibility: seen items get lower weight (reconstruction),
    # hidden items get higher weight (recommendation task)
    is_seen = tf.equal(batch_playlist_ids, input_ids)

    #  Input does NOT match Target => It was HIDDEN (Recommender task); Mask with 0 in input
    is_hidden = tf.not_equal(batch_playlist_ids, input_ids)

    # Create weight tensor
    sample_weights = tf.zeros_like(batch_playlist_ids, dtype=tf.float32)
    sample_weights = tf.where(is_seen, SEEN_WEIGHT, sample_weights)
    sample_weights = tf.where(is_hidden, HIDDEN_WEIGHT, sample_weights)

    # Apply Masking (Remove Padding) to Logits and Weights (get values only for valid songs)
    safe_ids = tf.where(valid_mask, batch_playlist_ids, tf.zeros_like(batch_playlist_ids))
    gathered_logits = tf.gather(logits, safe_ids, axis=1, batch_dims=1)

    # Use ragged tensor to collapse padding
    pos_scores_ragged = tf.ragged.boolean_mask(gathered_logits, valid_mask)
    weights_ragged = tf.ragged.boolean_mask(sample_weights, valid_mask)

    # Convert back to dense for map_fn (padded with 0)
    pos_scores_dense = pos_scores_ragged.to_tensor(default_value=0.0)
    weights_dense = weights_ragged.to_tensor(default_value=0.0)
    pos_lengths = pos_scores_ragged.row_lengths()

    # Sample Negatives
    neg_samples = sample_negative(batch_playlist_ids, num_songs, num_neg_samples)

    # Compute Loss per Playlist
    def _compute_single_playlist_loss(args):
        scores_i, weights_i, len_i, neg_idxs_i, logits_i = args

        # Slice to actual valid items
        real_scores = scores_i[:len_i]
        real_weights = weights_i[:len_i]

        def single_item_loss(pair):
            pos_score, weight = pair

            # Get negative scores for this playlist & check for WARP violations
            neg_scores = tf.gather(logits_i, neg_idxs_i)
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

    Calculates r-precision - the proportion of relevant tracks in top R recommendations

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
        # Extract valid Ground Truth IDs (Remove padding -1)
        true_ids = target_ids_padded[i]
        true_ids = true_ids[true_ids >= 0]

        # If no targets (e.g., empty playlist end), skip
        if len(true_ids) == 0:
            continue

        R = len(true_ids)

        # Retrieve the indices of the top R highest scores in logits
        # (Negative R because we want the largest values)
        if R >= logits.shape[1]:
            top_r_indices = np.arange(logits.shape[1])
        else:
            top_r_indices = np.argpartition(logits[i], -R)[-R:]
        relevant_matches = np.intersect1d(true_ids, top_r_indices)

        # R-Precision = (Intersection Count) / R
        r_precisions.append(len(relevant_matches) / R)

    # Aggregate Batch Results
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
        # Identify Hidden Songs (if target != input & input is not padded)
        original = target_ids[i]
        masked = input_ids[i]

        # Filter out padding (-1)
        valid_indices = original >= 0
        original = original[valid_indices]
        masked = masked[valid_indices]

        # Identify which specific IDs were hidden (Where original != masked)
        hidden_mask = (original != masked)
        hidden_ids = original[hidden_mask]
        
        # Identify which specific IDs were SEEN (kept)
        seen_ids = masked[masked > 0] # indices that are not UNK/0

        R = len(hidden_ids)

        # If no songs were hidden for this example, skip
        if R == 0:
            continue

        # Modify Logits to ignore SEEN songs (create copy to not impact gradients)
        row_logits = logits[i].copy()
        
        # Set seen songs to -infinity so they are never picked in Top-R
        row_logits[seen_ids] = -float('inf')

        # Get Top R Predictions
        if R >= row_logits.shape[0]:
            top_r_indices = np.arange(row_logits.shape[0])
        else:
            top_r_indices = np.argpartition(row_logits, -R)[-R:]

        relevant_matches = np.intersect1d(hidden_ids, top_r_indices)
        r_precisions.append(len(relevant_matches) / R)

    if not r_precisions:
        return 0.0

    return np.mean(r_precisions)