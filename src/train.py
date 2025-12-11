"""
Training loop and step functions for the playlist recommendation model.
Handles forward pass, loss computation, gradient updates, and checkpointing.
"""
import tensorflow as tf
import tensorflow_hub as hub
import tqdm
import sys
import os
import pickle
from .losses import warp_loss, calculate_batch_metrics, calculate_hidden_r_precision


#Pre-trained text embedding
USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def encode_title(title_texts):
    return USE(title_texts)   # shape (batch, 512)

# Optimizer
optimizer = tf.keras.optimizers.Adam(3e-4)

@tf.function
def train_step(model, batch_song_feats, batch_title_texts, batch_mask, batch_input_ids, labels, calculate_r_precision):

    with tf.GradientTape() as tape:
        title_emb = encode_title(batch_title_texts)

        # PlaylistModel call method takes song_features, title_emb, mask
        logits = model(batch_song_feats, title_emb, batch_mask, training=True)

        # The `labels` tensor (track tokens) is passed to `warp_loss` as `batch_playlist_ids`
        loss = warp_loss(logits, labels, batch_input_ids, num_neg_samples=50, margin=1.0)

        r_precision = tf.constant(-1.0, dtype=tf.float32) # Initialize as a tensor
        if calculate_r_precision:
          r_precision = tf.py_function(func=calculate_batch_metrics, inp=[logits, labels], Tout=tf.float32)
          hidden_r_precision = tf.py_function(
                func=calculate_hidden_r_precision, 
                inp=[logits, labels, batch_input_ids], 
                Tout=tf.float32
          )

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, r_precision, hidden_r_precision # Return both loss and r_precision


@tf.function
def val_step(model, batch_song_feats, batch_title_texts, batch_mask, batch_input_ids, labels, calculate_r_precision):
    # Encode titles
    title_emb = encode_title(batch_title_texts)

    # Forward pass (training=False is crucial for Dropout/BatchNorm layers)
    logits = model(batch_song_feats, title_emb, batch_mask, training=False)

    # Compute Loss
    loss = warp_loss(logits, labels, batch_input_ids,num_neg_samples=50, margin=1.0)

    # Compute Metrics
    r_precision = tf.constant(-1.0, dtype=tf.float32)
    if calculate_r_precision:
        r_precision = tf.py_function(func=calculate_batch_metrics, inp=[logits, labels], Tout=tf.float32)
        
        hidden_r_precision = tf.py_function(
            func=calculate_hidden_r_precision, 
            inp=[logits, labels, batch_input_ids], 
            Tout=tf.float32
        )

    return loss, r_precision, hidden_r_precision


def run_training(model, train_dataset, val_dataset, test_dataset, epochs=10, checkpoint_dir="./playlist_model_ckpts", continue_training=True):
    """
    Complete training loop with checkpointing and history tracking.
    
    Args:
        model: PlaylistModel instance
        train_dataset: tf.data.Dataset with training batches
        val_dataset: tf.data.Dataset with validation batches
        test_dataset: tf.data.Dataset with test batches
        epochs: Number of epochs to train
        checkpoint_dir: Directory to save checkpoints
        continue_training: Whether to restore from previous checkpoint
    
    Returns:
        tuple: (history, best_val_loss)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(checkpoint_dir, "history.pkl")

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=checkpoint_dir, max_to_keep=5)

    # Handle checkpoint restoration for continue training
    start_epoch = 1
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_r_prec": [],
        "train_r_prec": [],
        "val_r_prec_hidden": [],
        "train_r_prec_hidden": [],
    }
    best_val_loss = float('inf')

    if continue_training:
      # model checkpoint
      latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
      if latest_checkpoint:
         checkpoint.restore(latest_checkpoint)
          # Extract epoch number from checkpoint name
         try:
            start_epoch = int(latest_checkpoint.split('-')[-1])
            print(f"Resuming from epoch {start_epoch}")
         except:
              print("Could not determine start epoch, starting from 0")
      else:
          print("No checkpoint found, starting fresh")
      # load metrics history
      if os.path.exists(history_path):
          try:
              with open(history_path, "rb") as f:
                  history = pickle.load(f)

              # Recalculate best_val_loss from the loaded history
              if history["val_loss"]:
                  best_val_loss = min(history["val_loss"])
                  print(f"Restored history. Previous Best Val Loss: {best_val_loss:.4f}")
              else:
                  print("ℹHistory file loaded but empty.")
          except Exception as e:
              print(f"Error loading history file: {e}")
      else:
          print("No history file found. Starting new history.")
          best_val_loss = float('inf')

    r_precision_frequency = 1 # Calculate R-precision every N batches

    for current_epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs), desc="Training Progress", position=0,file=sys.stdout):
        print(f"Epoch: {current_epoch}")
        epoch_train_loss = []
        epoch_val_loss = []
        r_prec_val = []
        r_prec_train = []
        r_prec_val_hidden = []
        r_prec_train_hidden = []
        #look into shuffling without messing up the batches
        for i, (inputs, labels) in enumerate(train_dataset):
            # Unpack the dictionary keys to get the positional arguments
            batch_song_feats = inputs["features"]
            batch_title_texts = inputs["title"]
            batch_mask = inputs["mask"]
            batch_input_ids = inputs['track_tokens']
            
            # Run the corrected train_step
            train_loss, r_precision_train, hidden_r_prec_train = train_step(
                model,
                batch_song_feats,
                batch_title_texts,
                batch_mask,
                batch_input_ids,
                labels,
                True # Pass `calculate_r_precision` flag
            )
            epoch_train_loss.append(train_loss)
            r_prec_train.append(r_precision_train)
            r_prec_train_hidden.append(hidden_r_prec_train)
            
            # Print informative metrics
            if i % 50 == 0:
                print(f"Batch: {i} - Train Loss: {train_loss:.4f}, R-Precision: {r_precision_train:.4f}, R-Precision(hidden): {hidden_r_prec_train:.4f}")
                
                
        for i, (inputs, labels) in enumerate(val_dataset):
            batch_song_feats = inputs["features"]
            batch_title_texts = inputs["title"]
            batch_mask = inputs["mask"]
            
            # Run val_step
            v_loss, r_precision_val, hidden_r_prec_val = val_step(
                model,
                batch_song_feats,
                batch_title_texts,
                batch_mask,
                inputs['track_tokens'], # Added missing batch_input_ids (masked tokens)
                labels, # Now correctly positioned
                calculate_r_precision=True
            )
            epoch_val_loss.append(v_loss)
            r_prec_val.append(r_precision_val)
            r_prec_val_hidden.append(hidden_r_prec_val)
            
        avg_train_loss = tf.reduce_mean(epoch_train_loss).numpy()
        avg_train_r_prec = tf.reduce_mean(r_prec_train).numpy()
        avg_train_r_prec_hidden = tf.reduce_mean(r_prec_train_hidden).numpy()
        avg_val_loss = tf.reduce_mean(epoch_val_loss).numpy()
        avg_val_r_prec = tf.reduce_mean(r_prec_val).numpy()
        avg_val_r_prec_hidden = tf.reduce_mean(r_prec_val_hidden).numpy()
        
        history["train_loss"].append(avg_train_loss)
        history["train_r_prec"].append(avg_train_r_prec)
        history["val_loss"].append(avg_val_loss)
        history["val_r_prec"].append(avg_val_r_prec)
        history["train_r_prec_hidden"].append(avg_train_r_prec_hidden)
        history["val_r_prec_hidden"].append(avg_val_r_prec_hidden)
        
        
        
        print(f"Epoch {current_epoch} - Train Loss: {avg_train_loss:.4f} Train R-Prec: {avg_train_r_prec:4f} Train R-Prec (hidden): {avg_train_r_prec_hidden:4f} Val Loss: {avg_val_loss:.4f} Val R-Precision: {avg_val_r_prec:.4f} Val R-Precision: {avg_val_r_prec_hidden:.4f}")
        if avg_val_loss < best_val_loss:
            diff = best_val_loss - avg_val_loss
            # print(f"✅ Validation loss improved by {diff:.4f} (New Best: {avg_val_loss:.4f}). Saving model...")
            
            # Update best tracker
            best_val_loss = avg_val_loss
            
            # Save the checkpoint
            save_path = checkpoint_manager.save()

        # save history every epoch
        with open(history_path, "wb") as f:
            pickle.dump(history, f)
    
    return history, best_val_loss







