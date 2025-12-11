"""
Main training entry point.
Orchestrates data loading, model creation, and training loop.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
from src.data_loader import load_and_prepare_data
from src.model import PlaylistModel
from src.train import run_training


def main(csv_path="playlist_song_features_FINAL_FULL.csv", epochs=10):
    """
    Complete training pipeline.
    
    Args:
        csv_path: Path to the CSV file with playlist data
        epochs: Number of training epochs
    """
    # Load and prepare data
    print("\n" + "="*60)
    print("STEP 1: Loading and Preparing Data")
    print("="*60)
    
    train_dataset, val_dataset, test_dataset, vocab_info = load_and_prepare_data(csv_path)
    
    VOCAB_SIZE = vocab_info['VOCAB_SIZE']
    FEATURE_DIM = vocab_info['FEATURE_DIM']
    
    print(f"\n Data loaded:")
    print(f"   Vocabulary Size: {VOCAB_SIZE}")
    print(f"   Feature Dimension: {FEATURE_DIM}")
    
    # Create model
    print("\n" + "="*60)
    print("STEP 2: Creating Model")
    print("="*60)
    
    # Hyperparameters
    EMB_DIM = 64
    PLAYLIST_REPRESENTATION_SZ = 64

    playlist_model = PlaylistModel(
        num_songs=VOCAB_SIZE,
        song_feat_dim=FEATURE_DIM,
        emb_dim=EMB_DIM,
        playlist_representation_sz=PLAYLIST_REPRESENTATION_SZ
    )
    
    print(f"Model created with embedding dimension: {EMB_DIM}")
    
    # Train model
    print("\n" + "="*60)
    print("STEP 3: Training")
    print("="*60)
    
    history, best_val_loss = run_training(
        model=playlist_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        epochs=epochs,
        checkpoint_dir="./playlist_model_ckpts",
        continue_training=True
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: ./playlist_model_ckpts")
    
    return history, best_val_loss

if __name__ == "__main__":
    main()
