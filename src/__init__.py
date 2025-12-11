"""
Playlist recommendation system package.
Exposes main classes and utilities for training and inference.
"""

from .model import PMA, SAB, SetTransformer, build_song_encoder, PlaylistModel
from .losses import sample_negative, warp_loss, calculate_batch_metrics, calculate_hidden_r_precision
from .preprocessing import (
    serialize_playlist,
    parse_playlist,
    prepare_batched_dataset,
    create_input_target_with_dropout,
)
from .train import encode_title, train_step, val_step, run_training, optimizer, USE

__all__ = [
    # Model
    "PMA",
    "SAB",
    "SetTransformer",
    "build_song_encoder",
    "PlaylistModel",
    # Losses
    "sample_negative",
    "warp_loss",
    "calculate_batch_metrics",
    "calculate_hidden_r_precision",
    # Preprocessing
    "serialize_playlist",
    "write_tfrecord",
    "parse_playlist",
    "prepare_batched_dataset",
    "create_input_target_with_dropout",
    # Training
    "encode_title",
    "train_step",
    "val_step",
    "run_training",
    "optimizer",
    "USE",
]
