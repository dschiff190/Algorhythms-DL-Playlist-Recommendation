# Algo-Rhythms-DL-Playlist-Recommendation

This project is for Brown University's CSCI 1470: Deep Learning course. We use deep learning techniques for the task of playlist recommendation.

## Abstract:

In this project, we attempt using deep learning to address the problem of playlist recommendation. Specifically, we build a system that produces five recommended songs given a playlist of songs with a title. We use a title-conditioned set transformer architecture with pooling by multi-head attention trained on user Spotify playlists with titles and tabular audio features about the songs in the playlists. Furthermore, we create a custom weighted approximate-rank pairwise (WARP) loss function that is in part a reconstruction loss and in part a more traditional WARP loss. While our model is able to train and improve on the metrics we use such as R-precision, our qualitative results are underwhelming and we see indications of mode collapse in our system’s recommendations. We hypothesize that the title conditioning in our model is too weak and that the training data we use may be too noisy or even might not represent a function that is learnable.

## Codebase Overview

### [`data.py`](data.py)

Handles data acquisition and preprocessing. Downloads and merges multiple Spotify datasets, standardizes and normalizes audio features, handles missing values, and writes the final processed CSV used for training and evaluation.

### [`src/data_loader.py`](src/data_loader.py)

Loads the processed CSV, constructs the vocabulary, splits data into train/val/test, serializes playlists to TFRecords, and prepares TensorFlow datasets for model training. Handles batching and applies input masking for the denoising autoencoder objective.

### [`src/preprocessing.py`](src/preprocessing.py)

Provides utilities for serializing/deserializing playlists, batching with sequence length bucketing, and creating masked input/target pairs for training. Ensures efficient data pipeline and correct input formatting for the model.

### [`src/model.py`](src/model.py)

Defines the neural network architecture:

- Implements the Set Transformer with Pooling by Multihead Attention (PMA) and Set Attention Blocks (SAB).
- Encodes both song features and playlist titles.
- Outputs logits over the song vocabulary for playlist completion.

### [`src/losses_and_metrics.py`](src/losses_and_metrics.py)

Implements the custom Weighted Approximate-Rank Pairwise (WARP) loss, which combines reconstruction and recommendation objectives. Provides R-Precision metrics (i.e. number of songs in playlist that were correctly predicted) for evaluating model performance on both seen and hidden playlist items.

### [`src/train.py`](src/train.py)

Contains the training loop, including forward/backward passes, loss computation, metric tracking, checkpointing, and history saving. Supports resuming from checkpoints and tracks best validation loss.

### [`main.py`](main.py)

Main entry point for training. Orchestrates data loading, model instantiation, and the training process. Can be run directly to train the model from scratch or resume from checkpoints.

### [`predict.py`](predict.py)

Provides inference utilities for generating playlist recommendations. Loads trained models and feature dictionaries, supports both top-k and top-p (nucleus) sampling, and includes scripts for running manual "human test" playlists to qualitatively evaluate recommendations.

## Architecture

Our playlist recommendation system is built around a **title-conditioned Set Transformer** architecture. The main components are:

- **Song Encoder:** A feed-forward neural network that embeds tabular audio features for each song.
- **Title Encoder:** Uses the Universal Sentence Encoder (USE) to embed playlist titles, which are projected to match the song embedding dimension.
- **Set Transformer:** Stacks multiple Set Attention Blocks (SAB) to model relationships between songs, followed by Pooling by Multihead Attention (PMA) to produce a fixed-size playlist representation.
- **Output Layer:** A dense layer predicts logits over the entire song vocabulary for playlist completion.

**Custom Loss:**  
We use a hybrid Weighted Approximate-Rank Pairwise (WARP) loss. This loss combines two objectives:

- **Reconstruction loss:** Encourages the model to reconstruct songs that were visible in the input (with a lower weight).
- **Recommendation loss:** Focuses on predicting songs that were masked/hidden from the input (with a higher weight).
  The loss is computed using sampled negatives and a margin ranking approach, with different weights for seen and hidden items to balance reconstruction and recommendation.

The model is trained as a denoising autoencoder: random songs are masked from the input, and the model is tasked with reconstructing the full playlist.

## Hyperparameters

- **Embedding dimension (song & title):** 64
- **Playlist representation size:** 64
- **Set Transformer layers:** 3 SAB layers
- **Attention heads:** 4 per SAB/PMA
- **Dropout rate:** 0.1
- **Optimizer:** Adam, learning rate 3e-4
- **WARP loss:** 50 negative samples per positive, margin = 1.0
- **WARP loss weights:**
  - Seen (reconstruction) weight: 0.5
  - Hidden (prediction) weight: 1.0
- **Batch size:** Variable, bucketed by playlist length (up to 32)
- **Denoising mask:** Randomly hides 10–50% of songs per playlist during training
- **Vocab Size:** 30,000 most frequent tracks seen in full playlist dataset

## How to Run

We recommend using OSCAR (Brown University's computing cluster) with SLURM job submission. All instructions use `.launcher` files for job submission. Modify the virtual environment path in each launcher file before running.

### 0. Setup

Clone the repository and create a Python virtual environment:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Important:** Update the virtual environment path in each `.launcher` file:

```bash
source /path/to/your/venv/bin/activate
```

### 1. Prepare Data

Submit the data preprocessing job to OSCAR:

```sh
sbatch gen_datasets.launcher
```

This will:

- Download multiple Spotify datasets from Kaggle
- Merge and standardize all song features
- Join playlist data with song features
- Write the final CSV: `playlist_song_features_FINAL_FULL.csv`

### 2. Train the Model

Once data preprocessing is complete, submit the training job:

```sh
sbatch train.launcher
```

This will:

- Load and preprocess the data from the CSV
- Build the model
- Train and validate, saving checkpoints to `./playlist_model_ckpts/`
- Save training history to `./playlist_model_ckpts/history.pkl`
- Save additional TFRecord metrics!

To resume training from a previous checkpoint, run the same command again.

### 3. Generate Visualizations

After training completes, visualize the metrics:

```sh
sbatch gen_graphs.launcher
```

This will:

- Load the training history
- Generate a 3-panel figure showing:
  - Training/Validation loss
  - R-Precision on all songs (reconstruction)
  - R-Precision on hidden songs (recommendation)
- Save the plot to `graphs/training_metrics_updated.png` (will create new graphs folder)

### 4. Run Inference & Generate Recommendations

To generate recommendations for custom playlists and run manual "human test" evaluations:

```sh
sbatch predict.launcher
```

This will:

- Load the trained model and cached data
- Run pre-defined human test playlists (Pop, Rap, Country, Jazz, Workout, Chill, Hard Rock, Taylor Swift)
- Print top-10 recommendations for each test playlist using autoregressive generation

  - **First run:** Will compute and cache vocabulary and feature dictionaries as pickle files.
  - **Subsequent runs:** Will load from cache for faster startup.

---

## Project Structure

```
data.py                          # Data download and preprocessing
graphs.py                        # Visualization of training metrics
main.py                          # Training entry point
predict.py                       # Inference and recommendation generation
README.md                        # This file

gen_datasets.launcher            # SLURM job for data.py
gen_graphs.launcher              # SLURM job for graphs.py
train.launcher                   # SLURM job for main.py
predict.launcher                 # SLURM job for predict.py

src/
    __init__.py
    data_loader.py               # Data loading and TFRecord pipeline
    losses_and_metrics.py        # WARP loss and R-Precision metrics
    model.py                     # Model architecture (PMA, SAB, SetTransformer)
    preprocessing.py             # TFRecord serialization and masking
    train.py                     # Training loop and step functions
    requirements.txt             # For installing pip dependencies
```

## Results

_See the final report for quantitative and qualitative results._

## Final Resources

- You can access our weights at [add link eventually]
- You can view our final reports at [add link eventually]
- You can view our poster here [add link eventually]

## Authors

Authors listed in alphabetical order:

* Brendan Rathier - Brown University - brendan_rathier@brown.edu
* Camilo Tamayo-Rousseau - Brown University - camilo_tamayo-rousseau@brown.edu
* Daniel Schiffman - Brown University - daniel_schiffman@brown.edu
* Matias Bronner - Brown University - matias_bronner@brown.edu

## Acknowledgments

We thank the CSCI 1470 teaching staff at Brown University for their guidance and support throughout this project!
