# Algo-Rhythms-DL-Playlist-Recommendation
This project is for Brown University's CSCI 1470: Deep Learning course. We use deep learning techniques for the task of playlist recommendation.

# Background:



# Abstract:
In this project, we attempt using deep learning to address the problem of playlist recommendation. Specifically, we build a system that produces five recommended songs given a playlist of songs with a title. We use a title-conditioned set transformer architecture with pooling by multi-head attentiontrained on user Spotify playlists with titles and tabular audio features about the songs in the playlists. Furthermore, we create a custom weighted approximate-rank pairwise (WARP) loss function that is in part a reconstruction loss and in part a more traditional WARP loss. While our model is able to train and improve on the metrics we use such as R-precision, our qualitative results are underwhelming and we see indications of mode collapse in our system’s recommendations. We hypothesize that the title conditioning in our model is too weak and that the training data we use may be too noisy or even might not represent a function that is learnable.


