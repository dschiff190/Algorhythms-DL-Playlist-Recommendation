# Algo-Rhythms-DL-Playlist-Recommendation
This project is for Brown University's CSCI 1470: Deep Learning course. We use deep learning techniques for the task of playlist recommendation.

## Abstract:
In this project, we attempt using deep learning to address the problem of playlist recommendation. Specifically, we build a system that produces five recommended songs given a playlist of songs with a title. We use a title-conditioned set transformer architecture with pooling by multi-head attentiontrained on user Spotify playlists with titles and tabular audio features about the songs in the playlists. Furthermore, we create a custom weighted approximate-rank pairwise (WARP) loss function that is in part a reconstruction loss and in part a more traditional WARP loss. While our model is able to train and improve on the metrics we use such as R-precision, our qualitative results are underwhelming and we see indications of mode collapse in our system’s recommendations. We hypothesize that the title conditioning in our model is too weak and that the training data we use may be too noisy or even might not represent a function that is learnable.

## Background:
Creating cohesive music playlists is a core part of the modern listening experience. Thus, providing automated recommendations to users to extend their playlists is a key feature for music-listening platforms like Spotify and Apple Music. To make this feature effective, the system must understand the ”vibe” of a playlist. This task is challenging as the theme of some playlists are quite clear (e.g. ”rap music playlist”) whereas others have more complex themes that cannot be fully captured by features of the playlist such as genre or time period (e.g. ”beach trip playlist”). As music platform users who have felt dissatisfied with the playlist recommendation algorithms, we wanted to dive more deeply into how we could create a model that learns the ”vibe” of a playlist to make our own recommendation system.


## Code-base Overview

* train.py - 
* predict.py - 


## Architecture 

### Hyperparameters
* This model currently uses an ambedding size of , weight of 




## Results 




## Final Resources
* You can access our final report and weights at  
* You can view our poster here at 









