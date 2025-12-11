
"""# Model Architecture

THINGS TO THINK ABOUT:
*   Vocab size?
*   Embedding dim?
*   K number of seed vectors?
*   Better loss function!!!!!
"""

import tensorflow as tf
import tensorflow_hub as hub


"""## Pooling by multi-headed attention
This is so that once we have all the context-enriched embeddings of the songs in the playlist, we can learn how to make a final (pooled) representation of the playlist that we can give to our output projection layer.
"""

class PMA(tf.keras.layers.Layer):
    """
    Pooling by Multihead Attention (Lee et al. 2019)
    k seeds -> output shape (batch, k, dim)
    For playlist classification: we use k=1
    """
    #DOING K=1 FOR NOW - MAY EXPERIMENT WITH K > 1
    def __init__(self, dim, num_heads, k=1, dropout=0.1):
        super().__init__()
        self.k = k
        self.seed_vectors = self.add_weight(
            shape=(k, dim),
            initializer="glorot_uniform",
            trainable=True,
            name="pma_seeds"
        )

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=dim,
            dropout=dropout
        )

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        """
        x:      (batch, seq_len, dim)
        mask:   (batch, seq_len)
        """
        batch_size = tf.shape(x)[0]

        # Expand seeds to batch: (batch, k, dim)
        seeds = tf.repeat(
            tf.expand_dims(self.seed_vectors, 0),
            repeats=batch_size,
            axis=0
        )

        # Convert mask to attention mask (batch, k, seq_len)
        attn_mask = tf.expand_dims(mask, axis=1)

        # Attention: seeds attend to all playlist items
        h = self.mha(
            #query is the seed vectors
            query=seeds,
            #keys and values are the songs
            value=x,
            key=x,
            attention_mask=attn_mask,
            training=training
        )

        #residual connection
        h = seeds + self.dropout(h, training=training)
        h = self.norm(h)

        # If k=1, return shape (batch, dim)
        return h[:, 0, :]     # pick the single pooled vector

"""## Attention block
For learning relationships between songs in the playlist
"""

class SAB(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=dim, dropout=dropout
        )

        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(dim * 4, activation='relu'),
            tf.keras.layers.Dense(dim)
        ])

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training=False):
        attn_mask = tf.expand_dims(mask, axis=1)

        h = self.norm1(x)
        attn_output = self.mha(h, h, attention_mask=attn_mask, training=training)
        x = x + self.dropout(attn_output, training=training)

        h = self.norm2(x)
        ffn_output = self.ff(h)
        x = x + self.dropout(ffn_output, training=training)

        return x

"""## Final Set Transformer Model"""

class SetTransformer(tf.keras.Model):
    def __init__(self, dim=128, num_heads=4, num_layers=3, dropout=0.1, playlist_representation_sz=64):
        super().__init__()
        self.sabs = [
            SAB(dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ]
        
        self.pre_pool_projector = tf.keras.layers.Dense(playlist_representation_sz)
        
        # Can set dim to something else
        self.pma = PMA(playlist_representation_sz, num_heads, k=1, dropout=dropout)

    def call(self, x, mask, training=False):
        for sab in self.sabs:
            x = sab(x, mask, training=training)
            
        x = self.pre_pool_projector(x)
         
        return self.pma(x, mask, training=training)

"""## Song Feature Encoder"""

def build_song_encoder(in_dim, out_dim=128):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(out_dim)
    ])

"""## Full Playlist Prediction Model"""

class PlaylistModel(tf.keras.Model):
    def __init__(self, num_songs, song_feat_dim, emb_dim=128, dropout=0.1, playlist_representation_sz=64):
        super().__init__()

        self.song_encoder = build_song_encoder(song_feat_dim, emb_dim)
        self.set_transformer = SetTransformer(
            dim=emb_dim,
            num_heads=4,
            num_layers=3,
            dropout=dropout,
            playlist_representation_sz=playlist_representation_sz
        )
        # NEW: Layer to project the title embedding (512-dim from USE) down to emb_dim (128-dim)
        self.title_projector = tf.keras.layers.Dense(emb_dim)

        self.final_dense = tf.keras.layers.Dense(num_songs)

    def call(self, song_features, title_emb, mask, training=False):

        # 1. Encode Song Features
        song_emb = self.song_encoder(song_features) # (batch, seq, emb_dim=128)

        # 2. Project Title Embedding to match song_emb dimension
        projected_title_emb = self.title_projector(title_emb) # (batch, emb_dim=128)

        # 3. Prepare for Concatenation
        title_token = tf.expand_dims(projected_title_emb, axis=1) # (batch, 1, emb_dim=128)

        # 4. Concatenate (Axis 2 dimensions now match: 128)
        x = tf.concat([title_token, song_emb], axis=1)

        # ... rest of the function (mask concatenation, set_transformer call) ...
        # Check this out mayhaps
        title_mask = tf.ones((mask.shape[0], 1), dtype=mask.dtype)
        full_mask = tf.concat([title_mask, mask], axis=1)
        pooled = self.set_transformer(x, full_mask, training=training)
        return self.final_dense(pooled)