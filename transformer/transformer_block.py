from attr import NOTHING
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.model_selection import train_test_split

"""
## Implement multi head self attention as a Keras layer
"""

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim # 32
        self.num_heads = num_heads  # 2
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads #16
        self.query_dense = layers.Dense(embed_dim) # 32
        self.key_dense = layers.Dense(embed_dim) # 32
        self.value_dense = layers.Dense(embed_dim)  # 32
        self.combine_heads = layers.Dense(embed_dim) # 32

    def attention(self, query, key, value): # 32 2 200 16
        score = tf.matmul(query, key, transpose_b=True) # 32 2 200 200
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1) # 32 2 200 200
        output = tf.matmul(weights, value)  # 32 2 200 16
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs): # 32 200 32
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0] # 32
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim) # 32 200 32
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim) # 32 200 32
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim) # 32 200 32
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim) # 32 2 200 16
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)  # 32 2 200 16
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)  # 32 2 200 16
        attention, weights = self.attention(query, key, value) # 32 2 200 16 , 32 2 200 200
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim) # 32 2 200 16
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim) # 32 200 32
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim) # 32 200 32
        return output # 32 200 32


"""
## Implement a Transformer block as a layer
"""


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.02):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),] # 32 32
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs) # 32 200 32
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # 32 200 32
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
        
"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


"""
## Download and prepare dataset
"""

vocab_size = 30000  # Only consider the top 20k words
maxlen = 200  # Only consider the first 200 words of each movie review
trainDataNew = np.load('data/all_train_vector_Data.npz')
x_train = trainDataNew['x']
y_train = trainDataNew['y']

"""
## Create classifier model using transformer layer
Transformer layer outputs one vector for each time step of our input sequence.
Here, we take the mean across all time steps and
use a feed forward network on top of it to classify text.
"""
embed_dim = 256  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

"""
## Train and Evaluate
"""

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_split=0.1)

print(history)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy', 'val_accuracy'], loc='upper left')
plt.show()

model.save("./all")