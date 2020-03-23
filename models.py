# Author: Harsh Kohli
# Date created: 2/15/2020

import tensorflow as tf
import numpy as np
from transformers import TFAlbertModel
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class AlbertEmbedder(Model):
    def __init__(self):
        super(AlbertEmbedder, self).__init__()
        self.albert_embedder = TFAlbertModel.from_pretrained('albert-base-v2')

    def call(self, features, training):
        inputs, masks = features[0], features[1]
        return tf.reduce_mean(self.albert_embedder(inputs, attention_mask=np.array(masks))[0], axis=1)


class AlbertPlusOutputLayer(Model):
    def __init__(self, nb_classes):
        super(AlbertPlusOutputLayer, self).__init__()
        self.albert_embedder = TFAlbertModel.from_pretrained('albert-base-v2')
        self.output_layer = Dense(nb_classes)

    def call(self, features, training):
        tokens, masks = features[0], features[1]
        albert_embedding = tf.reduce_mean(self.albert_embedder(tokens, attention_mask=np.array(masks))[0], axis=1)
        return tf.nn.softmax(self.output_layer(albert_embedding))
