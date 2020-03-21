# Author: Harsh Kohli
# Date created: 2/15/2020

import tensorflow as tf
import numpy as np
from transformers import TFAlbertModel
from tensorflow.keras import Model


class AlbertEmbedder(Model):
    def __init__(self):
        super(AlbertEmbedder, self).__init__()
        self.albert_embedder = TFAlbertModel.from_pretrained('albert-base-v2')

    def call(self, inputs, masks):
        return tf.reduce_mean(self.albert_embedder(inputs, attention_mask=np.array(masks))[0], axis=1)
