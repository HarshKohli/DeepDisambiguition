# Author: Harsh Kohli
# Date created: 2/15/2020

import tensorflow as tf
from transformers import TFAlbertModel
from tensorflow.keras import Model


class AlbertEmbedder(Model):
    def __init__(self):
        super(AlbertEmbedder, self).__init__()
        self.albert_embedder = TFAlbertModel.from_pretrained('albert-base-v2')

    def call(self, positives, negatives):
        emb1 = self.albert_embedder(positives)[1]
        emb2 = []
        unstacked = tf.unstack(negatives)
        #emb2 = self.albert_embedder(negatives[0])
        # emb2 = []
        for negative in unstacked:
            emb2.append(self.albert_embedder(negative)[1])
        return emb1, emb2
