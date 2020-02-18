# Author: Harsh Kohli
# Date created: 2/15/2020

import tensorflow as tf
from transformers import TFAlbertModel
from tensorflow.keras import Model


class TripletEmbedder(Model):
    def __init__(self):
        super(TripletEmbedder, self).__init__()
        self.albert_embedder = TFAlbertModel.from_pretrained('albert-base-v2')

    def call(self, positives, negatives, anchors, margin):
        emb1 = self.albert_embedder(positives)[1]
        negatives_shape = tf.shape(negatives)
        negatives_flattened = tf.reshape(negatives, shape=(negatives_shape[0] * negatives_shape[1], -1))
        emb2 = self.albert_embedder(negatives_flattened)[1]
        emb2 = tf.reshape(emb2, shape=(negatives_shape[0], negatives_shape[1], -1))
        broadcasted_anchors = tf.broadcast_to(tf.expand_dims(anchors, 1), shape=tf.shape(emb2))
        distances = tf.sqrt(tf.reduce_sum(tf.square(broadcasted_anchors - emb2), 2))
        min_distances = tf.math.reduce_min(distances, axis=1)
        positive_distances = tf.sqrt(tf.reduce_sum(tf.square(anchors - emb1), 1))
        loss = tf.reduce_mean(tf.maximum(0., margin + positive_distances - min_distances))
        return emb1, loss
