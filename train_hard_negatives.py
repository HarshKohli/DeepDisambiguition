# Author: Harsh Kohli
# Date created: 1/12/2020

import yaml
import tensorflow as tf
import numpy as np
from models import TripletEmbedder
from utils.ioutils import read_train_inputs
from utils.tf_utils import initialize_embeddings

tf.config.experimental_run_functions_eagerly(True)


if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml', 'r'))
    print('Reading Data....')
    train_samples, unique_entity_map = read_train_inputs(config['train_file'], config['negatives_delimiter'],
                                                         config['max_len'], config['max_negative_samples'])
    print('Initializing Embedding Matrix...')
    initialize_embeddings(config, unique_entity_map, train_samples)

    print('starting training')
    embedding_model = TripletEmbedder()
    optimizer = tf.keras.optimizers.Adam()


    @tf.function
    def train_step(positives, negatives, anchors):
        with tf.GradientTape() as tape:
            embeddings, loss = embedding_model(positives, negatives, anchors, config['margin'])
        gradients = tape.gradient(loss, embedding_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, embedding_model.trainable_variables))
        return embeddings, loss

    batch_size = config['batch_size']
    batches = [train_samples[i * batch_size:(i + 1) * batch_size] for i in
               range((len(train_samples) + batch_size - 1) // batch_size)]
    for epoch_num in range(config['num_epochs']):
        for batch_num, batch in enumerate(batches):
            tokens = np.asarray([sample.sentence_tokens for sample in batch])
            negative_tokens = np.asarray([sample.negative_tokens for sample in batch])
            anchor_embeddings = np.asarray([sample.entity_embedding for sample in batch])
            embeddings, loss = train_step(tokens, negative_tokens, anchor_embeddings)
            print(loss)
