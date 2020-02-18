# Author: Harsh Kohli
# Date created: 1/12/2020

import yaml
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from architecture.albert_embedding_model import AlbertEmbedder
from utils.ioutils import read_train_inputs
from utils.tf_utils import initialize_embeddings

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml', 'r'))
    print('Reading Data....')
    train_samples, unique_entity_map = read_train_inputs(config['train_file'], config['negatives_delimiter'],
                                                         config['max_len'])
    print('Initializing Embedding Matrix...')
    initialize_embeddings(config, unique_entity_map, train_samples)

    # loss_object = tfa.losses.contrastive_loss()
    print('starting training')
    embedding_model = AlbertEmbedder()
    tf.config.experimental_run_functions_eagerly(True)

    @tf.function
    def train_step(positives, negatives):
        with tf.GradientTape() as tape:
            emb1, emb2 = embedding_model(positives, negatives)
        return emb1, emb2
        #     loss = loss_object(labels, predictions)
        # gradients = tape.gradient(loss, model.trainable_variables)
        # optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        #
        # train_loss(loss)
        # train_accuracy(labels, predictions)


    batch_size = config['batch_size']
    batches = [train_samples[i * batch_size:(i + 1) * batch_size] for i in
               range((len(train_samples) + batch_size - 1) // batch_size)]
    for epoch_num in range(config['num_epochs']):
        for batch_num, batch in enumerate(batches):
            tokens = np.asarray([sample.sentence_tokens for sample in batch])
            negative_tokens = np.asarray([sample.negative_tokens for sample in batch])
            emb1, emb2 = train_step(tokens, negative_tokens)
            print(batch_num)
