# Author: Harsh Kohli
# Date created: 3/22/2020

import yaml
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models import AlbertEmbedder
from utils.ioutils import read_train_inputs
from utils.tf_utils import initialize_embeddings_from_canonical, initialize_embeddings_from_average_representations
from utils.metrics import get_p, get_mrr
from tensorflow.keras.losses import KLDivergence

tf.config.experimental_run_functions_eagerly(True)

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml', 'r'))
    print('Reading Data....')
    train_samples, dev_samples, unique_entity_map = read_train_inputs(config['train_file'],
                                                                      config['negatives_delimiter'],
                                                                      config['max_len'], config['max_negative_samples'],
                                                                      config['num_dev_samples'])
    print('Initializing Embedding Matrix...')
    if config['initialization_strategy'] == 'average_rep':
        index_to_eid, eid_to_index, embedding_matrix = initialize_embeddings_from_average_representations(config,
                                                                                                          unique_entity_map,
                                                                                                          train_samples)
    elif config['initialization_strategy'] == 'from_canonical':
        index_to_eid, eid_to_index, embedding_matrix = initialize_embeddings_from_canonical(config, unique_entity_map,
                                                                                            train_samples)

    print('starting training')
    embedding_model = AlbertEmbedder()
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss_obj = KLDivergence()

    @tf.function
    def train_step(tokens, masks, anchor_embeddings):
        with tf.GradientTape() as tape:
            emb1 = embedding_model(tokens, masks, training=True)
            loss = loss_obj(emb1, anchor_embeddings)
        gradients = tape.gradient(loss, embedding_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, embedding_model.trainable_variables))
        return emb1, loss


    @tf.function
    def dev_step(tokens, masks):
        return embedding_model(tokens, masks, training=False)


    batch_size = config['batch_size']
    batches = [train_samples[i * batch_size:(i + 1) * batch_size] for i in
               range((len(train_samples) + batch_size - 1) // batch_size)]
    dev_batches = [dev_samples[i * batch_size:(i + 1) * batch_size] for i in
                   range((len(dev_samples) + batch_size - 1) // batch_size)]
    for epoch_num in range(config['num_epochs']):
        for batch_num, batch in enumerate(batches):
            tokens = np.asarray([sample.sentence_tokens for sample in batch])
            masks = np.asarray([sample.sentence_attention_mask for sample in batch])
            anchor_embeddings = np.asarray([sample.entity_embedding for sample in batch])
            embeddings, loss = train_step(tokens, masks, anchor_embeddings)
            print(loss)
        print('Running dev set after ' + str(epoch_num + 1) + ' epochs')
        labels, ranks = [], []
        for batch_num, batch in enumerate(dev_batches):
            tokens = np.asarray([sample.sentence_tokens for sample in batch])
            masks = np.asarray([sample.sentence_attention_mask for sample in batch])
            embeddings = dev_step(tokens, masks)
            similarities = cosine_similarity(embeddings, embedding_matrix)
            ranks.extend(np.argsort(similarities))
            labels.extend([eid_to_index[x.entity_id] for x in batch])
        p_to_value, mrr = get_p(config, labels, ranks), get_mrr(labels, ranks)
        for p, value in p_to_value.items():
                print('P@' + str(p) + ' is ' + str(value))
        print('MRR is ' + str(mrr))
