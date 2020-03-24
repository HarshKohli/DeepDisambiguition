# Author: Harsh Kohli
# Date created: 3/22/2020

import yaml
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.losses import CategoricalCrossentropy
from models import AlbertPlusOutputLayer
from utils.ioutils import read_train_inputs
from utils.tf_utils import initialize_embeddings_from_canonical, initialize_embeddings_from_average_representations
from utils.metrics import get_p, get_mrr

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
    nb_classes = len(unique_entity_map)
    model = AlbertPlusOutputLayer(nb_classes)
    loss_obj = CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)


    @tf.function
    def train_step(tokens, masks, one_hot_labels):
        with tf.GradientTape() as tape:
            logits = model([tokens, masks], training=True)
            loss = loss_obj(y_true=one_hot_labels, y_pred=logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return logits, loss


    @tf.function
    def dev_step(tokens, masks):
        return model([tokens, masks], training=False)


    logfile = open(os.path.join(config['log_dir'], 'hard_negtive_logs.tsv'), 'w', encoding='utf8')

    logfile.write('Epoch' + '\t')
    for p in config['find_p']:
        logfile.write('P@' + str(p) + '\t')
    logfile.write('MRR\n')

    batch_size = config['batch_size']
    batches = [train_samples[i * batch_size:(i + 1) * batch_size] for i in
               range((len(train_samples) + batch_size - 1) // batch_size)]
    dev_batches = [dev_samples[i * batch_size:(i + 1) * batch_size] for i in
                   range((len(dev_samples) + batch_size - 1) // batch_size)]

    for epoch_num in range(config['num_epochs']):
        for batch_num, batch in enumerate(batches):
            tokens = np.asarray([sample.sentence_tokens for sample in batch])
            masks = np.asarray([sample.sentence_attention_mask for sample in batch])
            targets = np.asarray([eid_to_index[sample.entity_id] for sample in batch])
            one_hot_labels = np.eye(nb_classes)[targets]
            probabilites, loss = train_step(tokens, masks, one_hot_labels)
            print(loss)
        print('Running dev set after ' + str(epoch_num + 1) + ' epochs')
        labels, ranks = [], []
        for batch_num, batch in enumerate(dev_batches):
            tokens = np.asarray([sample.sentence_tokens for sample in batch])
            masks = np.asarray([sample.sentence_attention_mask for sample in batch])
            logits = dev_step(tokens, masks)
            ranks.extend(np.argsort(logits))
            labels.extend([eid_to_index[x.entity_id] for x in batch])
        p_to_value, mrr = get_p(config, labels, ranks), get_mrr(labels, ranks)
        logfile.write(str(epoch_num + 1) + '\t')
        for p, value in p_to_value.items():
            print('P@' + str(p) + ' is ' + str(value))
            logfile.write(str(value) + '\t')
        print('MRR is ' + str(mrr))
        logfile.write(str(mrr) + '\n')

    logfile.close()
