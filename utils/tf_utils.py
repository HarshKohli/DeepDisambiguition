# Author: Harsh Kohli
# Date created: 1/12/2020

from transformers import TFAlbertModel
import tensorflow as tf
import numpy as np


def initialize_embeddings_from_canonical(config, unique_entity_map, train_samples):
    batch_size = config['embedding_batch_size']
    unique_entities = list(unique_entity_map.values())
    input_data = [unique_entities[i * batch_size:(i + 1) * batch_size] for i in
                  range((len(unique_entities) + batch_size - 1) // batch_size)]
    model = TFAlbertModel.from_pretrained('albert-base-v2')
    for index, batch in enumerate(input_data):
        model_inputs = [x.canonical_tokens for x in batch]
        masks = [x.canonical_attention_mask for x in batch]
        embeddings = tf.reduce_mean(model(tf.constant(model_inputs), attention_mask=np.array(masks))[0], axis=1)
        for entity, embedding in zip(batch, embeddings):
            entity.set_embedding(embedding)
    for train_sample in train_samples:
        train_sample.set_embedding(unique_entity_map[train_sample.entity_id].entity_embedding)


def initialize_embeddings_from_average_representations(config, unique_entity_map, train_samples):
    batch_size = config['embedding_batch_size']
    unique_entities = list(unique_entity_map.values())
    model = TFAlbertModel.from_pretrained('albert-base-v2')
    for unique_entity in unique_entities:
        embedding = []
        utterances, masks = unique_entity.utterances, unique_entity.masks
        input_data = [utterances[i * batch_size:(i + 1) * batch_size] for i in
                      range((len(utterances) + batch_size - 1) // batch_size)]
        input_masks = [masks[i * batch_size:(i + 1) * batch_size] for i in
                       range((len(masks) + batch_size - 1) // batch_size)]
        for index, (batch, mask) in enumerate(zip(input_data, input_masks)):
            if index == 0:
                embedding = np.asarray(
                    tf.reduce_sum(tf.reduce_mean(model(tf.constant(batch), attention_mask=np.array(mask))[0], axis=1),
                                  axis=0))
            else:
                embedding = np.add(embedding, np.asarray(
                    tf.reduce_sum(tf.reduce_mean(model(tf.constant(batch), attention_mask=np.array(mask))[0], axis=1),
                                  axis=0)))
        embedding = np.divide(embedding, len(utterances))
        unique_entity.set_embedding(embedding)
    for train_sample in train_samples:
        train_sample.set_embedding(unique_entity_map[train_sample.entity_id].entity_embedding)
