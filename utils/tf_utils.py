# Author: Harsh Kohli
# Date created: 1/12/2020

from transformers import TFAlbertModel
import tensorflow as tf


def initialize_embeddings(config, unique_entity_map, train_samples):
    batch_size = config['embedding_batch_size']
    unique_entities = list(unique_entity_map.values())
    input_data = [unique_entities[i * batch_size:(i + 1) * batch_size] for i in
                  range((len(unique_entities) + batch_size - 1) // batch_size)]
    model = TFAlbertModel.from_pretrained('albert-base-v2')
    for index, batch in enumerate(input_data):
        model_inputs = [x.canonical_tokens for x in batch]
        embeddings = model(tf.constant(model_inputs))[1]
        for entity, embedding in zip(batch, embeddings):
            entity.set_embedding(embedding)
    for train_sample in train_samples:
        train_sample.set_embedding(unique_entity_map[train_sample.entity_id].entity_embedding)
