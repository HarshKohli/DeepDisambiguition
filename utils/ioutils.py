# Author: Harsh Kohli
# Date created: 1/12/2020

import tensorflow as tf
from transformers import AlbertTokenizer
from utils.entity_obj import EntityObj
from utils.train_sample import TrainSample
import random
import numpy as np



# Assumes schema - query entity_id cononical_name negative_samples
def read_train_inputs(train_file, delimiter, max_len):
    f = open(train_file, 'r', encoding='utf8')
    unique_entity_map, train_samples = {}, []
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    for line in f.readlines():
        info = line.split('\t')
        sentence, entity_id, canonical_name, negative_samples = info[0], info[1], info[2], info[3].split(delimiter)
        sentence_tokens = np.squeeze(tf.constant(tokenizer.encode(sentence, max_length=max_len, pad_to_max_length=True))[None, :])
        negative_tokens = []
        for negative_sample in negative_samples:
            negative_tokens.append(np.squeeze(tf.constant(tokenizer.encode(negative_sample, max_length=max_len, pad_to_max_length=True))[None, :]))
        train_sample = TrainSample(sentence, entity_id, negative_samples, sentence_tokens, negative_tokens)
        train_samples.append(train_sample)
        if entity_id not in unique_entity_map:
            # unique_entity_map[entity_id] = canonical_name
            entity_tokens = np.squeeze(tokenizer.encode(canonical_name, max_length=max_len, pad_to_max_length=True))
            unique_entity_map[entity_id] = EntityObj(entity_id, canonical_name, entity_tokens)
    random.shuffle(train_samples)
    return train_samples, unique_entity_map
