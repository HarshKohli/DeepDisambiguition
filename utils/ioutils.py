# Author: Harsh Kohli
# Date created: 1/12/2020

import tensorflow as tf
from transformers import AlbertTokenizer
from utils.entity_obj import EntityObj
from utils.train_sample import TrainSample
import random
import numpy as np


# Assumes schema - query entity_id cononical_name negative_samples
def read_train_inputs(train_file, delimiter, max_len, max_negatives):
    f = open(train_file, 'r', encoding='utf8')
    unique_entity_map, train_samples = {}, []
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    for line in f.readlines():
        info = line.strip().split('\t')
        sentence, entity_id, canonical_name, negative_samples = info[0], info[1], info[2], info[3].split(delimiter)
        if len(negative_samples) < max_negatives:
            negative_samples = negative_samples + [''] * (max_negatives - len(negative_samples))
        else:
            negative_samples = negative_samples[:max_negatives]
        token_info = tokenizer.encode_plus(sentence, max_length=max_len, pad_to_max_length=True)
        sentence_tokens = token_info['input_ids']
        sentence_attention_mask = np.array(token_info['attention_mask'])
        negative_tokens, negative_attention_masks = [], []
        for negative_sample in negative_samples:
            negative_token_info = tokenizer.encode_plus(negative_sample, max_length=max_len, pad_to_max_length=True)
            negative_tokens.append(negative_token_info['input_ids'])
            negative_attention_masks.append(np.array(negative_token_info['attention_mask']))
        train_sample = TrainSample(sentence, entity_id, negative_samples, sentence_tokens, negative_tokens,
                                   sentence_attention_mask, negative_attention_masks)
        train_samples.append(train_sample)
        if entity_id not in unique_entity_map:
            entity_token_info = tokenizer.encode_plus(canonical_name, max_length=max_len, pad_to_max_length=True)
            new_entity = EntityObj(entity_id, canonical_name, entity_token_info['input_ids'],
                                   np.array(entity_token_info['attention_mask']))
            unique_entity_map[entity_id] = new_entity
            new_entity.utterances.append(sentence_tokens)
            new_entity.masks.append(sentence_attention_mask)
        else:
            unique_entity_map[entity_id].utterances.append(sentence_tokens)
            unique_entity_map[entity_id].masks.append(sentence_attention_mask)
    random.shuffle(train_samples)
    return train_samples, unique_entity_map
