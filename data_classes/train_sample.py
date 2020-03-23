# Author: Harsh Kohli
# Date created: 1/12/2020

class TrainSample:

    def __init__(self, sentence, entity_id, negative_samples, sentence_tokens, negative_tokens, sentence_attention_mask,
                 negative_attention_masks):
        self.sentence = sentence
        self.entity_id = entity_id
        self.negative_samples = negative_samples
        self.sentence_tokens = sentence_tokens
        self.negative_tokens = negative_tokens
        self.sentence_attention_mask = sentence_attention_mask
        self.negative_attention_masks = negative_attention_masks

    def set_embedding(self, embedding):
        self.entity_embedding = embedding
