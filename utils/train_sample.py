# Author: Harsh Kohli
# Date created: 1/12/2020

class TrainSample:

    def __init__(self, sentence, entity_id, negative_samples, sentence_tokens, negative_tokens):
        self.sentence = sentence
        self.entity_id = entity_id
        self.negative_samples = negative_samples
        self.sentence_tokens = sentence_tokens
        self.negative_tokens = negative_tokens

    def set_embedding(self, embedding):
        self.entity_embedding = embedding