# Author: Harsh Kohli
# Date created: 2/14/2020

class EntityObj:

    def __init__(self, entity_id, canonical_name, canonical_tokens, canonical_attention_mask):
        self.entity_id = entity_id
        self.canonical_name = canonical_name
        self.canonical_tokens = canonical_tokens
        self.canonical_attention_mask = canonical_attention_mask
        self.utterances = []
        self.masks = []

    def set_embedding(self, embedding):
        self.entity_embedding = embedding
