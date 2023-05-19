import gensim.downloader as api
import numpy as np


class Embedder:
    def __init__(self, embedding_model:str = "glove-wiki-gigaword-200") -> None:
        self._model = api.load(embedding_model)
        self._stored_vectors = {}  # This will store all our vocabulary
        self.no_embeds = []  # Tells us how many words cant get an embedding
        self.no_embed_count = 0

    def reinitialize(self):
        self.no_embeds = 0

    def get_embedding(self, word:str):
        if word in self._stored_vectors:
            return self._stored_vectors[word]

        embedding = np.zeros(200)
        if word in self._model.index_to_key:
            embedding = self._model[word]
            self._stored_vectors[word] = embedding
            return embedding

        self.no_embed_count += 1
        return None
    
    def prepare_target_sets(self, TargetSet1:list[str], TargetSet2:list[str]):

        ts1embed = np.array([self.get_embedding(x) for x in TargetSet1])
        ts2embed = np.array([self.get_embedding(x) for x in TargetSet2])

        res1 = np.zeros(200, dtype=np.float32)
        res2 = np.zeros(200, dtype=np.float32)

        for wv1, wv2 in zip(ts1embed, ts2embed):
            res1 = np.add(wv1, res1)
            res2 = np.add(wv2, res2)

        ts1_centroid = res1/len(TargetSet1)
        ts2_centroid = res2/len(TargetSet2)

        return (ts1_centroid, ts2_centroid)
