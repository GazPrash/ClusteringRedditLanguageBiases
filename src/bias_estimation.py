import gensim.downloader as api
import pandas as pd
import numpy as np
import numpy as np
import nltk
from utils import cosine_similarity
from collections import OrderedDict
from clustering import WordCluster


class Embedder:
    def __init__(self, embedding_model:str = "glove-wiki-gigaword-200") -> None:
        self._model = api.load(embedding_model)
        self._stored_vectors = {}  # This will store all our vocabulary
        self.no_embeds = 0  # Tells us how many words cant get an embedding

    def reinitialize(self):
        self.no_embeds = 0

    def get_embedding(self, word:str):
        embedding = np.zeros(200)
        if word in self._model.index_to_key:
            embedding = self._model[word]
            self._stored_vectors[word] = embedding
            return embedding

        self.no_embeds += 1
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


def find_adjectives(sentence:str):
    """
    For identifying adjectives in a sentence
    """
    nltk.download("punkt")
    sentence = nltk.word_tokenize(sentence)
    semantics = nltk.pos_tag(sentence)
    adjectives = []
    for word, semantic in semantics:
        if semantic == "JJ" or semantic == "JJR" or semantic == "JJS":
            adjectives.append(word)

    return adjectives


def SelectNBiasedWords(data_path:str, n:int = 500):
    data = pd.read_csv(data_path)
    TargetSet1 = ["sister" , "female" , "woman" , "girl" , "daughter" , "she" , "hers" , "her"]
    TargetSet2   = ["brother" , "male" , "man" , "boy" , "son" , "he" , "his" , "him"]

    global_embedding = Embedder() 
    global_embedding.reinitialize()
    c1, c2 = global_embedding.prepare_target_sets(TargetSet1, TargetSet2)

    print("Target Centroids calculated...")

    # Word embeddings of the words (200-dimensional)
    wordembed_bias = OrderedDict()
    total_words = []
    total_biases = []
    # Centroid vectors for the male and female gender groups (200-dimensional)
    # centroid_male = np.array([0.1, 0.2, ..., 0.3])
    # centroid_female = np.array([0.5, 0.6, ..., 0.7])
    print("Generating Vocab & Calculating Biases...")

    for comment in data.Comment:
        adj_nouns = find_adjectives(comment)
        for word in adj_nouns:
            if word in wordembed_bias : continue
            word_embed = global_embedding.get_embedding(word)
            if (word_embed is None) : continue

            bias = cosine_similarity(word_embed, c1) - cosine_similarity(word_embed, c2)
            total_words.append(word)
            wordembed_bias[word] = word_embed
            total_biases.append(bias)

    print(f"Couldn't find embeddings for {global_embedding.no_embeds} words")

    biased_words = [(word, bias) for word, bias in zip(total_words, total_biases)]
    biased_words.sort(key=lambda x : x[1])

    print("Generating Clusters...")
    wcluster = WordCluster("None", wordembed_bias, total_biases, biased_words[:500], biased_words[-500:])
    cl1, cl2 = wcluster.partition_cluster()