import pandas as pd
import numpy as np
import numpy as np
import nltk
from utils import cosine_similarity
from collections import OrderedDict
from clustering import WordCluster
from embeddings import Embedder


class BiasEstimation:
    def __init__(
        self,
        tS1: list[str],
        tS2: list[str],
        global_embedding: Embedder,
        data_path: str,
        focus_column: str,
    ) -> None:
        self.target1 = tS1
        self.target2 = tS2
        self.global_embedding = global_embedding
        self.data_path = data_path
        self.focus_column = focus_column

        self.reload_variables()

    def reload_variables(self):
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")

        self.data = pd.read_csv(self.data_path)
        self.embeddings_bias = OrderedDict()
        self.total_words = []
        self.total_biases = []
        # Centroid of Target concept 1 & 2 (200-dimensional)
        self.c1, self.c2 = self.global_embedding.prepare_target_sets(
            self.target1, self.target2
        )


    def find_adjectives(self, sentence: str):
        """
        For identifying adjectives in a sentence
        """
        sentence = nltk.word_tokenize(sentence)
        semantics = nltk.pos_tag(sentence)
        adjectives = []
        for word, semantic in semantics:
            if semantic == "JJ" or semantic == "JJR" or semantic == "JJS":
                adjectives.append(word)

        return adjectives

    def NBiasedWords(self, n: int):
        """
        Genrates clusters of n-biased words towards self.target1 and self.target2
        """
        if (n < 1):return (None, None) 

        for comment in self.data[self.focus_column]:
            adj_nouns = self.find_adjectives(comment)
            for word in adj_nouns:
                if word in self.total_words:
                    continue
                word_embed = self.global_embedding.get_embedding(word)
                if word_embed is None:
                    continue

                bias = cosine_similarity(word_embed, self.c1) - cosine_similarity(word_embed, self.c2)
                self.total_words.append(word)
                self.total_biases.append(bias)


        print(f"Couldn't find embeddings for {self.global_embedding.no_embed_count} words")
        biased_words = [(word, bias) for word, bias in zip(self.total_words, self.total_biases)]
        biased_words.sort(key=lambda x: x[1])

        print("Generating Clusters...")

        wcluster = WordCluster(
            "None", self.global_embedding._stored_vectors, self.total_biases, biased_words[:n], biased_words[-n:]
        )

        cl1, cl2 = wcluster.partition_cluster()

        return (cl1, cl2)
