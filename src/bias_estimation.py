import pandas as pd
import numpy as np
import numpy as np

# import nltk
from utils import cosine_similarity
from collections import OrderedDict
from clustering import WordCluster
from embeddings import Embedder
from bert_embeddings import BertEmbedder


class BiasEstimation:
    def __init__(
        self,
        tS1: list[str],
        tS2: list[str],
        data_path: str,
        focus_column: str,
        global_embedding: Embedder = None,
        # global_embedding_bert: BertEmbedder = None
    ) -> None:
        self.target1 = tS1
        self.target2 = tS2
        self.global_embedding = global_embedding
        self.data_path = data_path
        self.focus_column = focus_column

        self.reload_variables()

    def reload_variables(self):
        # nltk.download("punkt")
        # nltk.download("averaged_perceptron_tagger")

        self.data = pd.read_csv(self.data_path)
        self.embeddings_bias = OrderedDict()
        self.total_words = []
        self.total_biases = []
        self.c1 = None
        self.c2 = None
        # Centroid of Target concept 1 & 2 (200-dimensional)
        self.novec_words = []

    def NBiasedWords_Bert(self, n: int, bert_embed:BertEmbedder):
        """
        Genrates clusters of n-biased words towards self.target1 and self.target2 using Bert's embedding
        """
        bert_embed.tokenize_sentences(self.data[self.focus_column])
        bert_embed.create_embeddings(self.data[self.focus_column])

        self.c1, self.c2 = bert_embed.prepare_target_sets(self.target1, self.target2)
        for comment in self.data[self.focus_column]:
            for word in comment:
                if not bert_embed.adj_verbs[word]:
                    continue
                if word in self.total_words:
                    continue

                word_embed = bert_embed._stored_vectors[word]
                if word_embed is None:
                    self.novec_words.append(word)
                    continue

                bias = cosine_similarity(word_embed, self.c1) - cosine_similarity(
                    word_embed, self.c2
                )

                self.total_words.append(word)
                self.total_biases.append(bias)


        biased_words = [
            (word, bias) for word, bias in zip(self.total_words, self.total_biases)
        ]
        biased_words.sort(key=lambda x: x[1])

        print("Generating Clusters...")

        wcluster = WordCluster(
            "None",
            bert_embed._stored_vectors,
            self.total_biases,
            biased_words[:n],
            biased_words[-n:],
        )

        cl1, cl2 = wcluster.partition_cluster()
        return (cl1, cl2)


    def NBiasedWords(self, n: int):
        """
        Genrates clusters of n-biased words towards self.target1 and self.target2
        """
        if n < 1:
            return (None, None)
        
        self.c1, self.c2 = self.global_embedding.prepare_target_sets(
            self.target1, self.target2
        )

        curr_emb_model = self.global_embedding.model_type
        for comment in self.data[self.focus_column]:
            adj_verbs = (
                self.global_embedding.find_adjectives(comment)
                if curr_emb_model == "Glove"
                else self.global_embedding.tokenize_sentence(comment)
            )
            for word in adj_verbs:
                if word in self.total_words:
                    continue
                word_embed = (
                    self.global_embedding.get_glove_embedding(word)
                    if curr_emb_model == "Glove"
                    else self.global_embedding.get_bert_embedding(word)
                )
                if word_embed is None:
                    self.novec_words.append(word)
                    continue
                print(word_embed.shape)
                bias = cosine_similarity(word_embed, self.c1) - cosine_similarity(
                    word_embed, self.c2
                )
                self.total_words.append(word)
                self.total_biases.append(bias)

        print(
            f"Couldn't find embeddings for {self.global_embedding.no_embed_count} words"
        )
        biased_words = [
            (word, bias) for word, bias in zip(self.total_words, self.total_biases)
        ]
        biased_words.sort(key=lambda x: x[1])

        print("Generating Clusters...")

        wcluster = WordCluster(
            "None",
            self.global_embedding._stored_vectors,
            self.total_biases,
            biased_words[:n],
            biased_words[-n:],
        )

        cl1, cl2 = wcluster.partition_cluster()

        return (cl1, cl2)
