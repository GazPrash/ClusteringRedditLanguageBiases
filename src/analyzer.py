import pandas as pd
from bert_embeddings import BertEmbedder
import spacy
import torch
import numpy as np
from utils import cosine_similarity


class Analyzer:
    def __init__(
        self,
        new_docs: pd.DataFrame,
        focus_column,
        embedder: BertEmbedder,
        ts1: list[str],
        ts2: list[str],
    ) -> None:
        self.new_documents = new_docs
        self.focus_column = focus_column
        self.embedder = embedder  # use the same embedder isntance that you used on the training data
        self.ts1, self.ts2 = self.embedder.prepare_target_sets(ts1, ts2)

    def prepare_embeddings(self):

        for i, sentence in enumerate(self.new_documents[self.focus_column]):

            tokens = self.embedder.tokenize(sentence)
            token_ids = self.embedder.convert_tokens_to_ids(tokens)
            input_tensor = torch.tensor([token_ids])

            try:
                with torch.no_grad():
                    self.embedder._model.eval()
                    outputs = self.embedder._model(input_tensor)
            except Exception as e:
                print(e)
                print(
                    f"Skipping the #{i} comment",
                    f"Total Words: {len(sentence.split(' '))}",
                )
                continue

            word_vectors = outputs.last_hidden_state.squeeze(0)
            for word, vec in zip(tokens, word_vectors):
                if word not in self.embedder._stored_vectors:
                    self.embedder._stored_vectors[word] = np.array(vec)

    def calculate_biases(self):
        s_biases = []
        for sentence in self.new_documents[self.focus_column]:
            focus_words = 0
            bias = 0
            doc = self.spacy_nlp(sentence)
            for token in doc:
                if token.pos_ == "ADJ":
                    word_vec = self.embedder._stored_vectors[token.text]
                    bias += cosine_similarity(word_vec, self.c1) - cosine_similarity(
                    word_vec, self.c2
                    )
                    focus_words += 1
            s_biases.append(bias/len(focus_words))

        return s_biases