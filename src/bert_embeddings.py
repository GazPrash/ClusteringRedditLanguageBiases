import spacy
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.decomposition import PCA


class BertEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self._model = BertModel.from_pretrained(model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self._stored_vectors = {}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.vecdim = 768
        self.adj_verbs = {}
        self.count_vocab = {}
        self.duplicates = {}

    def tokenize_sentences(self, documents: list[str]):
        # Trying to have a ledger which tells out of the total words
        # which of them are adjectives and verbs
        for sentence in documents:
            doc = self.spacy_nlp(sentence)
            for token in doc:
                if token.pos_ == "ADJ":
                    self.adj_verbs[token.text] = True
                    if token.text not in self.count_vocab:
                        self.count_vocab[token.text] = 1
                        continue
                    self.count_vocab[token.text] += 1

                else:
                    self.adj_verbs[token.text] = False

    def create_embeddings(self, documents: list[str]):
        print("Processing documents...")

        for i, sentence in enumerate(documents):

            tokens = self.bert_tokenizer.tokenize(sentence)
            token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_tensor = torch.tensor([token_ids])

            try:
                with torch.no_grad():
                    self._model.eval()
                    outputs = self._model(input_tensor)
            except Exception as e:
                print(e)
                print(f"Skipping the #{i} comment", f"Total Words: {len(sentence.split(' '))}")
                continue

            word_vectors = outputs.last_hidden_state.squeeze(0)
            for word, vec in zip(tokens, word_vectors):
                vec = np.array(vec)
                if word in self._stored_vectors:
                    if word not in self.duplicates:
                        self.duplicates[word] = [vec]
                    else : self.duplicates[word].append(vec)
                    continue
                self._stored_vectors[word] = vec

        print("Finished. Vocabulary Generated!")

    def prepare_target_sets(self, TargetSet1: list[str], TargetSet2: list[str]):

        ts1embed = np.array(
            [
                self._stored_vectors[word]
                for word in TargetSet1
                if word in self._stored_vectors
            ]
        )
        ts2embed = np.array(
            [
                self._stored_vectors[word]
                for word in TargetSet2
                if word in self._stored_vectors
            ]
        )

        res1 = np.zeros(self.vecdim, dtype=np.float32)
        res2 = np.zeros(self.vecdim, dtype=np.float32)

        for wv1, wv2 in zip(ts1embed, ts2embed):
            res1 = np.add(wv1, res1)
            res2 = np.add(wv2, res2)

        ts1_centroid = res1 / len(TargetSet1)
        ts2_centroid = res2 / len(TargetSet2)

        return (ts1_centroid, ts2_centroid)
