import spacy
import torch
from transformers import BertTokenizer, BertModel
import numpy as np


class BertEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self._model = BertModel.from_pretrained(model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)
        self._stored_vectors = {}
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.adj_verbs = {}
        self.vecdim = 768

    def tokenize_sentences(self, documents: list[str]):
        # Trying to have a ledger which tells out of the total words
        # which of them are adjectives and verbs
        for sentence in documents:
            doc = self.spacy_nlp(sentence)
            for token in doc:
                if token.pos_ == "ADJ" or token.pos_ == "VERB":
                    self.adj_verbs[token.text] = True
                else:
                    self.adj_verbs[token.text] = False

    def create_embeddings(self, documents: list[str]):
        print("Processing documents...")

        for sentence in documents:
            tokens = self.bert_tokenizer.encode(
                sentence,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=512,
            )
            token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
            input_tensor = torch.tensor([token_ids])

            try:
                with torch.no_grad():
                    self._model.eval()
                    outputs = self._model(input_tensor)
            except Exception:
                print(len(sentence))
                raise Exception

            word_vectors = outputs.last_hidden_state.squeeze(0)
            for word, vec in zip(tokens, word_vectors):
                self._stored_vectors[word] = np.array(vec)

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
