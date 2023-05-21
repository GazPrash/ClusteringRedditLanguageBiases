import gensim.downloader as api
import nltk
import numpy as np
import spacy
import torch
from transformers import BertTokenizer, BertModel


class Embedder:
    def __init__(
        self, model_type="Glove", model_name: str = "glove-wiki-gigaword-200"
    ) -> None:
        self.model_type = model_type
        self.model_name = model_name
        self.bert_tokenizer = None

        if self.model_type == "Glove":
            self.load_nltk_res()
            self._model = api.load(model_name)

        elif self.model_type == "Bert":
            # Here the favourable embedding model is : bert-base-uncased
            self.spacy_nlp = spacy.load("en_core_web_sm")
            self._model = BertModel.from_pretrained(model_name)
            self.bert_tokenizer = BertTokenizer.from_pretrained(model_name)

        else:
            raise Exception("Incorrect embedding model chosen!")

        self._stored_vectors = {}  # This will store all our vocabulary
        self.no_embeds = []  # Tells us how many words cant get an embedding
        self.no_embed_count = 0

    def reinitialize(self):

        self.no_embed_count = 0
        self.no_embeds = []

    def load_nltk_res(self):
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")

    # # Use only when working with Bert model, in other cases use the 'find_adjectives' method
    # def tokenize_sentence(self, sentence: str):
    #     # Here we are trying to use verbs and adjectives for checking biases
    #     doc = self.spacy_nlp(sentence)
    #     adj_verbs = []
    #     for token in doc:
    #         if token.pos_ == "ADJ" or token.pos_ == "VERB":
    #             adj_verbs.append(token.text)

    #     return adj_verbs

    # def get_bert_embedding(self, word: str):
    #     if self.model_type != "Bert":
    #         raise Exception(
    #             "Use 'get_bert_embedding' or any other suitable method for your model."
    #         )

    #     tokens = self.bert_tokenizer(word)
    #     token_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
    #     input_tensor = torch.tensor([token_ids])

    #     with torch.no_grad():
    #         self._model.eval()
    #         outputs = self._model(input_tensor)
    #     word_vector = outputs.last_hidden_state.squeeze(0)\
    #                                            .squeeze(0)\
    #                                            .numpy()
        
    #     self._stored_vectors[word] = word_vector
    #     return word_vector

    def get_glove_embedding(self, word: str):
        if self.model_type != "Glove":
            raise Exception(
                "Use 'get_bert_embedding' or any other suitable method for your model."
            )

        if word in self._stored_vectors:
            return self._stored_vectors[word]

        embedding = np.zeros(200)
        if word in self._model.index_to_key:
            embedding = self._model[word]
            self._stored_vectors[word] = embedding
            return embedding

        self.no_embed_count += 1
        return None

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

    def prepare_target_sets(self, TargetSet1: list[str], TargetSet2: list[str]):
        embd_method = self.get_glove_embedding if self.model_type == "Glove" else self.get_bert_embedding
        vecdim = 200 if self.model_type == "Glove" else 768
        
        ts1embed = np.array([embd_method(x) for x in TargetSet1])
        ts2embed = np.array([embd_method(x) for x in TargetSet2])

        res1 = np.zeros(vecdim, dtype=np.float32)
        res2 = np.zeros(vecdim, dtype=np.float32)

        for wv1, wv2 in zip(ts1embed, ts2embed):
            res1 = np.add(wv1, res1)
            res2 = np.add(wv2, res2)

        ts1_centroid = res1 / len(TargetSet1)
        ts2_centroid = res2 / len(TargetSet2)

        return (ts1_centroid, ts2_centroid)
