import numpy as np

def cosine_similarity(word_vector1, word_vector2):
  """
  Calculates the cosine similarity between two word vectors.
  """

  dot_product = np.dot(word_vector1, word_vector2)
  wv1_mag = np.linalg.norm(word_vector1)
  wv2_mag = np.linalg.norm(word_vector2)
  cosine_similarity = dot_product / (wv1_mag * wv2_mag)

  return cosine_similarity