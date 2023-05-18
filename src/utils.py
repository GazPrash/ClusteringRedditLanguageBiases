import numpy as np

def cosine_similarity(word_vector1, word_vector2):
  """
  Calculates the cosine similarity between two word vectors.
  """

  dot_product = np.dot(word_vector1, word_vector2)
  magnitude1 = np.linalg.norm(word_vector1)
  magnitude2 = np.linalg.norm(word_vector2)
  cosine_similarity = dot_product / (magnitude1 * magnitude2)

  return cosine_similarity