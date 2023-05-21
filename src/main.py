import warnings
from bias_estimation import BiasEstimation
from embeddings import Embedder

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    datapath = "data/askmen.csv"  # 7000 Comments

    TargetSet1 = ["sister", "female", "woman", "girl", "daughter", "she", "hers", "her"]
    TargetSet2 = ["brother", "male", "man", "boy", "son", "he", "his", "him"]

    global_embedding = Embedder()
    global_embedding.reinitialize()
    bestimation = BiasEstimation(
        TargetSet1, TargetSet2, global_embedding, datapath, focus_column="Comment"
    )

    cl1, cl2 = bestimation.NBiasedWords(n=500)
