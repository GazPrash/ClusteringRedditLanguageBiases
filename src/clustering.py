import numpy as np
from sklearn.cluster import KMeans
from scipy import spatial
from utils import cosine_similarity


class WordCluster:
    def __init__(
        self,
        cluster_method: str,
        word_embeddings: dict,
        word_biases: list[tuple[str, float]],
        top_male_biased_words: list[str],
        top_female_biased_words: list[str],
    ) -> None:
        self.method = cluster_method
        self.word_embeddings = word_embeddings
        self.word_biases = word_biases
        self.malew = top_male_biased_words
        self.femalew = top_female_biased_words

        # PRESETS 
        self.r = 0.15
        self.repats = 100
        self.verbose = True

    def partition_cluster(self):
        k = int(self.r * (len(self.femalew) + len(self.malew)) / 2)
        emb1, emb2 = [self.word_embeddings[word1] for word1, _ in self.femalew], [
            self.word_embeddings[word2] for word2, _ in self.malew
        ]
        mis1, mis2 = [0,[]], [0,[]]	#here we will save partitions with max sim for both target sets
        for _ in range(self.repats):
            p1 = self.create_subclusters(emb1, self.femalew, k)
            if p1[0] > mis1[0]:
                mis1 = p1
            p2 = self.create_subclusters(emb2, self.malew, k)
            if p2[0] > mis2[0]:
                mis2 = p2
            if self.verbose == True:
                print("New partition for ts1, intrasim: ", p1[0])
                print("New partition for ts2, intrasim: ", p2[0])

        return [mis1[1], mis2[1]]

    def create_subclusters(self, embeddings, biasw, k):
        preds = KMeans(n_clusters=k).fit_predict(embeddings)
        # first create the proper clusters, then estiamte avg intra sim
        all_clusters = []
        for i in range(0, k):
            clust = []
            indexes = np.where(preds == i)[0]
            for idx in indexes:
                clust.append(biasw[idx])
            all_clusters.append(clust)
        score = self.getIntraSim(all_clusters)
        return [score, all_clusters]    
    
    def getIntraSim(self, partition):
        iS = 0
        for cluster in partition:
            iS += self.getIntraSimCluster(cluster)
        return iS / len(partition)

    def getIntraSimCluster(self, cluster):
        # Modification 
        if len(cluster) <= 1:
            return 0
        
        sim = 0
        c = 0
        for i in range(len(cluster)):
            w1 = self.word_embeddings[cluster[i][0]]
            for j in range(i + 1, len(cluster)):
                w2 = self.word_embeddings[cluster[j][0]]
                sim += 1 - cosine_similarity(w1, w2)
                c += 1
        
        return sim / c

