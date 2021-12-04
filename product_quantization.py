import numpy as np
from sklearn.cluster import KMeans
import pickle
import itertools

def nptype_from_bits(c):
    if c <= 8:
        return np.uint8
    elif c <= 16:
        return np.uint16
    elif c <= 32:
        return np.uint32
    else:
        return np.uint64


class PQ:
    def __init__(self, n_subvectors, n_bits_per_vector):
        self.n_subvectors = n_subvectors
        self.n_centroids = 2 ** n_bits_per_vector
        self.int_type = np.uint8
        self.random_state = 420
        self.partition_size = -1


        #### TODO: SAVE DICTS TO PICKLE ####
        self.subvector_centroids = {}
        self.kmeans = {} #??
        self.lookup = {}

    @staticmethod
    def calc_dist(vec1: np.ndarray, vec2: np.ndarray):
        return np.sum(np.square(vec2 - vec1), axis=-1)

    def save_index(self):
        pickle.dump([self.subvector_centroids,self.kmeans,self.lookup], open( "index.pickle", "wb"))

    def _get_data_partition(self, train_data, partition_idx):
        partition_start = partition_idx * self.partition_size
        partition_end = (partition_idx + 1) * self.partition_size
        train_data_partition = train_data[:, partition_start:partition_end]
        return train_data_partition

    def _compress_partition(self, partition_idx: int, train_data_partition):
        km = KMeans(n_clusters=self.n_centroids, n_init=1, random_state=self.random_state)
        compressed_data_partition = km.fit_predict(train_data_partition).astype(self.int_type)
        partition_centroids = km.cluster_centers_
        return partition_idx, compressed_data_partition, partition_centroids, km

    def calculate_lookup(self, centroids):
        table = np.zeros(shape=(self.n_centroids, self.n_centroids))
        for v in itertools.combinations(enumerate(centroids),2):
            i = v[0][0]
            j = v[1][0]
            c1 = v[0][1]
            c2 = v[1][1]
            table[i][j] = self.calc_dist(c1,c2)
            table[j][i] = table[i][j]
        return table

    def compress(self, train_data: np.ndarray, to_save: bool = False):

        nb_samples = len(train_data)
        self.compressed_data = np.empty(shape=(nb_samples, self.n_subvectors), dtype=self.int_type)

        d = len(train_data[0])
        self.partition_size = d // self.n_subvectors

        for partition_idx in range(self.n_subvectors):
            train_data_partition = self._get_data_partition(train_data, partition_idx)
            partition_idx, compressed_data_partition, partition_centroids, km = \
                self._compress_partition(partition_idx, train_data_partition)
            self.compressed_data[:, partition_idx] = compressed_data_partition
            self.subvector_centroids[partition_idx] = partition_centroids
            self.kmeans[partition_idx] = km
            self.lookup[partition_idx] = self.calculate_lookup(partition_centroids)

        if to_save: self.save_index()


    def predict_single(self, test_sample: np.ndarray, nearest_neighbors: int, using_lookup=False):

        if using_lookup:
            nb_stored_samples = len(self.compressed_data)
            distances = np.empty(shape=(nb_stored_samples, self.n_subvectors), dtype=np.float64)
            for partition_idx in range(self.n_subvectors):
                partition_start = partition_idx * self.partition_size
                partition_end = (partition_idx + 1) * self.partition_size
                test_sample_partition = test_sample[partition_start:partition_end]
                centroids_partition = self.compressed_data[:, partition_idx]
                compressed_test_partition = self.kmeans[partition_idx].predict(test_sample_partition.reshape(1, -1))
                distances[:, partition_idx] = [self.lookup[partition_idx][compressed_test_partition][0][i] for i in
                                               centroids_partition]

            # print(distances)
            distances = list(map(lambda x: sum(x), distances))
            indices = np.argpartition(distances, nearest_neighbors)
        else:
            distances = np.empty(shape=(self.n_centroids, self.n_subvectors), dtype=np.float64)

            for partition_idx in range(self.n_subvectors):
                partition_start = partition_idx * self.partition_size
                partition_end = (partition_idx + 1) * self.partition_size
                test_sample_partition = test_sample[partition_start:partition_end]
                centroids_partition = self.subvector_centroids[partition_idx]
                # todo replace lookup
                distances[:, partition_idx] = self.calc_dist(test_sample_partition, centroids_partition)

            # Calculate (approximate) distance to stored data
            nb_stored_samples = len(self.compressed_data)
            distance_sums = np.zeros(shape=nb_stored_samples)
            for partition_idx in range(self.n_subvectors):
                distance_sums += distances[:, partition_idx][self.compressed_data[:, partition_idx]]

            indices = np.argpartition(distance_sums, nearest_neighbors)

        return indices[:nearest_neighbors]


    def predict(self, test_data: np.ndarray, nearest_neighbors: int, using_lookup=False):
        preds = [self.predict_single(row, nearest_neighbors, using_lookup) for row in test_data]
        return np.array(preds)
