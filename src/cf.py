from tqdm import tqdm
import pandas as pd
import numpy as np
from ranker import *


class CFRanker:
    def __init__(self) -> None:
        """
        Initializes a L2RRanker system.

        Args:
            score_index: The index with colums as users, rows as charging stations, and values as scores
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.score_index = []
        self.sim_index = []
        self.row_means = []
        self.id_map = []
        self.get_scores()
        self.get_similarities()

    def get_scores(self):
        dataset_df = pd.read_csv(
            '../data/Google_Map_review_data_AA_DTW.csv', sep=',', header=0)
        sample_df = pd.read_csv(
            '../data/processed_nrel.csv', sep=',', header=0)
        sample_df['index'] = sample_df.index
        authors_list = dataset_df.author_name.unique()
        stations_list = []
        coor_list = set()
        for row in dataset_df.itertuples():
            if (float(row.lat), float(row.lng)) not in coor_list:
                coor_list.add((float(row.lat), float(row.lng)))
                mask = (abs(sample_df.Latitude - row.lat) < 0.001) & (
                    abs(sample_df.Longitude - row.lng) < 0.001)
                masked_nrel = sample_df[mask]
                if len(masked_nrel) > 0:
                    self.id_map.append(masked_nrel.iloc[0]['index'])
                    stations_list.append(row.name)
        self.score_index = pd.DataFrame(
            index=stations_list, columns=authors_list)
        for row in dataset_df.itertuples():
            self.score_index.loc[row.name, row.author_name] = row.rating
        self.score_index.reset_index(drop=True, inplace=True)
        self.row_means = self.score_index.mean(axis=1, skipna=True)
        self.score_index = self.score_index.apply(
            lambda row: row - self.row_means[row.name] if np.isfinite(row.sum()) else row, axis=1)

    def get_similarities(self):
        """
        Computes the similarity matrix for the documents in the index.
        """

        self.sim_index = np.zeros(
            (len(self.score_index), len(self.score_index)))
        for i in tqdm(range(len(self.score_index))):
            for j in range(len(self.score_index)):
                if i == j:
                    self.sim_index[i][j] = 1
                    continue
                self.sim_index[i][j] = self.pearson_correlation(i, j)

    def cosine_similarity(self, doc1, doc2):
        '''
        Computes the cosine similarity between two arrays. 
        '''
        vector_1 = self.score_index.iloc[doc1].values
        vector_2 = self.score_index.iloc[doc2].values
        mask = ~np.isnan(vector_1) & ~np.isnan(vector_2)
        dot_prod = np.dot(vector_1[mask], vector_2[mask])
        return dot_prod

    def pearson_correlation(self, doc1, doc2):
        vector_1 = self.score_index.iloc[doc1].values
        vector_2 = self.score_index.iloc[doc2].values
        mask = ~np.isnan(vector_1) & ~np.isnan(vector_2)
        dot_prod = np.dot(vector_1[mask], vector_2[mask])
        norms = np.linalg.norm(vector_1[mask]) * \
            np.linalg.norm(vector_2[mask])
        if norms == 0:
            return 0
        return dot_prod / norms

    def get_prediction(self, x, i):
        '''
        Computes the prediction for a pair of documents. 
        '''
        sum = 0
        normalizer = 0
        for j in range(len(self.score_index)):
            if np.isnan(self.score_index.iloc[j][x]):
                continue
            sum += self.sim_index[i][j] * self.score_index.iloc[j][x]
            normalizer += self.sim_index[i][j]
        if normalizer == 0:
            return self.row_means[x]
        return sum / normalizer + self.row_means[x]


if __name__ == '__main__':
    pass
