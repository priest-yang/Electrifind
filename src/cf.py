from tqdm import tqdm
import pandas as pd
import numpy as np
from ranker import *


class CFRanker:
    def __init__(self, index, ranker: Ranker) -> None:
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
        self.index = index
        self.score_index = []
        self.sim_index = []
        self.row_means = []
        self.id_map = []
        self.ranker = ranker
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

    def predict(self, X_pred, user_id):
        scores = []
        for x in X_pred:
            scores.append(self.get_prediction(x, user_id))

    def query(self, query: str, user_id: int):
        query_parts = [float(x) for x in query.split(',')]
        if len(query_parts) == 0:
            return []
        mask = (abs(query_parts[0] - self.index.Latitude) <
                0.01) & (abs(query_parts[1] - self.index.Longitude) < 0.01)
        relevant_docs = self.index[mask]
        if len(relevant_docs) == 0:
            return []

        relevant_docs['score'] = relevant_docs.apply(
            lambda x: self.ranker.scorer.score(x, query_parts), axis=1)
        relevant_docs = relevant_docs.sort_values(
            by=['score'], ascending=False)
        relevant_docs['id'] = relevant_docs.index
        results = relevant_docs[['id', 'score']].values.tolist()

        results_top_100 = results[:100]
        results_tails = results[100:]

        X_pred = []
        for item in results_top_100:
            docid = int(item[0])
            X_pred.append(self.id_map.index(docid))

        scores = self.predict(X_pred, user_id)

        # TODO: Sort posting_lists based on scores
        for i in range(len(results_top_100)):
            results_top_100[i] = (results_top_100[i][0], scores[i])
        results_top_100.sort(key=lambda x: x[1], reverse=True)

        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        results = results_top_100 + results_tails

        # TODO: Return the ranked documents
        return results

if __name__ == '__main__':
    pass
