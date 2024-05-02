import numpy as np
import pandas as pd
from numpy import ndarray
from sentence_transformers import util

from ranker import Ranker

CACHE_PATH = './__cache__/'
DATA_PATH = './data/'


class VectorRanker(Ranker):
    def __init__(self, index, ranker, stations_path=str, users_path=str) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding
            user_profile: A matrix where each row is an vector of user profile
            profile_row_to_userid: A list that is a mapping from the user id to the row number that row corresponds to
                the user profile

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        self.name = 'VectorRanker'
        self.index = index
        self.encode_docs(stations_path, users_path)
        self.ranker = ranker

    def encode_docs(self, docs_path: str, users_path: str):
        encoded_docs = pd.read_csv(docs_path)
        user_profile = pd.read_csv(users_path)
        self.row_to_docid = encoded_docs['docid'].to_list()
        self.profile_row_to_userid = (
            user_profile.index.to_numpy() + 1).tolist()

        # file_path = CACHE_PATH + 'row_to_docid.txt'
        # with open(file_path, 'w') as file:
        #     for element in row_to_docid:
        #         file.write(str(element) + '\n')

        self.encoded_docs = encoded_docs.to_numpy()
        # np.save(CACHE_PATH + 'encoded_stations.npy', encoded_docs)
        user_profile = user_profile.to_numpy()
        self.user_profile = np.concatenate((user_profile, -1 * user_profile), axis=1)
        # np.save(CACHE_PATH + 'encoded_user_profile.npy', encoded_user_profile)

    def personalized_re_rank(self, result: list[int] | list[tuple[int, float]], user_id: int = None) -> list[int] | list[tuple[int, float]]:
        '''
        Re-ranks the results based on the user's profile

        Args:
            result: A list of document ids or result to be re-ranked, mostly, from the previous ranker
            user_id: The user's id

        Returns:
            A list of document ids or result re-ranked based on the user's profile
        '''

        if user_id is None or user_id not in self.profile_row_to_userid:
            return result

        if type(result[0]) is tuple or type(result[0]) is list:
            pre_ranker_result = [res[0] for res in result]
        else:
            pre_ranker_result = result.copy()

        user_vec = self.user_profile[self.profile_row_to_userid.index(user_id)]
        encoded_len = len(self.encoded_docs[0])

        doc_vecs = []
        for docid in pre_ranker_result:
            if docid in self.row_to_docid:
                doc_vecs.append(
                    self.encoded_docs[self.row_to_docid.index(docid)])
            else:
                doc_vecs.append(np.zeros(encoded_len))

        scores = np.dot(doc_vecs, user_vec)
        sorted_idx = np.argsort(scores)[::-1]

        return_list = result.copy()
        for i in range(len(sorted_idx)):
            return_list[i] = [pre_ranker_result[sorted_idx[i]], 0]
        return return_list

    def query(self, query: str, radius: float = 0.03, user_id: int = None, threshold: int = 100) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.
        Performs query expansion using pseudo-relevance feedback if needed.

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: We don't use the user_id parameter in vector ranker. It is here just to align all the
                    Ranker interfaces.

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        query_parts = [float(x) for x in query.split(',')]

        if len(query_parts) == 0:
            return []
        mask = (abs(query_parts[0] - self.index.Latitude) <
                radius) & (abs(query_parts[1] - self.index.Longitude) < radius)
        relevant_docs = self.index[mask]
        if len(relevant_docs) == 0:
            return []

        relevant_docs['score'] = relevant_docs.apply(
            lambda x: self.ranker.scorer.score([x.Latitude, x.Longitude], query_parts), axis=1)
        relevant_docs = relevant_docs.sort_values(
            by=['score'], ascending=False)
        results = relevant_docs[['ID', 'score']].values.tolist()

        # re-rank based on user-id
        results = self.personalized_re_rank(results, user_id)

        return results

    def rank_docs(self, embedding: ndarray) -> list[tuple[int, float]]:
        # If the user has indicated we should use feedback, then update the
        # query vector with respect to the specified number of most-relevant documents

        # Get the most-relevant document vectors for the initial query

        # Compute the average vector of the specified number of most-relevant docs
        # according to how many are to be used for pseudofeedback

        # Combine the original query doc with the feedback doc to use
        # as the new query embedding

        # Score the similarity of the query vec and document vectors for relevance
        scores_list = []
        for row, doc_embedding in enumerate(self.encoded_docs):
            score = util.dot_score(embedding, doc_embedding).item()

        # Generate the ordered list of (document id, score) tuples
            scores_list.append((row, score))

        # Sort the list by relevance score in descending order (most relevant first)
        scores_list.sort(key=lambda x: x[1], reverse=True)

        return scores_list


if __name__ == '__main__':
    vector_ranker = VectorRanker(index=None, ranker=None, stations_path=DATA_PATH + 'station_personalized_features.csv',
                                 users_path=DATA_PATH + 'user_profile.csv')
