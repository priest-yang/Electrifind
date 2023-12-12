from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np
import csv
import gzip
import jsonlines
import pickle
from tqdm import tqdm
from collections import Counter


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here

        self.bi_encoder_model_name = bi_encoder_model_name
        self.model = SentenceTransformer(bi_encoder_model_name)
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid

    def query(self, query: str, pseudofeedback_num_docs=0,
              pseudofeedback_alpha=0.8, pseudofeedback_beta=0.2, user_id=None) -> list[tuple[int, float]]:
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
        # TODO: Encode the query using the bi-encoder
        embedding = self.model.encode(query)
        scores_list = self.rank_docs(embedding)

        if pseudofeedback_num_docs > 0:
            pseudo_docs_embedding = pseudofeedback_alpha * embedding
            for row, score in scores_list[:pseudofeedback_num_docs]:
                pseudo_doc = self.encoded_docs[row]
                pseudo_docs_embedding += pseudofeedback_beta * \
                    pseudo_doc / pseudofeedback_num_docs
            scores_list = self.rank_docs(pseudo_docs_embedding)

        return [(self.row_to_docid[row], score) for row, score in scores_list]

    def rank_docs(self, embedding: ndarray) -> list[tuple[int, float]]:
        # TODO: If the user has indicated we should use feedback, then update the
        #  query vector with respect to the specified number of most-relevant documents

        # TODO: Get the most-relevant document vectors for the initial query

        # TODO: Compute the average vector of the specified number of most-relevant docs
        #  according to how many are to be used for pseudofeedback

        # TODO: Combine the original query doc with the feedback doc to use
        #  as the new query embedding

        # TODO: Score the similarity of the query vec and document vectors for relevance
        scores_list = []
        for row, doc_embedding in enumerate(self.encoded_docs):
            score = util.dot_score(embedding, doc_embedding).item()

        # TODO: Generate the ordered list of (document id, score) tuples
            scores_list.append((row, score))

        # TODO: Sort the list by relevance score in descending order (most relevant first)
        scores_list.sort(key=lambda x: x[1], reverse=True)

        return scores_list

    def encode_docs(self, dataset_path: str):
        dev_docs = []
        with open('../data/hw3_relevance.dev.csv', 'r') as file:
            data = csv.reader(file)
            for idx, row in tqdm(enumerate(data)):
                if idx == 0:
                    continue
                dev_docs.append(row[2])
        encoded_docs = []
        encoded_map = []
        dataset_file = gzip.open(dataset_path, 'rt')
        with jsonlines.Reader(dataset_file) as reader:
            for _ in tqdm(range(200000)):
                try:
                    document = reader.read()
                    if str(document['docid']) in dev_docs:
                        embedded_doc = self.model.encode(document['text'])
                        encoded_docs.append(embedded_doc)
                        encoded_map.append(document['docid'])
                except EOFError:
                    break
        encoded_docs = np.stack(encoded_docs)
        np.save('../cache/' + self.bi_encoder_model_name.replace('/',
                '_') + '.npy', encoded_docs)
        with open('../cache/encoded_map.pkl', 'wb') as f:
            pickle.dump(encoded_map, f)

    # TODO (HW5): Find the dot product (unnormalized cosine similarity) for the list of documents (pairwise)
    # NOTE: You should return a matrix where element [i][j] would represent similarity between
    #   list_docs[i] and list_docs[j]
    def document_similarity(self, list_docs: list[int]) -> np.ndarray:
        """
        Calculates the pairwise similarities for a given list of documents

        Args:
            list_docs: A list of document IDs

        Returns:
            A matrix where element [i][j] is a similarity score between list_docs[i] and list_docs[j]
        """
        sim_mat = np.zeros((len(list_docs), len(list_docs)))
        for i in range(len(list_docs)):
            for j in range(i, len(list_docs)):
                sim_mat[i][j] = util.cos_sim(
                    self.encoded_docs[self.row_to_docid.index(list_docs[i])],
                    self.encoded_docs[self.row_to_docid.index(list_docs[j])])
                sim_mat[j][i] = sim_mat[i][j]
        return sim_mat    


if __name__ == '__main__':
    pass
