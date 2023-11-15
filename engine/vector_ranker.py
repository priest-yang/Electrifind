from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
import numpy as np
import csv
import gzip
import json
import pickle
import jsonlines
from tqdm import tqdm

from ranker import Ranker


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
        # Use device='cpu' when doing model instantiation (for AG)
        # If you know what the parameter does, feel free to play around with it
        # TODO: Instantiate the bi-encoder model here
        self.bi_encoder_model_name = bi_encoder_model_name
        self.model = SentenceTransformer(bi_encoder_model_name)
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first
        """
        # NOTE: Do not forget to handle edge cases

        # TODO: Encode the query using the bi-encoder
        embedding = self.model.encode(query)

        # TODO: Score the similarity of the query vector and document vectors for relevance
        # Calculate the dot products between the query embedding and all document embeddings
        scores_list = []
        for doc_embedding, doc_id in zip(self.encoded_docs, self.row_to_docid):
            score = util.pytorch_cos_sim(embedding, doc_embedding).item()
            scores_list.append((doc_id, score))

        # TODO: Generate the ordered list of (document id, score) tuples

        # TODO: Sort the list so most relevant are first
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
        np.save('../cache/' + self.bi_encoder_model_name.replace('/', '_') + '.npy', encoded_docs)
        with open('../cache/encoded_map.pkl', 'wb') as f:
            pickle.dump(encoded_map, f)


if __name__ == '__main__':
    pass
