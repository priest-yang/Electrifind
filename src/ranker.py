import math
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from geopy import distance


class Ranker:
    def __init__(self, index, scorer) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.scorer = scorer

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for
            pseudofeedback_num_docs: If pseudo-feedback is requested, the number
                 of top-ranked documents to be used in the query,
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query

        Returns:
            A list containing tuples of the documents (ids) and their relevance scores

        NOTE: We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """

        query_parts = [float(x) for x in query.split(',')]
        if len(query_parts) == 0:
            return []

        results = self.rank_docs(query_parts)

        return results

    def rank_docs(self, query_parts: list[float]) -> list[tuple[int, float]]:
        mask = (abs(query_parts[0] - self.index.Latitude) <
                0.01) & (abs(query_parts[1] - self.index.Longitude) < 0.01)
        relevant_docs = self.index[mask]
        if len(relevant_docs) == 0:
            return []

        relevant_docs['score'] = relevant_docs.apply(
            lambda x: self.scorer.score(x, query_parts), axis=1)
        relevant_docs = relevant_docs.sort_values(
            by=['score'], ascending=False)
        relevant_docs['id'] = relevant_docs.index
        results = relevant_docs[['id', 'score']].values.tolist()
        return results


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """

    def __init__(self, index, parameters=None) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, doc, query_parts) -> float:
        raise NotImplementedError


class DistScorer(RelevanceScorer):
    def __init__(self, index, parameters=None) -> None:
        super().__init__(index, parameters)

    def score(self, doc, query_parts) -> float:
        return 1 / (1 + distance.distance((query_parts[0], query_parts[1]), (doc.Latitude, doc.Longitude)).km)


if __name__ == '__main__':
    pass
