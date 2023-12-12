import math
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from geopy import distance

from sentence_transformers import CrossEncoder
from indexing import InvertedIndex

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: Return a list of sorted relevant documents.

    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str]) -> None:
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
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str, pseudofeedback_num_docs=0, pseudofeedback_alpha=0.8,
              pseudofeedback_beta=0.2) -> list[tuple[int, float]]:
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
        # TODO: Tokenize the query and remove stopwords, if needed
        query_parts = self.tokenize(query)
        if len(query_parts) == 0:
            return []
        query_word_count = Counter(query_parts)

        results = self.rank_docs(query_parts, query, query_word_count)

        # TODO: If the user has indicated we should use feedback,
        #  create the pseudo-document from the specified number of pseudo-relevant results.
        #  This document is the cumulative count of how many times all non-filtered words show up
        #  in the pseudo-relevant documents. See the equation in the write-up. Be sure to apply the same
        #  token filtering and normalization here to the pseudo-relevant documents.
        if pseudofeedback_num_docs > 0:
            pseudo_docs_word_count = Counter()
            for docid, score in results[:pseudofeedback_num_docs]:
                if docid not in self.raw_text_dict:
                    continue
                pseudo_doc = self.tokenize(self.raw_text_dict[docid])
                pseudo_docs_word_count.update(pseudo_doc)

        # TODO: Combine the document word count for the pseudo-feedback with the query to create a new query
        # NOTE: When using alpha and beta to weight the query and pseudofeedback doc, the counts
        #  will likely be *fractional* counts (not integers).
            new_query_word_count = Counter()
            for word, count in query_word_count.items():
                new_query_word_count[word] += pseudofeedback_alpha * count
            for word, count in pseudo_docs_word_count.items():
                new_query_word_count[word] += pseudofeedback_beta * \
                    count / pseudofeedback_num_docs
            new_query_parts = list(new_query_word_count.keys())
            results = self.rank_docs(
                new_query_parts, None, new_query_word_count)

        return results

    def rank_docs(self, query_parts: list[str], query: str,
                  query_word_count: dict[str, int]) -> list[tuple[int, float]]:
        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #  a document ID to a dictionary of the counts of the query terms in that document.
        #  You will pass the dictionary to the RelevanceScorer as input.
        relevant_docs = set()
        if self.scorer.__class__.__name__ == 'CrossEncoderScorer':
            relevant_docs.update(self.scorer.raw_text_dict.keys())
        else:
            for item in query_word_count.keys():
                if item in self.index.index:
                    relevant_docs.update([x[0]
                                         for x in self.index.index[item]])
            doc_term_counts = self.accumulate_doc_term_counts(
                self.index, query_parts, relevant_docs)

        # TODO: Rank the documents using a RelevanceScorer
        results = []
        if self.scorer.__class__.__name__ == 'CrossEncoderScorer':
            for docid in relevant_docs:
                score = self.scorer.score(docid, query)
                results.append((docid, score))
        else:
            for docid in relevant_docs:
                score = self.scorer.score(
                    docid, doc_term_counts[docid], query_word_count)
                results.append((docid, score))

        # TODO: Return the **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str], relevant_docs: set[int]
                                   ) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        doc_term_count = defaultdict(Counter)

        for word in query_parts:
            if word in index.index:
                for index_doc in index.index[word]:
                    if index_doc[0] in relevant_docs:
                        doc_term_count[index_doc[0]][word] = index_doc[1]

        return doc_term_count

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


# TODO: Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        score = 0
        for item in query_word_counts:
            if item not in doc_word_counts:
                continue
            tf_d = doc_word_counts[item]
            tf_q = query_word_counts[item]
            score += tf_d * tf_q
        return score


# TODO: Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.mu = parameters['mu']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        score = 0
        if len(query_word_counts) == 0:
            return 0
        total_token_count = self.index.get_statistics()['total_token_count']
        for item in query_word_counts:
            if item not in doc_word_counts:
                continue
            tf_d = doc_word_counts[item]
            term_freq = self.index.get_term_metadata(item)['count']
            log_term = math.log(
                1 + (tf_d / (self.mu * term_freq / total_token_count)))
            score += query_word_counts[item] * log_term
        avr_dl = self.index.get_statistics()['mean_document_length']
        query_length = sum(query_word_counts.values())
        score += query_length * math.log(self.mu / (self.mu + avr_dl))
        return score


# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        score = 0
        num_docs = self.index.get_statistics()['number_of_documents']
        avr_dl = self.index.get_statistics()['mean_document_length']
        doc_length = self.index.get_doc_metadata(docid)['length']
        for item in query_word_counts:
            if item not in doc_word_counts:
                continue
            term_freq = self.index.get_term_metadata(item)['num_docs']
            idf = math.log((num_docs - term_freq + 0.5) / (term_freq + 0.5))
            tf_norm = ((self.k1 + 1) * doc_word_counts[item]) / (self.k1 * (
                1 - self.b + self.b * doc_length / avr_dl) + doc_word_counts[item])
            tf_q_norm = ((self.k3 + 1) *
                         query_word_counts[item]) / (self.k3 + query_word_counts[item])
            score += idf * tf_norm * tf_q_norm
        return score


# TODO: Implement Personalized BM25
class PersonalizedBM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, relevant_doc_index: InvertedIndex,
                 parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        """
        Initializes Personalized BM25 scorer.

        Args:
            index: The inverted index used to use for computing most of BM25
            relevant_doc_index: The inverted index of only documents a user has rated as relevant,
                which is used when calculating the personalized part of BM25
            parameters: The dictionary containing the parameter values for BM25

        Returns:
            The Personalized BM25 score
        """
        self.index = index
        self.relevant_doc_index = relevant_doc_index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # TODO: Implement Personalized BM25
        score = 0
        num_docs = self.index.get_statistics()['number_of_documents']
        num_docs_rel = self.relevant_doc_index.get_statistics()[
            'number_of_documents']
        avr_dl = self.index.get_statistics()['mean_document_length']
        doc_length = self.index.get_doc_metadata(docid)['length']
        for item in query_word_counts:
            if item not in doc_word_counts:
                continue
            term_freq = self.index.get_term_metadata(item)['num_docs']
            term_freq_rel = self.relevant_doc_index.get_term_metadata(item)[
                'num_docs']
            idf = math.log(((term_freq_rel + 0.5) * (num_docs - term_freq - num_docs_rel + term_freq_rel + 0.5)
                            ) / ((term_freq - term_freq_rel + 0.5) * (num_docs_rel - term_freq_rel + 0.5)))
            tf_norm = ((self.k1 + 1) * doc_word_counts[item]) / (self.k1 * (
                1 - self.b + self.b * doc_length / avr_dl) + doc_word_counts[item])
            tf_q_norm = ((self.k3 + 1) *
                         query_word_counts[item]) / (self.k3 + query_word_counts[item])
            score += idf * tf_norm * tf_q_norm
        return score


# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        score = 0
        num_docs = self.index.get_statistics()['number_of_documents']
        avr_dl = self.index.get_statistics()['mean_document_length']
        doc_length = self.index.get_doc_metadata(docid)['length']
        for item in query_word_counts:
            if item not in doc_word_counts:
                continue
            tf_norm = (1 + math.log(1 + math.log(doc_word_counts[item]))) / (
                1 - self.b + self.b * doc_length / avr_dl)
            term_freq = self.index.get_term_metadata(item)['num_docs']
            idf = math.log((num_docs + 1) / term_freq)
            score += query_word_counts[item] * tf_norm * idf
        return score


# TODO: Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        score = 0
        num_docs = self.index.get_statistics()['number_of_documents']
        for item in query_word_counts:
            if item not in doc_word_counts:
                continue
            tf_d = tf_d = math.log(1 + doc_word_counts[item])
            tf_q = query_word_counts[item]
            term_freq = self.index.get_term_metadata(item)['num_docs']
            idf_q = 1 + math.log(num_docs / term_freq)
            score += tf_d * tf_q * idf_q
        return score


class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str],
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model

        NOTE 1: The CrossEncoderScorer class uses a pre-trained cross-encoder model
            from the Sentence Transformers package to score a given query-document pair.

        NOTE 2: This is not a RelevanceScorer object because the method signature for score() does not match,
            but it has the same intent, in practice.
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.raw_text_dict = raw_text_dict
        self.model = CrossEncoder(cross_encoder_model_name)

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle edge cases
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if docid not in self.raw_text_dict:
            return 0

        # TODO: Get a score from the cross-encoder model
        #  Refer to IR_Encoder_Examples.ipynb in Demos folder if needed
        return self.model.predict([(query, self.raw_text_dict[docid])])[0]


if __name__ == '__main__':
    pass
