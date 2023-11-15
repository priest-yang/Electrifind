"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
import numpy as np
from collections import Counter, defaultdict
from sentence_transformers import CrossEncoder
from indexing import InvertedIndex
import math
from tqdm import tqdm


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: Implement this class properly; this is responsible for returning a list of sorted relevant documents
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str], scorer: 'RelevanceScorer') -> None:
        """
        Initializes the state of the Ranker object.

        TODO (HW3): Previous homeworks had you passing the class of the scorer to this function
        This has been changed as it created a lot of confusion.
        You should now pass an instantiated RelevanceScorer to this function.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A list of dictionary objects with keys "docid" and "score" where docid is
            a particular document in the collection and score is that document's relevance

        TODO (HW3): We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.
        """
        # TODO: Tokenize the query and remove stopwords, if needed
        results = []
        query_parts = self.tokenize(query)

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input.
        if len(query_parts) == 0:
            return results
        relevant_docs = set()
        if self.scorer.__class__.__name__ == 'CrossEncoderScorer':
            relevant_docs.update(self.scorer.raw_text_dict.keys())
        else:
            for item in query_parts:
                if item in self.index.index:
                    relevant_docs.update([x[0] for x in self.index.index[item]])

        # TODO: Rank the documents using a RelevanceScorer (like BM25 from below classes) 
            doc_term_counts = self.accumulate_doc_term_counts(self.index, query_parts)

        for docid in relevant_docs:
            if self.scorer.__class__.__name__ == 'CrossEncoderScorer':
                score = self.scorer.score(docid, query)
            else:
                score = self.scorer.score(docid, doc_term_counts[docid], query_parts)
            results.append((docid, score))

        # TODO: Return the **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
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

        relevant_docs = set()
        for word in query_parts:
            if word in index.index:
                relevant_docs.update([x[0] for x in index.index[word]])

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
    # TODO (HW1): Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM,
    #             BM25, PivotedNormalization, TF_IDF) and not in this one
    def __init__(self, index: InvertedIndex, parameters) -> None:
        raise NotImplementedError

    # NOTE (hw2): Note the change here: `score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float`
    #             See more in README.md.
    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies
            query_parts: A list of all the words in the query
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)
        """
        raise NotImplementedError


# TODO (HW1): Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        score = 0
        query_index = Counter(query_parts)

        for item in query_index:
            if item not in doc_word_counts:
                continue
            tf_d = doc_word_counts[item]
            tf_q = query_index[item]
            score += tf_d * tf_q

        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'mu': 2000}) -> None:
        self.index = index
        self.mu = parameters['mu']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        score = 0
        query_index = Counter(query_parts)
        if len(query_index) == 0:
            return 0
        total_token_count = self.index.get_statistics()['total_token_count']

        for item in query_index:
            if item not in doc_word_counts:
                continue
            tf_d = doc_word_counts[item]
            term_freq = self.index.get_term_metadata(item)['count']
            log_term = math.log(
                1 + (tf_d / (self.mu * term_freq / total_token_count)))
            score += query_index[item] * log_term
        avr_dl = self.index.get_statistics()['mean_document_length']
        score += len(query_parts) * math.log(self.mu / (self.mu + avr_dl))
        
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index
 
        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        score = 0
        query_index = Counter(query_parts)
        num_docs = self.index.get_statistics()['number_of_documents']
        avr_dl = self.index.get_statistics()['mean_document_length']
        doc_length = self.index.get_doc_metadata(docid)['length']

        for item in query_index:
            if item not in doc_word_counts:
                continue
            term_freq = self.index.get_term_metadata(item)['num_docs']
            idf = math.log((num_docs - term_freq + 0.5) / (term_freq + 0.5))
            try:
                tf_norm = ((self.k1 + 1) * doc_word_counts[item]) / (self.k1 * (
                    1 - self.b + self.b * doc_length / avr_dl) + doc_word_counts[item])
            except:
                tf_norm = 0
            tf_q_norm = ((self.k3 + 1) *
                         query_index[item]) / (self.k3 + query_index[item])
            score += idf * tf_norm * tf_q_norm

        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score
            # hint: 
            ## term_frq will always be >0
            ## doc_frq will always be >0 since only looking at terms that are in both query and doc

        # 4. Return the score
        score = 0
        query_index = Counter(query_parts)
        num_docs = self.index.get_statistics()['number_of_documents']
        avr_dl = self.index.get_statistics()['mean_document_length']
        doc_length = self.index.get_doc_metadata(docid)['length']
        for item in query_index:
            if item not in doc_word_counts:
                continue
            tf_norm = (1 + math.log(1 + math.log(doc_word_counts[item]))) / (
                1 - self.b + self.b * doc_length / avr_dl)
            term_freq = self.index.get_term_metadata(item)['num_docs']
            idf = math.log((num_docs + 1) / term_freq)
            score += query_index[item] * tf_norm * idf

        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW1): Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters={}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]):
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        score = 0
        query_index = Counter(query_parts)
        num_docs = self.index.get_statistics()['number_of_documents']

        for item in query_index:
            if item not in doc_word_counts:
                continue
            tf_d = tf_d = math.log(1 + doc_word_counts[item])
            tf_q = query_index[item]
            term_freq = self.index.get_term_metadata(item)['num_docs']
            idf_q = 1 + math.log(num_docs / term_freq)
            score += tf_d * tf_q * idf_q
        
        return score  # (`score` should be defined in your code; you can call it whatever you want)


# TODO (HW3): The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
class CrossEncoderScorer:
    def __init__(self, raw_text_dict: dict[int, str], cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
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
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if docid not in self.raw_text_dict:
            return 0

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        return self.model.predict([(query, self.raw_text_dict[docid])])[0]
    

# TODO (HW1): Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    pass


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10
        """
        # Print randomly ranked results
        return 10


if __name__ == '__main__':
    pass