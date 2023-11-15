from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer
import os
import pickle
import math
import csv


# TODO: scorer has been replaced with ranker in initialization, check README for more details
class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()
        self.trained = False

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores: A dictionary of queries mapped to a list of
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            X (list): A list of feature vectors for each query-document pair
            y (list): A list of relevance scores for each query-document pair
            qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y
        #       This is for LightGBM to know how many relevance scores we have per query
        X = []
        y = []
        qgroups = []

        # TODO: For each query and the documents that have been rated for relevance to that query,
        #       process these query-document pairs into features
        for query, doc_scores in tqdm(query_to_document_relevance_scores.items()):

            # TODO: Accumulate the token counts for each document's title and content here
            query_parts = self.document_preprocessor.tokenize(query)
            doc_term_counts = self.accumulate_doc_term_counts(
                self.document_index, query_parts)
            title_term_counts = self.accumulate_doc_term_counts(
                self.title_index, query_parts)

            # TODO: For each of the documents, generate its features, then append
            #       the features and relevance score to the lists to be returned
            for doc in doc_scores:
                docid = doc[0]
                score = doc[1]
                features = self.feature_extractor.generate_features(
                    docid, doc_term_counts[docid], title_term_counts[docid], query_parts, query)
                X.append(features)
                y.append(score)

            # Make sure to keep track of how many scores we have for this query
            qgroups.append(len(doc_scores))

        return X, y, qgroups

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
        #       create a dictionary that keeps track of their counts for the query word
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

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        print("Loading training data...")
        if os.path.exists('../cache/X.pkl'):
            X = pickle.load(open('../cache/X.pkl', 'rb'))
            y = pickle.load(open('../cache/y.pkl', 'rb'))
            qgroups = pickle.load(open('../cache/qgroups.pkl', 'rb'))
        else:
            query_to_doc_rel_scores = {}
            if training_data_filename.endswith('.csv'):
                with open(training_data_filename, 'r') as f:
                    reader = csv.reader(f)
                    reader.__next__()
                    if training_data_filename.endswith('data-relevance.csv'):
                        for row in tqdm(reader):
                            query = row[0]
                            docid = int(row[1])
                            rel_score = row[2]
                            if query not in query_to_doc_rel_scores:
                                query_to_doc_rel_scores[query] = [
                                    (docid, rel_score)]
                            else:
                                query_to_doc_rel_scores[query].append(
                                    (docid, rel_score))
                    elif training_data_filename.endswith('.train.csv'):
                        for row in tqdm(reader):
                            query = row[0]
                            docid = int(row[2])
                            rel_score = row[4]
                            if query not in query_to_doc_rel_scores:
                                query_to_doc_rel_scores[query] = [
                                    (docid, rel_score)]
                            else:
                                query_to_doc_rel_scores[query].append(
                                    (docid, rel_score))
            elif training_data_filename.endswith('.jsonl'):
                query_to_doc_rel_scores = pd.read_json(
                    training_data_filename, lines=True)
            else:
                raise ValueError("Unsupported file format.")

        # TODO: Prepare the training data by featurizing the query-doc pairs and
        #       getting the necessary datastructures
            print("Preparing training data...")
            X, y, qgroups = self.prepare_training_data(query_to_doc_rel_scores)
            pickle.dump(X, open('../cache/X.pkl', 'wb'))
            pickle.dump(y, open('../cache/y.pkl', 'wb'))
            pickle.dump(qgroups, open('../cache/qgroups.pkl', 'wb'))

        # TODO: Train the model
        print("Training model...")
        self.model.fit(X, y, qgroups)
        self.trained = True

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        # TODO: Return a prediction made using the LambdaMART model
        if self.trained == False:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        results = []
        query_parts = self.document_preprocessor.tokenize(query)
        if len(query_parts) == 0:
            return results
        relevant_docs = set()
        for word in query_parts:
            if word in self.document_index.index:
                relevant_docs.update(
                    [x[0] for x in self.document_index.index[word]])

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        #       a document ID to a dictionary of the counts of the query terms in that document.
        #       You will pass the dictionary to the RelevanceScorer as input
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        #       pass these doc-term-counts to functions later, so we need the accumulated representations

        # TODO: Accumulate the documents word frequencies for the title and the main body
        doc_term_counts = self.accumulate_doc_term_counts(
            self.document_index, query_parts)
        title_term_counts = self.accumulate_doc_term_counts(
            self.title_index, query_parts)

        # TODO: Score and sort the documents by the provided scorer for just the document's main text (not the title).
        #       This ordering determines which documents we will try to *re-rank* using our L2R model
        # for docid in relevant_docs:
        #     score = self.feature_extractor.get_BM25_score(
        #         docid, doc_term_counts[docid], query_parts)
        #     score = self.ranker.scorer.score(docid, doc_term_counts[docid], query_parts)
        #     results.append({'docid': docid, 'score': score})
        # results.sort(key=lambda x: x['score'], reverse=True)
        results = self.ranker.query(query)

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking
        results_top_100 = results[:100]
        results_tails = results[100:]

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        X_pred = []
        for item in results_top_100:
            docid = item[0]
            X_pred.append(self.feature_extractor.generate_features(
                docid, doc_term_counts[docid], title_term_counts[docid], query_parts, query))

        # TODO: Use your L2R model to rank these top 100 documents
        scores = self.predict(X_pred)

        # TODO: Sort posting_lists based on scores
        for i in range(len(results_top_100)):
            results_top_100[i] = (results_top_100[i][0], scores[i])
        results_top_100.sort(key=lambda x: x[1], reverse=True)

        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        results = results_top_100 + results_tails

        # TODO: Return the ranked documents
        return results

    def save_model(self) -> None:
        pickle.dump(self.model.ranker, open('../cache/l2r_model_' + self.ranker.__class__.__name__ + '.pkl', 'wb'))

    def load_model(self) -> None:
        self.model.ranker = pickle.load(open('../cache/l2r_model_' + self.ranker.__class__.__name__ + '.pkl', 'rb'))


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = recognized_categories
        self.docid_to_network_features = docid_to_network_features

        # TODO: For the recognized categories (i.e,. those that are going to be features), consider
        #       how you want to store them here for faster featurizing

        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring
        self.tf_idf_scorer = TF_IDF(self.document_index)
        self.bm25_scorer = BM25(self.document_index)
        self.pivoted_norm_scorer = PivotedNormalization(self.document_index)
        self.ce_scorer = ce_scorer

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.document_index.get_doc_metadata(docid)['length']

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['length']

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        score = 0
        query_index = Counter(query_parts)

        for item in query_index:
            if item not in word_counts:
                continue
            score += math.log(word_counts[item] + 1)

        return score

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        score = self.tf_idf_scorer.score(docid, word_counts, query_parts)
        return score

    # TODO: BM25
    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        score = self.bm25_scorer.score(docid, doc_word_counts, query_parts)
        return score

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        score = self.pivoted_norm_scorer.score(
            docid, doc_word_counts, query_parts)
        return score

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has
        """
        cat_list = [0] * len(self.recognized_categories)
        if docid in self.doc_category_info:
            for id, category in enumerate(self.recognized_categories):
                if category in self.doc_category_info[docid]:
                    cat_list[id] = 1
        return cat_list

    # TODO: PageRank
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        score = 0
        if docid in self.docid_to_network_features:
            score = self.docid_to_network_features[docid]['pagerank']
        return score

    # TODO: HITS Hub
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        score = 0
        if docid in self.docid_to_network_features:
            score = self.docid_to_network_features[docid]['hub_score']
        return score

    # TODO: HITS Authority
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        score = 0
        if docid in self.docid_to_network_features:
            score = self.docid_to_network_features[docid]['authority_score']
        return score

    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """
        return self.ce_scorer.score(docid, query)

    # TODO: Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str],
                          query: str) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """
        # NOTE: We can use this to get a stable ordering of features based on consistent insertion
        #       but it's probably faster to use a list to start

        feature_vector = []

        # TODO: Document Length
        feature_vector.append(self.get_article_length(docid))

        # TODO: Title Length
        feature_vector.append(self.get_title_length(docid))

        # TODO: Query Length
        feature_vector.append(len(query_parts))

        # TODO: TF (document)
        feature_vector.append(self.get_tf(
            self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(
            self.document_index, docid, doc_word_counts, query_parts))

        # TODO: TF (title)
        feature_vector.append(self.get_tf(
            self.title_index, docid, title_word_counts, query_parts))

        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(
            self.title_index, docid, title_word_counts, query_parts))

        # TODO: BM25
        feature_vector.append(self.get_BM25_score(
            docid, doc_word_counts, query_parts))

        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(
            docid, doc_word_counts, query_parts))

        # TODO: PageRank
        feature_vector.append(self.get_pagerank_score(docid))

        # TODO: HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))

        # TODO: HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))

        # TODO: Cross-Encoder Score
        feature_vector.append(self.get_cross_encoder_score(docid, query))

        # TODO: Add at least one new feature to be used with your L2R model

        # TODO: Document Categories
        #       This should be a list of binary values indicating which categories are present
        feature_vector.extend(self.get_document_categories(docid))

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": multiprocessing.cpu_count()-1,
            "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: Initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.params = default_params
        self.ranker = lightgbm.LGBMRanker(**self.params)

    def fit(self, X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples
            y_train (array-like): Target values
            qgroups_train (array-like): Query group sizes for training data

        Returns:
            self: Returns the instance itself
        """
        # TODO: Fit the LGBMRanker's parameters using the provided features and labels
        X_train = np.array(X_train)
        self.ranker.fit(X_train, y_train, group=qgroups_train)

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """
        # TODO: Generate the predicted values using the LGBMRanker
        featurized_docs = np.array(featurized_docs)
        if len(featurized_docs.shape) == 1:
            return []
        ypred = self.ranker.predict(featurized_docs)
        return ypred


if __name__ == "__main__":
    import gzip
    import json

    from indexing import Indexer, IndexType, BasicInvertedIndex
    from document_preprocessor import RegexTokenizer, Doc2QueryAugmenter
    from ranker import Ranker, BM25, CrossEncoderScorer
    from vector_ranker import VectorRanker
    from network_features import NetworkFeatures
    from l2r import L2RFeatureExtractor, L2RRanker

    # change these to point to actual file paths
    DRIVE_PATH = '../data/'
    CACHE_PATH = '../cache/'
    STOPWORD_PATH = DRIVE_PATH + 'stopwords.txt'
    DATASET_PATH = DRIVE_PATH + 'wikipedia_200k_dataset.jsonl.gz'
    EDGELIST_PATH = DRIVE_PATH + 'edgelist.csv'
    NETWORK_STATS_PATH = DRIVE_PATH + 'network_stats.csv'
    DOC2QUERY_PATH = DRIVE_PATH + 'doc2query.csv'
    MAIN_INDEX = 'main_index_augmented'
    TITLE_INDEX = 'title_index'
    RELEVANCE_TRAIN_DATA = DRIVE_PATH + 'hw3_relevance.train.csv'
    ENCODED_DOCUMENT_EMBEDDINGS_NPY_DATA = DRIVE_PATH + \
        'wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy'

    # Load in the stopwords

    stopwords = set()
    with open(STOPWORD_PATH, 'r', encoding='utf-8') as file:
        for stopword in file:
            stopwords.add(stopword.strip())
    f'Stopwords collected {len(stopwords)}'

    # Get the list of categories for each page (either compute it or load the pre-computed list)
    file_path = CACHE_PATH + 'docid_to_categories.pkl'
    if not os.path.exists(file_path):
        docid_to_categories = {}
        print('Collecting document categories...')
        with gzip.open(DATASET_PATH, 'rt') as file:
            for line in tqdm(file, total=200_000):
                document = json.loads(line)
                docid_to_categories[document['docid']] = document['categories']
        pickle.dump(docid_to_categories, open(file_path, 'wb'))
    else:
        docid_to_categories = pickle.load(open(file_path, 'rb'))
    f'Document categories collected'

    # Get or pre-compute the list of categories at least the minimum number of times (specified in the homework)
    category_counts = Counter()
    print("Counting categories...")
    for cats in tqdm(docid_to_categories.values(), total=len(docid_to_categories)):
        for c in cats:
            category_counts[c] += 1
    recognized_categories = set(
        [cat for cat, count in category_counts.items() if count >= 1000])
    print("saw %d categories" % len(recognized_categories))

    # Map each document to the smallert set of categories that occur frequently
    doc_category_info = {}
    print("Mapping documents to categories...")
    for docid, cats in tqdm(docid_to_categories.items(), total=len(docid_to_categories)):
        valid_cats = [c for c in cats if c in recognized_categories]
        doc_category_info[docid] = valid_cats

    network_features = {}
    # Get or load the network statistics for the Wikipedia link network

    if not os.path.exists(NETWORK_STATS_PATH):
        nf = NetworkFeatures()
        print('loading network')
        graph = nf.load_network(EDGELIST_PATH, total_edges=92650947)
        print('getting stats')
        net_feats_df = nf.get_all_network_statistics(graph)
        graph = None
        print('Saving')
        net_feats_df.to_csv(NETWORK_STATS_PATH, index=False)

        print("converting to dict format")
        network_features = defaultdict(dict)
        for i, row in tqdm(net_feats_df.iterrows(), total=len(net_feats_df)):
            for col in ['pagerank', 'hub_score', 'authority_score']:
                network_features[row['docid']][col] = row[col]
        net_feats_df = None
    else:
        with open(NETWORK_STATS_PATH, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                if idx == 0:
                    continue
                else:
                    # the indexes may change depending on your CSV
                    splits = line.strip().split(',')
                    network_features[int(splits[0])] = {
                        'pagerank': float(splits[1]),
                        'authority_score': float(splits[2]),
                        'hub_score': float(splits[3])
                    }
    f'Network stats collection {len(network_features)}'

    doc_augment_dict = defaultdict(lambda: [])
    with open(DOC2QUERY_PATH, 'r', encoding='utf-8') as file:
        dataset = csv.reader(file)
        print('Collecting document augmentations...')
        for idx, row in tqdm(enumerate(dataset), total=600_000):
            if idx == 0:
                continue
            doc_augment_dict[int(row[0])].append(row[2])

    # Load or build Inverted Indices for the documents' main text and titles
    #
    # Estiamted times:
    #    Document text token counting: 4 minutes
    #    Document text indexing: 5 minutes
    #    Title text indexing: 30 seconds
    preprocessor = RegexTokenizer('\w+')

    # Creating and saving the index

    main_index_path = CACHE_PATH + MAIN_INDEX
    if not os.path.exists(main_index_path):
        main_index = Indexer.create_index(
            IndexType.InvertedIndex, DATASET_PATH, preprocessor,
            stopwords, 50, doc_augment_dict=doc_augment_dict)
        main_index.save(main_index_path)
    else:
        main_index = BasicInvertedIndex()
        main_index.load(main_index_path)

    title_index_path = CACHE_PATH + TITLE_INDEX
    if not os.path.exists(title_index_path):
        title_index = Indexer.create_index(
            IndexType.InvertedIndex, DATASET_PATH, preprocessor,
            stopwords, 2, text_key='title')
        title_index.save(title_index_path)
    else:
        title_index = BasicInvertedIndex()
        title_index.load(title_index_path)

    # create the raw text dictionary by going through the wiki dataset
    # this dictionary should store only the first 500 characters of the raw documents text

    file_path = CACHE_PATH + 'raw_text_dict.pkl'
    if not os.path.exists(file_path):
        raw_text_dict = {}
        with gzip.open(DATASET_PATH, 'rt') as file:
            for line in tqdm(file, total=200_000):
                data = json.loads(line)
                raw_text_dict[int(data['docid'])] = data['text'][:500]
        pickle.dump(raw_text_dict, open(file_path, 'wb'))
    else:
        raw_text_dict = pickle.load(open(file_path, 'rb'))
    
    # Create the feature extractor. This will be used by both pipelines
    cescorer = CrossEncoderScorer(raw_text_dict)
    fe = L2RFeatureExtractor(main_index, title_index, doc_category_info,
                            preprocessor, stopwords, recognized_categories,
                            network_features, cescorer)

    # Create an intial ranker for determining what to re-rank
    # Use these with a L2RRanker and then train that L2RRanker model
    #
    # Estimated time (using 4 cores via n_jobs): 7 minutes

    # An initial ranking with BM25 with reranking done by LambdaMART optimizing NDCG
    bm25 = BM25(main_index)
    ranker = Ranker(main_index, preprocessor, stopwords, bm25)

    pipeline_1 = L2RRanker(main_index, title_index, preprocessor,
                        stopwords, ranker, fe)

    pipeline_1.train(RELEVANCE_TRAIN_DATA)

    # An initial ranking with VectorRanker with reranking done by LambdaMART optimizing NDCG
    with open(ENCODED_DOCUMENT_EMBEDDINGS_NPY_DATA, 'rb') as file:
        encoded_docs = np.load(file)

    vector_ranker = VectorRanker('sentence-transformers/msmarco-MiniLM-L12-cos-v5',
                                encoded_docs, list(main_index.document_metadata.keys()))

    pipeline_2 = L2RRanker(main_index, title_index, preprocessor,
                        stopwords, vector_ranker, fe)

    pipeline_2.train(RELEVANCE_TRAIN_DATA)

    # feature_extractor = L2RFeatureExtractor(
    #     document_index, title_index, doc_category_info,
    #     document_preprocessor, stopwords, recognized_categories,
    #     docid_to_network_features)

    # ranker = L2RRanker(
    #     document_index, title_index, document_preprocessor,
    #     stopwords, BM25, feature_extractor)

    # ranker.train(RELEVANCE_DATA_FILENAME)
    # ranker.save_model()

    # # ranker.load_model()