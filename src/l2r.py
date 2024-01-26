from tqdm import tqdm
import pandas as pd
import lightgbm
from indexing import InvertedIndex
import multiprocessing
from collections import defaultdict, Counter
import numpy as np
from document_preprocessor import Tokenizer
from ranker import Ranker, TF_IDF, BM25, PivotedNormalization, CrossEncoderScorer
import csv
import math
import os
import pickle


class L2RRanker:
    def __init__(self, frame: pd.DataFrame, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 ranker: Ranker, feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker system.

        Args:
            frame: The dataframe containing the numeric only data of charging stations
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.frame = frame
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.bm25_scorer = BM25(self.document_index)
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        self.name = 'L2RRanker'

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

            query_parts = [float(x) for x in query.split(',')]

            # TODO: For each of the documents, generate its features, then append
            #       the features and relevance score to the lists to be returned
            for doc in doc_scores:
                docid = doc[0]
                score = doc[1]
                features = self.feature_extractor.generate_features(
                    docid, query_parts)
                X.append(features)
                y.append(score)

            # Keep track of how many scores we have for this query
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

    def train(self, training_data_filename: str, model_name: str = '') -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation
        # if os.path.exists('../cache/' + model_name + 'X.pkl'):
        #     X = pickle.load(open('../cache/' + model_name + 'X.pkl', 'rb'))
        #     y = pickle.load(open('../cache/' + model_name + 'y.pkl', 'rb'))
        #     qgroups = pickle.load(open('../cache/' + model_name + 'qgroups.pkl', 'rb'))
        # else:
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
                        docid = int(row[7])
                        rel_score = row[8]
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
        X, y, qgroups = self.prepare_training_data(query_to_doc_rel_scores)
        # pickle.dump(X, open('../cache/' + model_name + 'X.pkl', 'wb'))
        # pickle.dump(y, open('../cache/' + model_name + 'y.pkl', 'wb'))
        # pickle.dump(qgroups, open('../cache/' + model_name + 'qgroups.pkl', 'wb'))

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

    def query(self, query: str, user_id=None, mmr_lambda: int = 1, mmr_threshold: int = 100) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string seperated by ", " containing the latitude, longitude, and prompt, in format
                "lat, lng, prompt, ..."

            pseudofeedback_num_docs: If pseudo-feedback is requested, the number of top-ranked documents
                to be used in the query
            pseudofeedback_alpha: If pseudo-feedback is used, the alpha parameter for weighting
                how much to include of the original query in the updated query
            pseudofeedback_beta: If pseudo-feedback is used, the beta parameter for weighting
                how much to include of the relevant documents in the updated query
            user_id: the integer id of the user who is issuing the query or None if the user is unknown
            mmr_lambda: Hyperparameter for MMR diversification scoring
            mmr_threshold: Documents to rerank using MMR diversification

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # Retrieve potentially-relevant documents

        if self.ranker.scorer.__class__.__name__ == 'DistScorer':
            query_parts = [x for x in query.split(',')]
            lat = float(query_parts[0])
            lng = float(query_parts[1])
            try:
                prompt = query_parts[2]
            except:
                prompt = None

            print('prompt:', prompt)

            if len(query_parts) == 0:
                return []
            mask = (abs(lat - self.frame.Latitude) <
                    0.01) & (abs(lng - self.frame.Longitude) < 0.01)
            relevant_docs = self.frame[mask]
            if len(relevant_docs) == 0:
                return []
            relevant_docs['score'] = relevant_docs.apply(
                lambda x: self.ranker.scorer.score(x, query_parts), axis=1)
            relevant_docs = relevant_docs.sort_values(
                by=['score'], ascending=False)
            relevant_docs['id'] = relevant_docs.index
            results = relevant_docs[['id', 'score']].values.tolist()

        # Filter to just the top 100 documents for the L2R part for re-ranking
        results_top_100 = results[:100]
        results_tails = results[100:]

        # Construct the feature vectors for each query-document pair in the top 100
        X_pred = []
        for item in results_top_100:
            docid = int(item[0])
            if self.ranker.scorer.__class__.__name__ == 'DistScorer':
                X_pred.append(self.feature_extractor.generate_features(
                    docid, query_parts))
            # else:
            #     X_pred.append(self.feature_extractor.generate_features(
            #         docid, doc_term_counts[docid], title_term_counts[docid], query_parts, query))

        # Use L2R model to rank these top 100 documents
        scores = self.predict(X_pred)

        # Sort posting_lists based on scores
        for i in range(len(results_top_100)):
            results_top_100[i] = (results_top_100[i][0], scores[i])
        results_top_100.sort(key=lambda x: x[1], reverse=True)

        # Add back the other non-top-100 documents that weren't re-ranked
        results = results_top_100 + results_tails

        # further rank prompt using BM25, cut-off = 100
        if prompt is not None:
            CUT_OFF = 100

            query_parts = self.document_preprocessor.tokenize(prompt)

            doc_term_counts = self.accumulate_doc_term_counts(
                self.document_index, query_parts)

            prompt_results = results[:CUT_OFF]
            prompt_results = [[res[0], res[1]] for res in prompt_results]
            for res in prompt_results:
                docid = int(res[0])
                res[1] = self.get_BM25_score(
                    docid, doc_term_counts[docid], query_parts)

            prompt_results.sort(key=lambda x: x[1], reverse=True)
            prompt_results = [(res[0], res[1]) for res in prompt_results]
            results = prompt_results + results[CUT_OFF:]
            # title_term_counts = self.accumulate_doc_term_counts(
            #     self.title_index, query_parts)

        # Return the ranked documents
        return results

    # BM25
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
        query_word_count = Counter(query_parts)
        return self.bm25_scorer.score(docid, doc_word_counts, query_word_count)

    def save_model(self) -> None:
        pickle.dump(self.model.ranker, open('../cache/l2r_model_' +
                    self.ranker.__class__.__name__ + '.pkl', 'wb'))

    def load_model(self) -> None:
        self.model.ranker = pickle.load(
            open('../cache/l2r_model_' + self.ranker.__class__.__name__ + '.pkl', 'rb'))


class L2RFeatureExtractor:
    def __init__(self, data_source, ranker) -> None:
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
        # Set the initial state using the arguments
        self.data_source = data_source
        self.ranker = ranker

    # Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, query_parts: list[float]) -> list:
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

        dist = self.ranker.scorer.score(
            query_parts, self.data_source.iloc[docid])
        feature_vector.append(dist)
        feature_vector.extend(self.data_source.iloc[docid].tolist())

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


if __name__ == '__main__':
    pass
