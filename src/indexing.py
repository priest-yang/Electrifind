import gzip
import json
import jsonlines
import os
import pickle
import shelve
from collections import Counter, defaultdict
from document_preprocessor import Tokenizer, RegexTokenizer
from enum import Enum
from tqdm import tqdm
from utils.py import merge_lat_lng

class IndexType(Enum):
    # The three types of index currently supported are InvertedIndex, PositionalIndex and OnDiskInvertedIndex
    InvertedIndex = 'BasicInvertedIndex'
    PositionalIndex = 'PositionalIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        self.statistics = Counter()  # Central statistics of the index
        self.index = {}  # Index
        # Metadata like length, number of unique tokens of the documents
        self.document_metadata = {}
        self.vocabulary = set()
        self.stop_words = set()
        self.min_word_freq = 0
        self.total_token_count = 0

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Remove a document from the entire index and statistics
        for idx, token in enumerate(self.index):
            for posting in token:
                if posting[0] == docid:
                    token.remove(posting)
                    if len(token) == 0:
                        del self.index[idx]
                        self.vocabulary.remove(token)
                    break
        del self.document_metadata[docid]
        self.total_token_count -= self.get_doc_metadata(docid)['length']

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Add documents to the index
        doc_index = Counter(tokens)
        for token in doc_index:
            if token in self.stop_words:
                continue
            if token not in self.vocabulary:
                postings_list = self.add_to_vocabulary(token)
            else:
                postings_list = self.get_postings(token)
            self.add_to_postings(postings_list, docid, doc_index[token])
        self.document_metadata[docid] = (len(doc_index), len(tokens))
        self.total_token_count += len(tokens)

    def add_to_vocabulary(self, token: str) -> list[tuple[int, int]]:
        self.vocabulary.add(token)
        self.index[token] = []
        return self.index[token]

    def add_to_postings(self, postings_list: list[tuple[int, int]], docid: int, freq: int) -> None:
        postings_list.append((docid, freq))

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        This information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document
        """
        # TODO: Fetch a term's postings from the index
        return self.index[term]

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        # TODO: Fetch a particular document stored in metadata
        unique_tokens = 0
        length = 0
        if doc_id in self.document_metadata:
            unique_tokens = self.document_metadata[doc_id][0]
            length = self.document_metadata[doc_id][1]
        return {'unique_tokens': unique_tokens, 'length': length}

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "count": How many times this term appeared in the corpus as a whole

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        # TODO: Fetch a particular term stored in metadata
        count = 0
        num_docs = 0
        if term in self.statistics:
            data = self.statistics[term]
            count = data[0]
            num_docs = data[1]
        return {'count': count, 'num_docs': num_docs}

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary mapping statistical properties (named as strings) about the index to their values.  
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens), 
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
              A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        # TODO: Calculate statistics like 'unique_token_count', 'total_token_count',
        #       'number_of_documents', 'mean_document_length' and any other relevant central statistic
        unique_token_count = len(self.vocabulary)
        total_token_count = self.total_token_count
        number_of_documents = len(self.document_metadata)
        mean_document_length = 0
        if number_of_documents > 0:
            mean_document_length = total_token_count / number_of_documents
        return {
            'unique_token_count': unique_token_count,
            'total_token_count': total_token_count,
            'number_of_documents': number_of_documents,
            'mean_document_length': mean_document_length
        }

    def sync_metadata(self) -> None:
        self.statistics = Counter()
        count = 0
        delete_list = []
        for token in self.index:
            term_freq = 0
            term_num_docs = 0
            count += 1
            for posting in self.index[token]:
                term_freq += posting[1]
                term_num_docs += 1
            if term_freq < self.min_word_freq:
                delete_list.append(token)
                continue
            self.statistics[token] = (term_freq, term_num_docs)
        for token in delete_list:
            del self.index[token]
            self.vocabulary.remove(token)

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        # TODO: Save the index files to disk
        if not os.path.exists(index_directory_name):
            os.makedirs(index_directory_name)
        pickle.dump(self.index, open(index_directory_name + '/index', 'wb'))
        pickle.dump(self.document_metadata, open(
            index_directory_name + '/document_metadata.pkl', 'wb'))
        pickle.dump(self.statistics, open(
            index_directory_name + '/statistics.pkl', 'wb'))
        pickle.dump(self.vocabulary, open(
            index_directory_name + '/vocabulary.pkl', 'wb'))
        pickle.dump(self.total_token_count, open(
            index_directory_name + '/total_token_count.pkl', 'wb'))

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save().

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        # TODO: Load the index files from disk to a Python object
        self.index = pickle.load(open(index_directory_name + '/index', 'rb'))
        self.document_metadata = pickle.load(
            open(index_directory_name + '/document_metadata.pkl', 'rb'))
        self.statistics = pickle.load(
            open(index_directory_name + '/statistics.pkl', 'rb'))
        self.vocabulary = pickle.load(
            open(index_directory_name + '/vocabulary.pkl', 'rb'))
        self.total_token_count = pickle.load(
            open(index_directory_name + '/total_token_count.pkl', 'rb'))


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'

    # This is the typical inverted index where each term keeps track of documents and the term count per document


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self, index_name) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Adds a document to the index and updates the index's metadata on the basis of this
        document's addition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # TODO: Add documents to the index
        doc_index = {}
        for idx, token in enumerate(tokens):
            if token.lower() not in doc_index:
                doc_index[token.lower()] = [1, [idx]]
            else:
                doc_index[token.lower()][0] += 1
                doc_index[token.lower()][1].append(idx)
        for token in doc_index:
            if token in self.stop_words:
                continue
            if token not in self.vocabulary:
                postings_list = self.add_to_vocabulary(token)
            else:
                postings_list = self.get_postings(token)
            self.add_to_postings(postings_list, docid,
                                 doc_index[token][0], doc_index[token][1])
        self.document_metadata[docid] = (len(doc_index), len(tokens))
        self.total_token_count += len(tokens)

    def add_to_postings(self, postings_list: list[tuple[int, int]], docid: int, freq: int, pos: int) -> None:
        postings_list.append((docid, freq, pos))


class Indexer:
    """
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    """
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None,
                     rel_ids: list[int] = None) -> InvertedIndex:
        """
        Creates an inverted index.

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text
                and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the document at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index
        """
        # TODO: Argument doc_augment_dict.
        #       This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document
        if index_type == IndexType.InvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies
        index.min_word_freq = minimum_word_frequency
        index.stop_words = stopwords

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input
        if max_docs == -1:
            max_docs = 200000
        if dataset_path.endswith('.jsonl.gz'):
            dataset_file = gzip.open(dataset_path, 'rt', encoding='utf8')
        elif dataset_path.endswith('.jsonl'):
            dataset_file = open(dataset_path, 'r', encoding='utf8')
        else:
            raise Exception('Unsupported file extension')
        with jsonlines.Reader(dataset_file) as reader:
            if rel_ids is not None:
                for _ in tqdm(range(max_docs)):
                    try:
                        document = reader.read()
                        if document['docid'] not in rel_ids:
                            continue
                        tokens = document_preprocessor.tokenize(
                            document[text_key])
                        if doc_augment_dict is not None:
                            queries = doc_augment_dict[document['docid']]
                            for query in queries:
                                tokens.extend(
                                    document_preprocessor.tokenize(query))
                        index.add_doc(document['docid'], tokens)
                    except EOFError:
                        break
            else:
                for _ in tqdm(range(max_docs)):
                    try:
                        document = reader.read()
                        tokens = document_preprocessor.tokenize(
                            document[text_key])
                        if doc_augment_dict is not None:
                            queries = doc_augment_dict[document['docid']]
                            for query in queries:
                                tokens.extend(
                                    document_preprocessor.tokenize(query))
                        index.add_doc(document['docid'], tokens)
                    except EOFError:
                        break

        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency
        index.sync_metadata()

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency

        return index


if __name__ == '__main__':

    # read stop words
    stopwords_file_path = 'data/stopwords.txt'
    stopwords = set()
    with open(stopwords_file_path, 'r') as file:
        for line in file:
            cleaned_line = line.strip()
            stopwords.add(cleaned_line)

    index = Indexer.create_index(index_type=IndexType.InvertedIndex, 
                                 dataset_path='data/google_map_charging_station_all.jsonl.gz', 
                                 document_preprocessor=RegexTokenizer("\\w+"),
                                 stopwords=stopwords, 
                                minimum_word_frequency=2,
                                text_key='comment', 
                                max_docs=-1, 
                                doc_augment_dict=None, 
                                rel_ids=None)
    
    index

    pass
