from enum import Enum
import json
import os
from tqdm import tqdm
from collections import Counter, defaultdict
import shelve
from document_preprocessor import Tokenizer
import gzip
import pickle
import jsonlines


class IndexType(Enum):
    # The three types of index currently supported are InvertedIndex, PositionalIndex and OnDiskInvertedIndex
    InvertedIndex = 'BasicInvertedIndex'
    # NOTE: You don't need to support the following three
    PositionalIndex = 'PositionalIndex'
    OnDiskInvertedIndex = 'OnDiskInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.
        """
        self.statistics = Counter()
        self.index = {}  # Index
        # Metadata like length, number of unique tokens of the documents
        self.document_metadata = {}
        self.vocabulary = set()
        self.stop_words = set()
        self.min_word_freq = 0
        self.total_token_count = 0

    # NOTE: The following functions have to be implemented in the three inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        # TODO: Implement this to remove a document from the entire index and statistics
        for pos, token in enumerate(self.index):
            for posting in token:
                if posting[0] == docid:
                    token.remove(posting)
                    if len(token) == 0:
                        del self.index[pos]
                        self.vocabulary.remove(token)
                    break
        del self.document_metadata[docid]
        self.total_token_count -= self.get_doc_metadata(docid)["length"]

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
        # TODO: Implement this to add documents to the index
        tokens_index = Counter(tokens)
        for token in tokens_index:
            if token in self.stop_words:
                continue
            if token not in self.vocabulary:
                postings_list = self.add_to_vocabulary(token)
            else:
                postings_list = self.get_postings(token)
            self.add_to_postings(postings_list, docid, tokens_index[token])
        self.document_metadata[docid] = (len(tokens_index), len(tokens))
        self.total_token_count += len(tokens)

    def add_to_vocabulary(self, token: str) -> list[tuple[int, str]]:
        self.vocabulary.add(token)
        self.index[token] = []
        return self.index[token]

    def add_to_postings(self, postings_list: list[tuple[int, str]], docid: int, freq: int) -> None:
        postings_list.append((docid, freq))

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in 
            the document
        """
        # TODO: Implement this to fetch a term's postings from the index
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
        # TODO: Implement to fetch a particular document stored in metadata
        unique_tokens = 0
        length = 0
        if doc_id in self.document_metadata:
            unique_tokens = self.document_metadata[doc_id][0]
            length = self.document_metadata[doc_id][1]
        return {"unique_tokens": unique_tokens, "length": length}

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
        # TODO: Implement to fetch a particular term stored in metadata
        count = 0
        num_docs = 0
        try:
            data = self.statistics[term]
            count = data[0]
            num_docs = data[1]
        except:
            pass
        return {"count": count, "num_docs": num_docs}

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
            "unique_token_count": unique_token_count,
            "total_token_count": total_token_count,
            "number_of_documents": number_of_documents,
            "mean_document_length": mean_document_length
        }

    def synchronize(self) -> None:
        self.statistics = Counter()
        count = 0
        delete_list = []
        for key in self.index:
            term_freq = 0
            term_num_docs = 0
            count += 1
            for posting in self.index[key]:
                term_freq += posting[1]
                term_num_docs += 1
            if term_freq < self.min_word_freq:
                delete_list.append(key)
                continue
            self.statistics[key] = (term_freq, term_num_docs)
        for key in delete_list:
            del self.index[key]
            self.vocabulary.remove(key)

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
        pickle.dump(self.index, open(
            index_directory_name + "/index.pkl", "wb"))
        pickle.dump(self.document_metadata, open(
            index_directory_name + "/document_metadata.pkl", "wb"))
        pickle.dump(self.vocabulary, open(
            index_directory_name + "/vocabulary.pkl", "wb"))
        pickle.dump(self.statistics, open(
            index_directory_name + "/statistics.pkl", "wb"))
        pickle.dump(self.total_token_count, open(
            index_directory_name + "/total_token_count.pkl", "wb"))

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save().

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        # TODO: Load the index files from disk to a Python object
        self.index = pickle.load(
            open(index_directory_name + "/index.pkl", "rb"))
        self.document_metadata = pickle.load(
            open(index_directory_name + "/document_metadata.pkl", "rb"))
        self.vocabulary = pickle.load(
            open(index_directory_name + "/vocabulary.pkl", "rb"))
        self.statistics = pickle.load(
            open(index_directory_name + "/statistics.pkl", "rb"))
        self.total_token_count = pickle.load(
            open(index_directory_name + "/total_token_count.pkl", "rb"))


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        # For example, you can initialize the index and statistics here:
        #    self.statistics['docmap'] = {}
        #    self.index = defaultdict(list)
        #    self.doc_id = 0

    # TODO: Implement all the functions mentioned in the interface
    # This is the typical inverted index where each term keeps track of documents and the term count per document


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self, index_name) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__(index_name)
        self.statistics['index_type'] = 'PositionalInvertedIndex'
        # For example, you can initialize the index and statistics here:
        #   self.statistics['offset'] = [0]
        #   self.statistics['docmap'] = {}
        #   self.doc_id = 0
        #   self.postings_id = -1

    # TODO: Do nothing, unless you want to explore using a positional index for some cool features


class OnDiskInvertedIndex(BasicInvertedIndex):
    def __init__(self, shelve_filename) -> None:
        """
        This is an inverted index where the inverted index's keys (words) are kept in memory but the
        postings (list of documents) are on desk.
        The on-disk part is expected to be handled via a library.
        """
        super().__init__()
        self.shelve_filename = shelve_filename
        self.statistics['index_type'] = 'OnDiskInvertedIndex'
        # Ensure that the directory exists
        # self.index = shelve.open(self.shelve_filename, 'index')
        # self.statistics['docmap'] = {}
        # self.doc_id = 0

    # NOTE: Do nothing, unless you want to re-experience the pain of cross-platform compatibility :'(


class Indexer:
    """
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    """
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
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
        # TODO (HW3): This function now has an optional argument doc_augment_dict; check README

        # HINT: Think of what to do when doc_augment_dict exists, how can you deal with the extra information?
        #       How can you use that information with the tokens?
        #       If doc_augment_dict doesn't exist, it's the same as before, tokenizing just the document text

        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        #       For HW3, only the BasicInvertedIndex is required to be supported
        index = BasicInvertedIndex()

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies
        index.min_word_freq = minimum_word_frequency
        if stopwords is not None:
            index.stop_words = stopwords

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input
        if dataset_path.endswith('.jsonl.gz'):
            dataset_file = gzip.open(dataset_path, 'rt')
        elif dataset_path.endswith('.jsonl'):
            dataset_file = open(dataset_path, 'r')
        else:
            raise Exception(
                "Unsupported file format. Only .jsonl.gz and .jsonl are supported")
        with jsonlines.Reader(dataset_file) as reader:
            if max_docs == -1:
                max_docs = 200000
            print("Indexing...")
            for _ in tqdm(range(max_docs)):
                try:
                    document = reader.read()
                    tokens = document_preprocessor.tokenize(
                        document[text_key])
                    if doc_augment_dict is not None:
                        queries = doc_augment_dict[document['docid']]
                        for query in queries:
                            tokens += document_preprocessor.tokenize(query)
                    index.add_doc(document['docid'], tokens)
                except EOFError:
                    break

        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency
        index.synchronize()

        # HINT: This homework should work fine on a laptop with 8GB of memory but if you need,
        #       you can delete some unused objects here to free up some space

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency

        return index


if __name__ == "__main__":
    import csv

    from indexing import Indexer, IndexType, BasicInvertedIndex
    from document_preprocessor import RegexTokenizer, Doc2QueryAugmenter
    from ranker import Ranker, BM25, CrossEncoderScorer
    from vector_ranker import VectorRanker
    from network_features import NetworkFeatures
    from l2r import L2RFeatureExtractor, L2RRanker

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

    # Load or build Inverted Indices for the documents' main text and titles
    #
    # Estiamted times:
    #    Document text token counting: 4 minutes
    #    Document text indexing: 5 minutes
    #    Title text indexing: 30 seconds
    preprocessor = RegexTokenizer('\w+')

    doc_augment_dict = defaultdict(lambda: [])
    with open(DOC2QUERY_PATH, 'r', encoding='utf-8') as file:
        dataset = csv.reader(file)
        for idx, row in tqdm(enumerate(dataset), total=600_000):
            if idx == 0:
                continue
            doc_augment_dict[int(row[0])].append(row[2])

    # Creating and saving the index

    main_index_path = CACHE_PATH + MAIN_INDEX
    if not os.path.exists(main_index_path):
        main_index = Indexer.create_index(
            IndexType.InvertedIndex, DATASET_PATH, preprocessor,
            stopwords, 50, doc_augment_dict=doc_augment_dict,
            max_docs=1000)
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
        title_index.load(TITLE_INDEX)
