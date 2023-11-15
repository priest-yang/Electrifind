'''
Author: Prithvijit Dasgupta
This file is a template code file

Also check out Interactive_Example.ipynb
'''

from models import BaseSearchEngine, SearchResponse

# your library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType
from ranker import Ranker, BM25
from l2r import L2RRanker

class SearchEngine(BaseSearchEngine):
    def __init__(self) -> None:
        # 1. Create a document tokenizer using document_preprocessor Tokenizers
        # 2. Load stopwords, network data, categories, etc
        # 3. Create an index using the Indexer and IndexType (with the Wikipedia JSONL and stopwords)
        # 4. Initialize a Ranker/L2RRanker with the index, stopwords, etc.
        # 5. If using L2RRanker, train it here.
      
        # HINT: your code should switch to using a L2RRanker for ranking query results
        pass
      
    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.ranker.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    search_obj = SearchEngine()
    return search_obj
