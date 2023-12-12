'''
Author: Zim Gong
This file is a template code file for the Search Engine. 
'''
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import BaseSearchEngine, SearchResponse

# your library imports go here
from ranker import *
from cf import CFRanker

DATA_PATH = '../data/'
CACHE_PATH = '../cache/'
DATASET_PATH = DATA_PATH + 'NREL_All_Stations_data_si618.csv'
DOC2QUERY_PATH = DATA_PATH + 'doc2query.csv'
ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = DATA_PATH + \
    'wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy'
NETWORK_STATS_PATH = DATA_PATH + 'network_stats.csv'
EDGELIST_PATH = DATA_PATH + 'edgelist.csv'
RELEVANCE_TRAIN_PATH = DATA_PATH + 'relevance.train.csv'
DOC_IDS_PATH = DATA_PATH + 'document-ids.txt'


class SearchEngine(BaseSearchEngine):
    def __init__(self,
                 max_docs: int = -1,
                 ranker: str = 'dist',
                 cf: bool = True
                 ) -> None:

        self.cf = False

        print('Loading indexes...')
        self.main_index = pd.read_csv(DATASET_PATH, delimiter='\t')

        print('Loading ranker...')
        self.set_ranker(ranker)
        self.set_cf(cf)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'dist') -> None:
        if ranker == 'dist':
            self.scorer = DistScorer(self.main_index)
        else:
            raise ValueError("Invalid ranker type")
        self.ranker = Ranker(self.main_index, self.scorer)
        if self.cf:
            self.pipeline.ranker = self.ranker
        else:
            self.pipeline = self.ranker

    def set_cf(self, cf: bool = True) -> None:
        if self.cf == cf:
            return
        if not cf:
            self.pipeline = self.ranker
        else:
            print('Loading cf ranker...')
            self.pipeline = cfRanker()

            self.cf = True

    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    search_obj = SearchEngine()
    return search_obj


if __name__ == "__main__":
    search_obj = SearchEngine(cf=False)