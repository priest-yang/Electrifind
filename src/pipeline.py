'''
Author: Zim Gong
This file is a template code file for the Search Engine. 
'''
import csv
import gzip
import json
import jsonlines
import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

from models import BaseSearchEngine, SearchResponse

# your library imports go here
from ranker import *
from l2r import L2RRanker, L2RFeatureExtractor

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
                 l2r: bool = True
                 ) -> None:

        self.l2r = False

        print('Loading indexes...')
        self.main_index = pd.read_csv(DATASET_PATH, delimiter='\t')

        print('Loading ranker...')
        self.set_ranker(ranker)
        self.set_l2r(l2r)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'dist') -> None:
        if ranker == 'dist':
            self.scorer = DistScorer(self.main_index)
        else:
            raise ValueError("Invalid ranker type")
        self.ranker = Ranker(self.main_index, self.scorer)
        if self.l2r:
            self.pipeline.ranker = self.ranker
        else:
            self.pipeline = self.ranker

    def set_l2r(self, l2r: bool = True) -> None:
        if self.l2r == l2r:
            return
        if not l2r:
            self.pipeline = self.ranker
        else:
            self.fe = L2RFeatureExtractor(
                self.main_index, self.title_index, self.doc_category_info,
                self.preprocessor, self.stopwords, self.recognized_categories,
                self.network_features, self.cescorer
            )

            print('Loading L2R ranker...')
            self.pipeline = L2RRanker(
                self.main_index, self.title_index, self.preprocessor,
                self.stopwords, self.ranker, self.fe
            )

            print('Training L2R ranker...')
            self.pipeline.train(RELEVANCE_TRAIN_PATH)
            self.l2r = True

    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]


def initialize():
    search_obj = SearchEngine()
    return search_obj


if __name__ == "__main__":
    search_obj = SearchEngine(l2r=False)