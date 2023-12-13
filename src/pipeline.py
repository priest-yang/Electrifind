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
from l2r import L2RRanker, L2RFeatureExtractor
from vector_ranker import VectorRanker
from utils import DATA_PATH, CACHE_PATH
# DATA_PATH = 'data/'
# CACHE_PATH = 'cache/'
DATASET_PATH = DATA_PATH + 'processed_nrel.csv'
DOC2QUERY_PATH = DATA_PATH + 'doc2query.csv'
ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = DATA_PATH + \
    'wiki-200k-vecs.msmarco-MiniLM-L12-cos-v5.npy'
NETWORK_STATS_PATH = DATA_PATH + 'network_stats.csv'
EDGELIST_PATH = DATA_PATH + 'edgelist.csv'
RELEVANCE_TRAIN_PATH = DATA_PATH + 'relevance.train.csv'
DOC_IDS_PATH = DATA_PATH + 'document-ids.txt'


class SearchEngine(BaseSearchEngine):
    def __init__(self, ranker: str = 'dist',
                 reranker: str = None) -> None:

        self.reranker = None

        print('Loading indexes...')
        self.main_index = pd.read_csv(DATASET_PATH)

        print('Loading ranker...')
        self.set_ranker(ranker)
        self.set_reranker(reranker)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'dist') -> None:
        if ranker == 'dist':
            self.scorer = DistScorer(self.main_index)
        else:
            raise ValueError("Invalid ranker type")
        self.ranker = Ranker(self.main_index, scorer=self.scorer)
        self.pipeline = self.ranker

    def set_reranker(self, reranker: str = None) -> None:
        if reranker == 'cf':
            print('Loading cf ranker...')
            self.pipeline = CFRanker(self.main_index, self.ranker)
            self.reranker = 'cf'
        elif reranker == 'l2r':
            print('Loading l2r ranker...')
            self.feature_extractor = L2RFeatureExtractor(
                self.main_index, self.ranker)
            self.pipeline = L2RRanker(
                index=self.main_index, ranker=self.ranker, feature_extractor=self.feature_extractor)
            self.pipeline.train(DATA_PATH + 'relevance.train.csv')
            self.reranker = 'l2r'
        elif reranker == 'l2r+cf':
            print('Loading l2r ranker...')
            self.feature_extractor = L2RFeatureExtractor(
                self.main_index, self.ranker)
            self.l2r = L2RRanker(
                index=self.main_index, ranker=self.ranker, feature_extractor=self.feature_extractor)
            self.l2r.train(DATA_PATH + 'relevance.train.csv')
            self.reranker = 'l2r+cf'
            self.pipeline = CFRanker(self.main_index, self.l2r)
        elif reranker == 'vector':
            encoded_docs = np.load(DATA_PATH + 'encoded_station.npy')
            user_profile = np.load(DATA_PATH + 'encoded_user_profile.npy')

            file_path = DATA_PATH + 'row_to_docid.txt'
            with open(file_path, 'r') as file:
                row_to_docid = file.read().splitlines()

            row_to_docid = [int(i) for i in row_to_docid]

            profile_row_to_userid = [1, 2]
            self.pipeline = VectorRanker(
                index=self.main_index,
                ranker=self.ranker,
                bi_encoder_model_name=None,
                encoded_docs=encoded_docs,
                row_to_docid=row_to_docid,
                user_profile=user_profile,
                profile_row_to_userid=profile_row_to_userid)
            self.reranker = 'vector'
        else:
            self.reranker = None
            self.pipeline = self.ranker

    def search(self, query: str) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query)
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]

    def get_station_info(self, docid_list):
        detailed_data = pd.read_csv(DATA_PATH + 'NREL_All_Stations_data_si618.csv', delimiter='\t')
        return detailed_data.iloc[docid_list]

def initialize():
    search_obj = SearchEngine(cf=False, l2r=False)
    return search_obj


if __name__ == "__main__":
    search_obj = SearchEngine(cf=False)
