'''
Author: Zim Gong
This file is a template code file for the Search Engine. 
'''
from IPython.display import display as Display
import numpy as np
import pandas as pd
from tqdm import tqdm
from models import BaseSearchEngine, SearchResponse
from document_preprocessor import RegexTokenizer
from indexing import IndexType, Indexer
# your library imports go here
from ranker import *
from cf import CFRanker
from l2r import L2RRanker, L2RFeatureExtractor
from vector_ranker import VectorRanker
from utils import DATA_PATH, CACHE_PATH

NREL_CORPUS_PATH = DATA_PATH + 'NREL_corpus.jsonl'
NREL_NUMERICAL_PATH = DATA_PATH + 'NREL_numerical.csv'
STOPWORDS_PATH = DATA_PATH + 'stopwords.txt'
STATIONS_INFO_PATH = DATA_PATH + 'NREL_raw.csv'
RELEVANCE_TRAIN_PATH = DATA_PATH + 'relevance.train.csv'
DOC_IDS_PATH = DATA_PATH + 'document-ids.txt'
SEARCH_RADIUS = 5


class SearchEngine(BaseSearchEngine):
    def __init__(self, ranker: str = 'dist', reranker: str = None) -> None:
        self.document_preprocessor = RegexTokenizer("\\w+")
        print('Loading stopwords...')
        self.stopwords = set()
        with open(STOPWORDS_PATH, 'r') as file:
            for line in file:
                cleaned_line = line.strip()
                self.stopwords.add(cleaned_line)

        print('Loading indexes...')
        self.frame = pd.read_csv(NREL_NUMERICAL_PATH)
        self.document_index = Indexer.create_index(index_type=IndexType.InvertedIndex,
                                                   dataset_path=NREL_CORPUS_PATH,
                                                   document_preprocessor=RegexTokenizer(
                                                       "\\w+"),
                                                   stopwords=self.stopwords,
                                                   minimum_word_frequency=2,
                                                   text_key='text',
                                                   max_docs=-1,
                                                   doc_augment_dict=None,
                                                   rel_ids=None)

        self.title_index = Indexer.create_index(index_type=IndexType.InvertedIndex,
                                                dataset_path=NREL_CORPUS_PATH,
                                                document_preprocessor=RegexTokenizer(
                                                    "\\w+"),
                                                stopwords=self.stopwords,
                                                minimum_word_frequency=1,
                                                text_key='name',
                                                max_docs=-1,
                                                doc_augment_dict=None,
                                                rel_ids=None)

        print('Loading ranker...')
        self.set_ranker(ranker)
        self.set_reranker(reranker)

        print('Search Engine initialized!')

    def set_ranker(self, ranker: str = 'dist') -> None:
        if ranker == 'dist':
            self.scorer = DistScorer(self.frame)
        else:
            raise ValueError("Invalid ranker type")
        self.ranker = Ranker(self.frame, scorer=self.scorer)
        self.pipeline = self.ranker

    def set_reranker(self, reranker: str = None) -> None:
        if reranker == 'cf':
            print('Loading cf ranker...')
            self.pipeline = CFRanker(self.frame, self.ranker)
            self.reranker = 'cf'
        elif reranker == 'l2r':
            print('Loading l2r ranker...')
            self.feature_extractor = L2RFeatureExtractor(self.document_index, self.title_index,
                                                         self.document_preprocessor, self.stopwords,
                                                         self.frame, self.ranker)
            self.pipeline = L2RRanker(
                frame=self.frame,
                document_index=self.document_index,
                title_index=self.title_index,
                document_preprocessor=self.document_preprocessor,
                stopwords=self.stopwords,
                ranker=self.ranker,
                feature_extractor=self.feature_extractor)
            self.pipeline.train(DATA_PATH + 'relevance.train.csv')
            self.reranker = 'l2r'
        elif reranker == 'l2r+cf':
            print('Loading l2r ranker...')
            self.feature_extractor = L2RFeatureExtractor(
                self.frame, self.ranker)
            self.l2r = L2RRanker(
                frame=self.frame,
                document_index=self.document_index,
                title_index=self.title_index,
                document_preprocessor=self.document_preprocessor,
                stopwords=self.stopwords,
                ranker=self.ranker,
                feature_extractor=self.feature_extractor)
            self.l2r.train(DATA_PATH + 'relevance.train.csv')
            self.reranker = 'l2r+cf'
            self.pipeline = CFRanker(self.frame, self.l2r)
        elif reranker == 'vector':
            encoded_docs = np.load(DATA_PATH + 'encoded_station.npy')
            user_profile = np.load(DATA_PATH + 'encoded_user_profile.npy')

            file_path = DATA_PATH + 'row_to_docid.txt'
            with open(file_path, 'r') as file:
                row_to_docid = file.read().splitlines()

            row_to_docid = [int(i) for i in row_to_docid]

            profile_row_to_userid = [1, 2]
            self.pipeline = VectorRanker(
                index=self.frame,
                ranker=self.ranker,
                bi_encoder_model_name=None,
                encoded_docs=encoded_docs,
                row_to_docid=row_to_docid,
                user_profile=user_profile,
                profile_row_to_userid=profile_row_to_userid)
            self.reranker = 'vector'
        elif reranker == 'l2r+vector':
            print('Loading l2r ranker...')
            self.feature_extractor = L2RFeatureExtractor(self.document_index, self.title_index,
                                                         self.document_preprocessor, self.stopwords,
                                                         self.frame, self.ranker)
            self.l2r = L2RRanker(
                frame=self.frame,
                document_index=self.document_index,
                title_index=self.title_index,
                document_preprocessor=self.document_preprocessor,
                stopwords=self.stopwords,
                ranker=self.ranker,
                feature_extractor=self.feature_extractor)

            self.l2r.train(DATA_PATH + 'relevance.train.csv')
            encoded_docs = np.load(DATA_PATH + 'encoded_station.npy')
            user_profile = np.load(DATA_PATH + 'encoded_user_profile.npy')

            file_path = DATA_PATH + 'row_to_docid.txt'
            with open(file_path, 'r') as file:
                row_to_docid = file.read().splitlines()

            row_to_docid = [int(i) for i in row_to_docid]

            profile_row_to_userid = [1, 2]
            self.pipeline = VectorRanker(
                index=self.frame,
                ranker=self.l2r,
                bi_encoder_model_name=None,
                encoded_docs=encoded_docs,
                row_to_docid=row_to_docid,
                user_profile=user_profile,
                profile_row_to_userid=profile_row_to_userid)
            self.reranker = 'l2r+vector'
        else:
            self.reranker = None
            self.pipeline = self.ranker

    def search(self, query: str, **kwargs) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        results = self.pipeline.query(query, **kwargs)
        if results is None or results == []:
            print('No results found')
            return []
        try:
            return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]
        except:
            return [SearchResponse(id=idx+1, docid=result, score=0) for idx, result in enumerate(results)]

    def get_station_info(self, docid_list):
        self.detailed_data = pd.read_csv(STATIONS_INFO_PATH, delimiter='\t')
        return self.detailed_data[self.detailed_data['ID'].isin(docid_list)][[
            'Station Name', 'Street Address', 'Intersection Directions',
            'City', 'State', 'ZIP', 'Plus4', 'Station Phone',
            'Access Days Time', 'Cards Accepted',
            'EV Level1 EVSE Num', 'EV Level2 EVSE Num', 'EV DC Fast Count',
            'EV Other Info', 'EV Network', 'EV Network Web',
            'Geocode Status', 'Latitude', 'Longitude',
            'Date Last Confirmed', 'ID', 'Updated At', 'Owner Type Code',
            'Federal Agency ID', 'Federal Agency Name', 'Open Date',
            'EV Connector Types', 'Country', 'Access Code', 'Access Detail Code',
            'Federal Agency Code', 'Facility Type',
            'EV Pricing', 'EV On-Site Renewable Source', 'Restricted Access',
            'NPS Unit Name', 'Maximum Vehicle Class', 'EV Workplace Charging'
        ]]
    
    def get_gpt_info(self, docid_list):
        self.detailed_data = pd.read_csv('../data/station_personalized_features.csv')
        return self.detailed_data[self.detailed_data['docid'].isin(docid_list)]


def initialize():
    search_obj = SearchEngine(cf=False, l2r=False)
    return search_obj


# DATA_PATH = 'data/'
DEFAULT_PROMPT = "1401"
DEFAULT_LNG = "-83.0703"
DEFAULT_LAT = "42.3317"
DEFAULT_USER = 1


def get_results_all(engine: SearchEngine, lat=DEFAULT_LAT, lng=DEFAULT_LNG, prompt=DEFAULT_PROMPT, top_n=10, user_id=DEFAULT_USER):
    query = str(lat) + ", " + str(lng) + ", " + str(prompt)
    # + str(prompt)
    param = {
        "user_id": user_id,
    }
    print(query)
    results = engine.search(query, **param)
    results = results[:top_n]
    return results


if __name__ == "__main__":
    search_obj = SearchEngine(reranker="l2r")
    res = get_results_all(search_obj, lat=DEFAULT_LAT,
                          lng=DEFAULT_LNG, prompt=DEFAULT_PROMPT, user_id=1)
    result_df = search_obj.get_station_info([i.docid for i in res])
    Display(result_df)
