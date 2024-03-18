'''
Author: Zim Gong
This file is a template code file for the Search Engine. 
'''
import numpy as np
import pandas as pd

from .utils import CACHE_PATH, DATA_PATH
from .models import BaseSearchEngine, SearchResponse
from .document_preprocessor import *
from .indexing import IndexType, Indexer
from .ranker import *
from .cf import CFRanker
from .l2r import L2RRanker, L2RFeatureExtractor
from .vector_ranker import VectorRanker

NREL_PATH = DATA_PATH + 'NREL_raw.csv'
NREL_CORPUS_PATH = DATA_PATH + 'NREL_corpus.jsonl'
NREL_NUMERICAL_PATH = DATA_PATH + 'NREL_numerical.csv'
STOPWORDS_PATH = DATA_PATH + 'stopwords.txt'
RELEVANCE_TRAIN_PATH = DATA_PATH + 'relevance.train.csv'
DOC_IDS_PATH = DATA_PATH + 'document-ids.txt'
SEARCH_RADIUS = 5
DEFAULT_LAT = "42.30136771768067"
DEFAULT_LNG = "-83.71907280246434"
DEFAULT_USER = 0
DEFAULT_PROMPT = None


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
        self.frame = pd.read_csv(NREL_NUMERICAL_PATH, low_memory=False)
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
            self.pipeline.ranker = self.ranker

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

    def get_results_all(self, lat, lng, prompt, user_id, radius=0.03, top_n=10):
        query = str(lat) + ", " + str(lng)
        if prompt:
            query = query + ", " + prompt
        param = {
            "user_id": user_id,
            "radius": radius
        }
        results = self.search(query, **param)
        return [r.docid for r in results]

    def get_station_info(self, docid_list, all=False):
        self.detailed_data = pd.read_csv(
            NREL_PATH, delimiter='\t', low_memory=False)
        print('Searching...')
        if docid_list[0] in self.detailed_data['id'].values:
            print('Using NREL data')
        if not all:
            return self.detailed_data[self.detailed_data['id'].isin(docid_list)][[
                'station_name', 'street_address', 'station_phone',
                'ev_network', 'latitude', 'longitude'
            ]]
        else:
            return self.detailed_data[self.detailed_data['ID'].isin(docid_list)][[
                'id', 'station_name', 'street_address', 'intersection_directions',
                'city', 'state_name', 'zip', 'plus4', 'station_phone',
                'access_days_time', 'cards_accepted', 'ev_level1_evse_num',
                'ev_level2_evse_num', 'ev_dc_fast_count', 'ev_other_info', 'ev_network',
                'ev_network_web', 'geocode_status', 'latitude', 'longitude',
                'date_last_confirmed', 'updated_at', 'owner_type_code',
                'federal_agency_id', 'federal_agency_name', 'open_date',
                'ev_connector_types', 'country', 'access_code', 'access_detail_code',
                'federal_agency_code', 'facility_type', 'ev_pricing',
                'ev_on_site_renewable_source', 'restricted_access', 'nps_unit_name',
                'maximum_vehicle_class', 'ev_workplace_charging'
            ]]

    def get_gpt_info(self, docid_list):
        self.detailed_data = pd.read_csv(
            '../data/station_personalized_features.csv')
        return self.detailed_data[self.detailed_data['docid'].isin(docid_list)]


def initialize():
    search_obj = SearchEngine(cf=False, l2r=False)
    return search_obj


def main():
    pass


if __name__ == "__main__":
    main()
