'''
Author: Zim Gong
This file is a template code file for the Search Engine. 
'''
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_ranker import VectorRanker
from l2r import L2RRanker, L2RFeatureExtractor
from cf import CFRanker
from ranker import *
from indexing import IndexType, Indexer
from document_preprocessor import *
from models import BaseSearchEngine, SearchResponse
from utils import CACHE_PATH, DATA_PATH

NREL_PATH = DATA_PATH + 'NREL_raw.csv'
REVIEW_PATH = DATA_PATH + 'Google_Map_review_data_AA_DTW.csv'
NREL_CORPUS_PATH = DATA_PATH + 'NREL_corpus.jsonl'
NREL_NUMERICAL_PATH = DATA_PATH + 'NREL_numerical.csv'
STOPWORDS_PATH = DATA_PATH + 'stopwords.txt'
DOC_IDS_PATH = DATA_PATH + 'document-ids.txt'
SEARCH_RADIUS = 0.03
DEFAULT_LAT = "42.30136771768067"
DEFAULT_LNG = "-83.71907280246434"
DEFAULT_USER = 2
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
        elif reranker == 'vector+cf':
            print('Loading vector ranker...')
            self.ranker = VectorRanker(
                index=self.frame,
                ranker=self.ranker,
                stations_path=DATA_PATH + 'station_personalized_features.csv',
                users_path=DATA_PATH + 'user_profile.csv'
            )
            self.reranker = 'vector+cf'
            print('Loading cf ranker...')
            self.pipeline = CFRanker(self.frame, self.ranker)
        elif reranker == 'vector':
            print('Loading vector ranker...')
            self.pipeline = VectorRanker(
                index=self.frame,
                ranker=self.ranker,
                stations_path=DATA_PATH + 'station_personalized_features.csv',
                users_path=DATA_PATH + 'user_profile.csv'
            )
            self.reranker = 'vector'
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
        return [SearchResponse(id=idx+1, docid=result[0], score=result[1]) for idx, result in enumerate(results)]

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
        self.review_data = pd.read_csv(REVIEW_PATH, low_memory=False)
        if not all:
            res = []
            for docid in docid_list:
                row = self.detailed_data[self.detailed_data['id'] == docid]
                if row.empty:
                    continue
                mask = (abs(row['latitude'].values[0] - self.review_data.lat) <
                        0.001) & (abs(row['longitude'].values[0] - self.review_data.lng) < 0.001)
                reviews_data = self.review_data[mask]
                if reviews_data.empty:
                    continue
                reviews = []
                for index, line in reviews_data.iterrows():
                    reviews.append({
                        'rating': line['rating'],
                        'text': line['text']
                    })
                res.append({
                    'id': docid,
                    'station_name': row['station_name'].values[0],
                    'street_address': row['street_address'].values[0],
                    'station_phone': row['station_phone'].values[0],
                    'ev_network': row['ev_network'].values[0],
                    'latitude': row['latitude'].values[0],
                    'longitude': row['longitude'].values[0],
                    'ev_connector_types': row['ev_connector_types'].values[0],
                    'ev_pricing': row['ev_pricing'].values[0],
                    'access_days_time': row['access_days_time'].values[0],
                    'cards_accepted': row['cards_accepted'].values[0],
                    'reviews': reviews
                })
            return res
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


def initialize(ranker='dist', reranker=None):
    search_obj = SearchEngine(ranker='dist', reranker='vector')
    return search_obj


def main():
    search_obj = initialize()
    print(search_obj.get_results_all(DEFAULT_LAT, DEFAULT_LNG,
          DEFAULT_PROMPT, DEFAULT_USER, SEARCH_RADIUS))
    search_obj.set_reranker('vector')
    print(search_obj.get_results_all(DEFAULT_LAT, DEFAULT_LNG,
          DEFAULT_PROMPT, DEFAULT_USER, SEARCH_RADIUS))


if __name__ == "__main__":
    main()
