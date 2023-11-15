import math
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm


def map_score(search_result_relevances: list[int], cut_off=10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # TODO: Implement MAP
    sum_precision = 0
    num_retrieved = 0

    total = min(cut_off, len(search_result_relevances))

    if len(search_result_relevances) == 0:
        return 0

    for i in range(total):
        if search_result_relevances[i] == 1:
            num_retrieved += 1
            sum_precision += num_retrieved / (i + 1)

    if num_retrieved == 0:
        return 0
    
    return sum_precision / total


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off=10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
            
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # TODO: Implement NDCG
    dcg_list = []
    ideal_dcg_list = []

    if len(search_result_relevances) == 0:
        return 0

    for i in range(cut_off):
        if i >= len(search_result_relevances) or i >= len(ideal_relevance_score_ordering):
            break
        if i == 0:
            dcg = search_result_relevances[i]
        else:
            dcg = search_result_relevances[i] / np.log2(i + 1)
        dcg_list.append(dcg)

    for i in range(cut_off):
        if i >= len(ideal_relevance_score_ordering):
            break
        if i == 0:
            ideal_dcg = ideal_relevance_score_ordering[i]
        else:
            ideal_dcg = ideal_relevance_score_ordering[i] / np.log2(i + 1)
        ideal_dcg_list.append(ideal_dcg)

    return np.sum(dcg_list) / np.sum(ideal_dcg_list)


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded

        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset
    relevance_df = pd.read_csv(relevance_data_filename)
    queries = relevance_df['query'].unique()
    MAP_list = []
    NDCG_list = []

    # TODO: Run each of the dataset's queries through your ranking function
    print("Running relevance tests...")
    for i in tqdm(range(len(queries))):
        query = queries[i]
        ideal = relevance_df[relevance_df['query'] == query]
        ideal = ideal.sort_values(by=['rel'], ascending=False)
        response = ranker.query(query)
        relevances_map = []
        relevances_ndcg = []
        for j in range(len(response)):
            docid = response[j][0]
            if docid in ideal['docid'].values:
                rel = ideal[ideal['docid'] == docid].rel.to_numpy()[0]
                relevances_map.append(1 if rel >= 4 else 0)
                relevances_ndcg.append(rel)
            else:
                relevances_map.append(0)
                relevances_ndcg.append(0)
        map = map_score(relevances_map)
        ndcg = ndcg_score(relevances_ndcg, ideal['rel'].values)
        MAP_list.append(map)
        NDCG_list.append(ndcg)

    map = np.mean(MAP_list)
    ndcg = np.mean(NDCG_list)
    print("MAP: ", MAP_list)
    print("NDCG: ", NDCG_list)

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance 
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
  
    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    return {'map': map, 'ndcg': ndcg}

