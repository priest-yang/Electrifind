import math
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm


# TODO (HW5): Implement NFaiRR
def nfairr_score(actual_omega_values: list[int], cut_off=200) -> float:
    """
    Computes the normalized fairness-aware rank retrieval (NFaiRR) score for a list of omega values
    for the list of ranked documents.
    If all documents are from the protected class, then the NFaiRR score is 0.

    Args:
        actual_omega_values: The omega value for a ranked list of documents
            The most relevant document is the first item in the list.
        cut_off: The rank cut-off to use for calculating NFaiRR
            Omega values in the list after this cut-off position are not used. The default is 200.

    Returns:
        The NFaiRR score
    """
    # TODO (HW5): Compute the FaiRR and IFaiRR scores using the given list of omega values
    # TODO (HW5): Implement NFaiRR
    actual_omega_values = actual_omega_values[:cut_off]
    ideal_values = sorted(actual_omega_values, reverse=True)
    FaiRR_list = []
    IFaiRR_list = []
    for i in range(cut_off):
        if i >= len(actual_omega_values):
            break
        if i == 0:
            FaiRR = actual_omega_values[i]
            IFaiRR = ideal_values[i]
        else:
            FaiRR = actual_omega_values[i] / (math.log2(i + 1))
            IFaiRR = ideal_values[i] / (math.log2(i + 1))
        FaiRR_list.append(FaiRR)
        IFaiRR_list.append(IFaiRR)
    if np.sum(FaiRR_list) == 0:
        return 0
    return np.sum(FaiRR_list) / np.sum(IFaiRR_list)


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


def run_relevance_tests(relevance_data_filename: str, ranker,
                        pseudofeedback_num_docs: int = 0,
                        pseudofeedback_alpha: float = 0.8,
                        pseudofeedback_beta: float = 0.2,
                        mmr_lambda: int = 1,
                        mmr_threshold: int = 100
                        ) -> dict[str, float]:
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
    for i in tqdm(range(len(queries))):
        query = queries[i]
        ideal = relevance_df[relevance_df['query'] == query]
        ideal = ideal.sort_values(by=['rel'], ascending=False)
        if ranker.__class__.__name__ == 'L2RRanker':
            response = ranker.query(
                query,
                pseudofeedback_num_docs,
                pseudofeedback_alpha,
                pseudofeedback_beta,
                mmr_lambda=mmr_lambda,
                mmr_threshold=mmr_threshold
            )
        else:
            response = ranker.query(query)

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). Use relevance
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: NDCG can use any scoring range, so no conversion is needed.
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

    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    map = np.mean(MAP_list)
    ndcg = np.mean(NDCG_list)
    print("MAP: ", MAP_list)
    print("NDCG: ", NDCG_list)

    return {'map': map, 'ndcg': ndcg, 'map_list': MAP_list, 'ndcg_list': NDCG_list}


# TODO (HW5): Implement NFaiRR metric for a list of queries to measure fairness for those queries
# NOTE: This has no relation to relevance scores and measures fairness of representation of classes
def run_fairness_test(attributes_file_path: str, protected_class: str, queries: list[str],
                      ranker, cut_off: int = 200) -> float:
    """
    Measures the fairness of the IR system using the NFaiRR metric.

    Args:
        attributes_file_path: The filename containing the documents about people and their demographic attributes
        protected_class: A specific protected class (e.g., Ethnicity, Gender)
        queries: A list containing queries
        ranker: A ranker configured with a particular scoring function to search through the document collection
        cut_off: The rank cut-off to use for calculating NFaiRR

    Returns:
        The average NFaiRR score across all queries
    """
    # TODO (HW5): Load person-attributes.csv
    attributes_df = pd.read_csv(attributes_file_path)

    # TODO (HW5): Find the documents associated with the protected class
    rel_docs = attributes_df[attributes_df[protected_class].isnull(
    ) == False]['docid'].values

    score = []

    # TODO (HW5): Loop through the queries and
    #       1. Create the list of omega values for the ranked list.
    #       2. Compute the NFaiRR score
    # NOTE: This fairness metric has some 'issues' (and the assignment spec asks you to think about it)
    for query in tqdm(queries):
        response = ranker.query(query)
        actual_omega_values = []
        for i in range(len(response)):
            docid = response[i][0]
            if docid in rel_docs:
                actual_omega_values.append(0)
            else:
                actual_omega_values.append(1)
        score.append(nfairr_score(actual_omega_values, cut_off))

    return sum(score)


if __name__ == '__main__':
    pass
