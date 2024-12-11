import numpy as np
from typing import List, Dict, Tuple

import utils.clustering as clustering
import utils.vector_store_management as vsm
import utils.core as core

def prepare_data_for_clustering(data_from_pinecone: Dict[str, Dict[str, dict|list]]) -> Tuple[np.ndarray, list, list]:
    """
    Prepares Pinecone data for scikit-learn clustering.

    Args:
        data_from_pinecone: The dictionary returned by get_all_data.

    Returns:
        A tuple containing:
            - vectors: A NumPy array of vectors.
            - ids: A list of IDs corresponding to the vectors.
            - metadata: A list of metadata dictionaries corresponding to the vectors.
    """
    ids = []
    vectors = []
    metadata = []

    for record_id, record_data in data_from_pinecone.items():
        ids.append(record_id)
        vectors.append(record_data['vector'])
        metadata.append(record_data['metadata'])

    return np.array(vectors), ids, metadata

def annotate_clustering_results(ids: list, metadata: list, labels) -> list:
    """
    Annotates the clustering results with IDs and metadata.

    Args:
        vectors: The NumPy array of vectors.
        ids: The list of IDs.
        metadata: The list of metadata dictionaries.
        n_clusters: The number of clusters.

    Returns:
        A list of dictionaries, where each dictionary contains 'id', 'metadata', and 'cluster'.
    """
    assert(len(labels) == len(ids))
    clustered_data = []
    for i, label in enumerate(labels):
        assert(ids[i] == metadata[i]['id'])
        clustered_data.append({
            'id': ids[i],
            'metadata': metadata[i],
            'cluster': label
        })
    return clustered_data

def pipeline1():
    """
    To ensure coherence and relevance it is proposed to group hacks based on the predefined tags. 
    The grouping process will follow a prioritized hierarchy:

    1. Group by 'Financial Goals'. We would get 4 groups with hacks appearing in at least one and possibly various.

    2. For each of those groups I would like to determine 5 subgroups, one for each 'Audience and Life Stage'. 

    In the end we would get 20 groups, each containing all the hacks that target the same goal and audience. 
    Under the hypothesis that this will cover relatedness, we must search for clusters of sizes from 2-10 (with a normal distribution centered around 4 or 5).
    
    For all the acceptable clusters inside of the 20 groups we must test them with LLMs to validate or refute the possibility of a SuperHack.
    Then, in a global (shared by the 20 groups) place store the groups of hacks that were tested together, regardless if they could form a SuperHack or not. 
    So afterwards we can check first if a set of hacks has already been tested, to not repeat the SuperHack check.

    To create the 20 groups we wil use the function VS_Manager.get_by_filter(filter) ; where the filter is an and including a category from 'Audience and Life Stage' and one from 'Financial Goals'.
    The filters must look like: `filter = {"$and": [{"Financial Goals": "Debt Reduction"}, {"Audience and Life Stage": "Families"} ]}`
    ```
    """
    def generate_filters() -> List[Dict]:
        """Generates all filter combinations for Financial Goals and Audience and Life Stage."""
        categories = core.HACKS_CLASSIFICATIONS_CATEGORIES
        financial_goals = [goal for goal in categories["Financial Goals"]["categories"] if goal != categories["Financial Goals"]["exclude"]]
        audience_stages = categories["Audience and Life Stage"]["categories"]

        filters = []
        for goal in financial_goals:
            for audience in audience_stages:
                filter = {"$and": [{"Financial Goals": goal}, {"Audience and Life Stage": audience}]}
                filters.append(filter)

        assert(len(filter) == 20)
        return filters
    
    def fetch_hack_groups(filters: List[Dict]) -> Dict[str, Dict]:
        """Fetches hack groups using the provided filters."""
        hack_groups = {}
        for filter in filters:
            goal = filter["$and"][0]["Financial Goals"]
            audience = filter["$and"][1]["Audience and Life Stage"]
            group_key = f"{goal}-{audience}" # Create a unique key for each group
            vs_manager = vsm.VS_Manager()
            hacks = vs_manager.get_by_filter(filter) # Fetch hacks using the filter
            hack_groups[group_key] = hacks
        return hack_groups

    hack_groups = fetch_hack_groups(generate_filters())
    hack_group_clusters = {}
    for group_key, group_data in hack_groups.items():
        vectors, ids, metadata = prepare_data_for_clustering(group_data)
        labels = clustering.perform_best_clustering(vectors, '')
        clustered_data = annotate_clustering_results(ids, metadata, labels)
        hack_group_clusters[group_key] = clustered_data
