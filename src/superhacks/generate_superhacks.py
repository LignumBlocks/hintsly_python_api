import os
import pickle
import json
import numpy as np
import random
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity

import utils.clustering as clustering
import utils.vector_store_management as vsm
import utils.core as core
import utils.base_llm as base_llm

SUPERHACK_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SUPERHACK_DIR)               
BASE_DIR = os.path.dirname(SRC_DIR)       
PROMPT_DIR = os.path.join(BASE_DIR, 'data', 'prompts') 
PROMPTS_TEMPLATES = {
    'CHECK_FEASIBILITY':os.path.join(PROMPT_DIR, "generic_superhack"),
    'CHECK_FEASIBILITY_SYSTEM':os.path.join(PROMPT_DIR, "generic_superhack_system"),
    'SUPERHACK_STRUCTURE':os.path.join(PROMPT_DIR, "superhack_fields"),
}
HACK_GROUPS = os.path.join(BASE_DIR, 'data', 'files', "hack_groups.bin")

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
        record_data['metadata']['id'] = str(int(record_data['metadata']['id']))
        metadata.append(record_data['metadata'])

    return np.array(vectors), ids, metadata

def annotate_clustering_results(vectors: list, ids: list, metadata: list, labels) -> Tuple[dict, dict]:
    """
    Annotates the clustering results with IDs and metadata.

    Args:
        vectors: The NumPy array of vectors.
        ids: The list of IDs.
        metadata: The list of metadata dictionaries.
        n_clusters: The number of clusters.

    Returns:
        A tuple of 2 dictionaries. The first one, for each cluster contains a list of the hacks in the cluster. 
        The second one, for each hack id contains the hack information and the cluster where it was located.
    """
    assert(len(labels) == len(ids))
    cluster_data = {}
    data_with_labels = {}
    for i, label in enumerate(labels):
        # print(metadata[i], ids[i])
        assert(ids[i] == metadata[i]['id'])
        if label not in cluster_data:
            cluster_data[label] = []
        cluster_data[label].append({ 'id': ids[i], 'metadata': metadata[i], 'vector': vectors[i]})
        data_with_labels[ids[i]] = { 'metadata': metadata[i], 'vector': vectors[i], 'cluster': label }
    return cluster_data, data_with_labels

def check_feasibility(hack_descriptions: str):
    prompt_template = base_llm.load_prompt(PROMPTS_TEMPLATES['CHECK_FEASIBILITY'])    
    prompt = prompt_template.format(group_of_hacks=hack_descriptions)
    system_prompt = base_llm.load_prompt(PROMPTS_TEMPLATES['CHECK_FEASIBILITY_SYSTEM'])
    model = base_llm.Base_LLM()
    retries = 0
    while retries < 3:
        try:
            result = model.run(prompt, system_prompt)
            cleaned_string = result.replace("```json\n", "").replace("```","")
            cleaned_string = cleaned_string.strip()
            json_result =  json.loads(cleaned_string)
            return json_result
        except (Exception) as e:  #Catch more general exceptions
            print(f"Error during LLM call or JSON parsing (attempt {retries+1}/{3}): {e}")
            print(cleaned_string)
            retries += 1

def superhack_structure(hack_descriptions: str, superhack_analysis: str):
    prompt_template = base_llm.load_prompt(PROMPTS_TEMPLATES['SUPERHACK_STRUCTURE'])    
    prompt = prompt_template.format(hack_descriptions=hack_descriptions, superhack_analysis=superhack_analysis)
    model = base_llm.Base_LLM()
    retries = 0
    while retries < 3:
        try:
            result = model.run(prompt)
            cleaned_string = result.replace("```json\n", "").replace("```","")
            cleaned_string = cleaned_string.strip()
            json_result =  json.loads(cleaned_string)
            return json_result
        except (Exception) as e:  #Catch more general exceptions
            print(f"Error during LLM call or JSON parsing (attempt {retries+1}/{3}): {e}")
            print(cleaned_string)
            retries += 1

def generate_superhack(metadata, candidates_ids):
    # https://python.langchain.com/docs/how_to/structured_output/
    # Filter the metadata list to include only the hacks with IDs in candidates_ids
    selected_hacks = [m for m in metadata if m['id'] in candidates_ids]
    hack_descriptions = ""
    for hack in selected_hacks:
        # Format each hack's information
        hack_str = (
            f"ID: {hack['id']}\n"
            f"Title: {hack['title']}\n"
            f"Description: {hack['description']}\n"
            f"Main Goal: {hack['main_goal']}\n"
            f"Resources Needed: {hack['resources_needed']}\n"
            f"Expected Benefits: {hack['expected_benefits']}\n"
            "---\n"
        )
        hack_descriptions += hack_str
    json_result = check_feasibility(hack_descriptions)
    # print(json_result)

    analysis: str = json_result["analysis"]
    superhack_feasible: bool = json_result["superhack_feasible"]
    combined_strategies: List[str] = json_result["combined_strategies"]
    explanation: str = json_result["explanation"]
    if superhack_feasible and len(combined_strategies) >= 2 :
        selected_hacks = [m for m in metadata if m['id'] in combined_strategies]
        hack_descriptions = ""
        for hack in selected_hacks:
            # Format each hack's information
            hack_str = (
                f"ID: {hack['id']}\n"
                f"Title: {hack['title']}\n"
                f"Resources Needed: {hack['resources_needed']}\n"
                f"Expected Benefits: {hack['expected_benefits']}\n"
                f"Steps to Implement: {hack['steps_summary']}\n"
                "---\n"
            )
            hack_descriptions += hack_str

        json_result = superhack_structure(hack_descriptions, explanation)
        # print(json_result)
        return True, json_result
    return False, None


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
                
        assert(len(filters) == 20)
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

    def save_item_to_pickle(item, filename):
        """Saves objects to a pickle file."""
        with open(filename, "wb") as f:
            pickle.dump(item, f)
            # print(f"Hack groups saved to {filename}")

    def get_hack_groups():
        hack_groups = None
        if os.path.exists(HACK_GROUPS):
            with open(HACK_GROUPS, "rb") as f:
                print("Loading hack groups from pickle file...") # Indicate loading
                hack_groups = pickle.load(f)
        if hack_groups is None:
            hack_groups = fetch_hack_groups(generate_filters())
            save_item_to_pickle(hack_groups, HACK_GROUPS)
        return hack_groups

    def select_high_similarity_hacks_from_cluster(cluster_of_hacks:list, similarity_threshold=0.5) -> List[List[str]]:
        """
        cluster_of_hacks: list [..., { 'id': ids[i], 'metadata': metadata[i], 'vector': vector}, ...]
        """
        embeddings = np.array([hack['vector'] for hack in cluster_of_hacks])

        print("Original shape of embeddings:", embeddings.shape)
        similarity_matrix = cosine_similarity(embeddings)

        all_groups = []
        all_group_ids = set() # Keep track of unique group combinations.

        for i, hack in enumerate(cluster_of_hacks):
            # select only the hacks with a high similarity to the current i hack
            hack_id = hack['id']
            similar_hacks_indices = np.where(similarity_matrix[i] >= similarity_threshold)[0]
            similar_hacks = [cluster_of_hacks[j]['id'] for j in similar_hacks_indices if j != i]

            num_similar = len(similar_hacks)
            if num_similar > 0:
                group_size = max(1, min(num_similar, int(np.random.normal(5, 1.5)))) # Random group size
                # group_size = min(num_similar, random.randint(1, min(9, num_similar))) # Random group size
                group_indices = random.sample(range(num_similar), group_size)
                group_ids = [hack_id] + [similar_hacks[j] for j in group_indices]

                group_ids_tuple = tuple(sorted(group_ids)) #Make hashable for set membership check.

                if group_ids_tuple not in all_group_ids:
                    all_groups.append(group_ids)
                    all_group_ids.add(group_ids_tuple)

        return all_groups
    
    hack_groups = get_hack_groups()
    hack_group_clusters = {}
    checked_groups = []
    
    for group_key, group_data in hack_groups.items():
        vectors, ids, metadata = prepare_data_for_clustering(group_data)
        labels = clustering.perform_best_clustering(vectors, '')
        clustered_data = annotate_clustering_results(vectors, ids, metadata, labels)
        hack_group_clusters[group_key] = clustered_data
        for label in clustered_data[0]:
            hacks_in_label = clustered_data[0][label]
            groups_for_superhacks = select_high_similarity_hacks_from_cluster(hacks_in_label)
            for t in groups_for_superhacks:
                checked_groups.append(t)
                is_superhack, super_hack_fields = generate_superhack(metadata, t)

def determine_best_clustering_and_hyperparameters():
    vs_manager = vsm.VS_Manager()
    hacks = vs_manager.get_all_data()
    vectors, ids, metadata = prepare_data_for_clustering(hacks)
    data = np.array(vectors)
    print(data.shape)
    # optimal_k = clustering.find_optimal_k_kmeans(data, max_k=10, log_dir=clustering.LOG_DIR)
    # print(f"Optimal k for KMeans: {optimal_k}")
    # kmeans_labels = clustering.perform_kmeans(data, optimal_k, process_name='best_cluster', log_dir=clustering.LOG_DIR)
    # # KMeans with PCA
    # optimal_k_pca = clustering.find_optimal_k_kmeans(data, max_k=10, pca_components=434, log_dir=clustering.LOG_DIR)
    # print(f"Optimal k for KMeans PCA: {optimal_k_pca}")
    # kmeans_labels_pca = clustering.perform_kmeans(data, optimal_k_pca, pca_components=434, process_name='best_cluster', log_dir=clustering.LOG_DIR)

    # eps_range = np.arange(0.1, 5, 0.3)  # Adjust the range as needed
    # min_samples_range = range(2, 7) # Adjust the range as needed
    # best_eps, best_min_samples, dbscan_labels = clustering.tune_dbscan_hyperparameters(data, eps_range, min_samples_range, log_dir=clustering.LOG_DIR)
    # print(f"Best eps for DBSCAN: {best_eps}, Best min_samples: {best_min_samples}")
    # dbscan_labels = clustering.perform_dbscan(data, best_eps, best_min_samples, process_name='best_cluster', log_dir=clustering.LOG_DIR)

    # # DBSCAN with PCA
    # best_eps_pca, best_min_samples_pca, dbscan_labels_pca = clustering.tune_dbscan_hyperparameters(data, eps_range, min_samples_range, pca_components=434, log_dir=clustering.LOG_DIR)
    # print(f"Best eps for DBSCAN (with PCA): {best_eps_pca}, Best min_samples: {best_min_samples_pca}")
    dbscan_labels_pca = clustering.perform_dbscan(data, 0.7, 6, pca_components=434, process_name='best_cluster', log_dir=clustering.LOG_DIR)
    unique_labels = set(dbscan_labels_pca)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Count elements in each cluster
    cluster_counts = {}
    for label in unique_labels:
        if label != -1:  # Exclude noise
            cluster_counts[label] = list(dbscan_labels_pca).count(label)

    # Print results
    print(f"Estimated number of clusters: {n_clusters}")
    print("Number of elements in each cluster:")
    for label, count in cluster_counts.items():
        print(f"Cluster {label}: {count} elements")