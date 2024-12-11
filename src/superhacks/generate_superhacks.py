import numpy as np
from typing import List, Dict, Tuple

import utils.clustering as clustering
import utils.vector_store_management as vsm

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


def cluster_and_annotate(vectors: np.ndarray, ids: list, metadata: list, n_clusters: int) -> list:
    """
    Performs clustering and annotates the results with IDs and metadata.

    Args:
        vectors: The NumPy array of vectors.
        ids: The list of IDs.
        metadata: The list of metadata dictionaries.
        n_clusters: The number of clusters.

    Returns:
        A list of dictionaries, where each dictionary contains 'id', 'metadata', and 'cluster'.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0) #Example using KMeans, change as needed
    labels = kmeans.fit_predict(vectors)

    clustered_data = []
    for i, label in enumerate(labels):
        clustered_data.append({
            'id': ids[i],
            'metadata': metadata[i],
            'cluster': label
        })

    return clustered_data

#Example Usage:
pinecone_data = get_all_data() #Get the data from your function
vectors, ids, metadata = prepare_data_for_clustering(pinecone_data)

#Perform clustering
clustered_results = cluster_and_annotate(vectors, ids, metadata, n_clusters=5)  # Adjust n_clusters as needed

#Now you can process clustered_results.  It's a list of dictionaries, and you can easily group by cluster
for cluster_label in range(5): #Replace 5 with your actual number of clusters.
    cluster_items = [item for item in clustered_results if item['cluster'] == cluster_label]
    print(f"Cluster {cluster_label}:")
    for item in cluster_items:
        print(f"  ID: {item['id']}, Metadata: {item['metadata']}")