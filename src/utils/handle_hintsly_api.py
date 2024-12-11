import os
import requests
import time
from typing import List, Dict
import utils.core as core

# ================== Configuration ==================
# Base URL of the API
BASE_URL = "https://clawed-frog.hintsly-dashboard-2.c66.me/api/v1"
# Endpoint paths
CUSTOM_HACKS_ENDPOINT = "/custom-hacks"
MARK_SENT_ENDPOINT = "/hacks-sent-to-python"
# Headers
HEADERS = {
    "Content-Type": "application/json",
    # "Authorization": f"Bearer {os.getenv('HINTSLY_API_AUTH_TOKEN')}",
    "x-api-key": "mykey"
}
# Pagination settings
PER_PAGE = 10
# ====================================================

def fetch_all_hacks(per_page: int = PER_PAGE) -> List[Dict]:
    """
    Fetch all hacks from the custom-hacks endpoint, handling pagination.

    :param per_page: Number of hacks to fetch per page.
    :return: A list of hack dictionaries.
    """
    hacks = []
    page = 1

    while True:
        params = {
            "per_page": per_page,
            "page": page
        }
        url = f"{BASE_URL}{CUSTOM_HACKS_ENDPOINT}"
        print(f"Fetching page {page}...")

        try:
            response = requests.get(url, headers=HEADERS, params=params)
            response.raise_for_status()  # Raise an error for bad status codes
            data = response.json()
            # data is a dictionary with the following structure:
            # data = {
            # 'pagination': {'page': 1, 'items': 10, 'count': 435, 'pages': 44, 'next': 2, 'prev': None},
            # 'hacks': [{hacks_fields}],
            # 'hack_categories': { 
            #     id1: [[cat1, class], [cat2, class], ...], 
            #     id2: [[cat1, class], [cat2, class], ...], 
            #     ... 
            #    }
            # }
            
            current_page_hacks = data.get('hacks', [])
            current_page_hacks_cat = data.get('hack_categories', {})
            processed_categories = preprocess_hack_categories(current_page_hacks_cat, core.HACKS_CLASSIFICATIONS_CATEGORIES)
            if not current_page_hacks:
                print("No more hacks found.")
                break
            for hack in current_page_hacks:
                hacks.append({
                    'id':hack['id'],                              # metadata_only
                    'is_advice':hack['is_advice'],                # metadata_only
                    'title':hack['free_title'], 
                    'summary':hack['summary'],                    # metadata_only
                    'description':hack['description'],
                    'main_goal':hack['main_goal'],
                    'resources_needed':hack['resources_needed'],
                    'expected_benefits':hack['expected_benefits'],
                    'steps_summary':hack['steps_summary'],        # metadata_only
                    **processed_categories.get(str(hack['id']), {})
                    })
            print(f"Fetched {len(current_page_hacks)} hacks from page {page}.")

            # Check if we've fetched all pages
            total_pages = data['pagination']['pages']
            if page >= total_pages:
                print("All hacks have been fetched.")
                break

            page += 1

        except requests.exceptions.RequestException as e:
            print(f"An error occurred while fetching hacks: {e}")
            break

    return hacks

def mark_hacks_as_sent(hack_ids: List[int]) -> bool:
    """
    Mark the given hacks as sent to Python by sending their IDs to the endpoint.

    :param hack_ids: A list of hack IDs to mark as sent.
    :return: True if marking was successful, False otherwise.
    """
    url = f"{BASE_URL}{MARK_SENT_ENDPOINT}"
    payload = {
        "hack_ids": hack_ids
    }

    try:
        response = requests.post(url, headers=HEADERS, json=payload)
        response.raise_for_status()
        print(f"Successfully marked {len(hack_ids)} hacks as sent.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while marking hacks as sent: {e}")
        return False

def preprocess_hack_categories(current_page_hacks_cat, classifications_dict):
    """
    Preprocess hack categories to create a more polished version of the category information for each hack.

    :param current_page_hacks_cat: A dictionary of hack categories.
    :param classifications_dict: A dictionary of classifications and their corresponding categories.
    :return: A dictionary of with the preprocessed category information for each hack.
    """
    preprocessed_hacks = {}
    for hack_id, categories_list in current_page_hacks_cat.items():
        classification = {}
        # print(categories_list)
        for cat_class in categories_list:
            classification_name = cat_class[1]
            category = cat_class[0]
            if classification_name in classifications_dict:
                excluding_cat = classifications_dict[classification_name].get("exclude", '') == category
                if excluding_cat: continue
                if classification.get(classification_name):
                    classification[classification_name].append(category)
                    # classification[classification_name] += "," + category
                else:
                    classification[classification_name] = [category]
                    # classification[classification_name] = f"{category}"
        preprocessed_hacks[hack_id] = classification
    return preprocessed_hacks

def main():
    # Step 1: Fetch all hacks
    hacks = fetch_all_hacks(3)

    if not hacks:
        print("No hacks to process.")
        return
    
    # # Step 2: Process the hacks
    # processed_ids = process_hacks(hacks)

    # if not processed_ids:
    #     print("No hacks were processed.")
    #     return
    # print(processed_ids)
    # # Step 3: Mark hacks as sent
    # success = mark_hacks_as_sent(processed_ids)

    # if success:
    #     print("All processed hacks have been marked as sent.")
    # else:
    #     print("Failed to mark some hacks as sent.")

# main()