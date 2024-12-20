import os
import requests
import time
from typing import List, Dict
import utils.core as core

# ================== Configuration ==================
# Base URL of the API
BASE_URL = "https://clawed-frog.hintsly-dashboard-2.c66.me/api/v1"
# Cookie
COOKIE = "_rails_ralix_tailwind_session=tkcF0EWxD4AAk2sXevvtEQZII0AM%2By9gRy9oJ1oHXUpwLtAoJ1V4RUB3gUWvtU0icBklehDqYDFDomxvvolBqiPLSVw%2FVnkwis0QJFP87anWPncbijKR2jP7zXhwVzUHeMrIFL%2FG%2BqYrk1tMepKxn%2BlrcPdwifazoAqVK2tZrqC7fV38fzHsZrbGt0N6MhLN5twzdkkqfWxbgudnaTXKuyxTMj1H5jUCVMBwZwODCjwSAtLSY8ASpzqmEE2MLXXI9%2FOpy0di8%2BSqOtOxEAj3cgavNDHK8k3Z5eTiVhUmcgo35a7hwUO5H%2FV3HvwKVxcH3Wqh%2BRpMNLhBe%2B78DHLa3MmndzA9k4W0Iw%2BYu3QgUEnpkUCBXw%3D%3D--jziZJAk2UPE5Sp3o--sDwlqPpW3PqqTfdKbNzt7Q%3D%3D"
# Endpoint paths
CUSTOM_HACKS_ENDPOINT = "/custom-hacks"
MARK_SENT_ENDPOINT = "/hacks-sent-to-python"
# SYNCHRONIZE_HACKS = "/hacks/synchronize"
SHOW_HACKS = "/hacks"
GET_SUPERHACKS_CATEGORIES_ENDPOINT = "/superhacks/categories"
CREATE_SUPERHACKS_ENDPOINT = "/superhacks/create"
HACKS_FOR_SUPERHACKS_ENDPOINT = "/superhacks/combined-hacks"
SAVE_HACKS_COMB_ENDPOINT = "/superhacks/save-combined-hacks"
# Headers
HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": "mykey"
}
# Pagination settings
PER_PAGE = 10
# ====================================================

def get_superhack_categories():
    url = f"{BASE_URL}{GET_SUPERHACKS_CATEGORIES_ENDPOINT}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def create_superhack(data):
    """
    body = {
        "title": "sh",
        "description": "qrfqwrfqwfr",
        "implementation_steps": "qfqwefqwef",
        "expected_results": "qwefqwfqwef",
        "risks_and_mitigation": "fwqefqwef",
        "hack_ids": [
            5,
            6
        ],
        "category_ids": [
            40,
            41
        ]
    }
    """
    url = f"{BASE_URL}{CREATE_SUPERHACKS_ENDPOINT}"
    response = requests.post(url, headers=HEADERS, json=data)
    response.raise_for_status()
    return response.json()

def get_combined_hacks():
    url = f"{BASE_URL}{HACKS_FOR_SUPERHACKS_ENDPOINT}"
    headers = HEADERS.copy()
    headers['Cookie'] = COOKIE
    response = requests.get(url, headers=headers)
    print(response)
    response.raise_for_status()
    return response.json()

def save_combined_hacks(data):
    """
    body =  {
                "data": {"mr": "kjwefiwwefwefwefwefe"}
            }
    """
    url = f"{BASE_URL}{SAVE_HACKS_COMB_ENDPOINT}"
    headers = HEADERS.copy()
    headers['Cookie'] = COOKIE
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()

# def synchronize_hacks():
#     url = f"{BASE_URL}{SYNCHRONIZE_HACKS}"
#     response = requests.post(url, headers=HEADERS)
#     response.raise_for_status()
#     return response.json()

def get_hack_by_id(hack_id):
    url = f"{BASE_URL}{SHOW_HACKS}/{hack_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def get_custom_hacks(per_page=PER_PAGE, page=1):
    url = f"{BASE_URL}{CUSTOM_HACKS_ENDPOINT}"
    params = { "per_page": per_page, "page": page }
    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()
    return response.json()

def mark_hacks_as_sent(hack_ids: List[int]):
    """
    body = {
        "hack_ids": []
    }
    """
    url = f"{BASE_URL}{MARK_SENT_ENDPOINT}"
    payload = {"hack_ids": hack_ids}
    response = requests.post(url, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

def fetch_all_hacks(per_page:int=PER_PAGE) -> List[Dict]:
    """
    Fetch all hacks from the custom-hacks endpoint, handling pagination.

    :param per_page: Number of hacks to fetch per page.
    :return: A list of hack dictionaries.
    """
    hacks = []
    page = 1

    while True:
        try:            
            print(f"Fetching page {page}...")
            data = get_custom_hacks(per_page=per_page, page=page)
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
    print(get_superhack_categories())
    # # Step 1: Fetch all hacks
    # hacks = fetch_all_hacks(3)

    # if not hacks:
    #     print("No hacks to process.")
    #     return
    
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