"""
Given a user profile or a financial goal information:
- find related hacks that in conjunction can work for that goal
- validate if those hacks form a Superhack

To find the hacks that will conform a SuperHack
"""

import os
import json
from dotenv import load_dotenv
from typing import List, Dict
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_pinecone import PineconeVectorStore

import handle_hintsly_api as hapi

# Load environment variables from .env file
load_dotenv()
FREE_GOOGLE_API_KEY = os.getenv('FREE_GOOGLE_API_KEY')
EMBEDDING_MODELS = ['models/embedding-001', 'models/text-embedding-004']
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Replace with your Pinecone API key or store it in an env variable
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # Replace with your Pinecone environment
INDEX_NAME = os.getenv("PINECONE_INDEX")
NAMESPACE = 'superhacks-hacks'

def store_hacks_embeddings(hacks: List[Dict]):
    """
    Embeds each hack and stores the embeddings in Pinecone.

    Args:
        hacks (List[Dict]): List of hack dictionaries.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", 
                                            task_type="clustering", 
                                            google_api_key=FREE_GOOGLE_API_KEY)
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    documents = []
    ids = []
    stored_hacks = []
    for hack in hacks:
        document = preprocess_hack(hack)
        documents.append(document)
        ids.append(str(hack['id']))
        stored_hacks.append(int(hack['id']))
    # print(ids)
    vector_store.add_documents(documents=documents, ids=ids)
    return stored_hacks


def preprocess_hack(hack: Dict) -> Document:
    """
    Preprocess a hack to create a representative string.

    Args:
        hack (Dict): The hack dictionary.

    Returns:
        Document: Langchain Document with the representative string of the hack and the dictionary of metadata.
    """
    representative_string = f"{hack['title']}\n\nDescription: \n{hack['description']}\n\nMain Goal: \n{hack['main_goal']}\n\n" + \
                            f"Resources Needed: \n{hack['resources_needed']}\n\nExpected Benefits: \n{hack['expected_benefits']}"
    metadata = hack
    return Document(page_content=representative_string, metadata=metadata)

def main():
    
    hacks = hapi.fetch_all_hacks(10)
    hacks_ids = store_hacks_embeddings(hacks)
    print(hapi.mark_hacks_as_sent(hacks_ids))

# main()