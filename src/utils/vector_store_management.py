
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Tuple
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()
FREE_GOOGLE_API_KEY = os.getenv('FREE_GOOGLE_API_KEY')
EMBEDDING_MODELS = ['models/embedding-001', 'models/text-embedding-004']
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  
INDEX_NAME = os.getenv("PINECONE_INDEX")
UPPER_LIMIT_NO_EMBED = 500

class VS_Manager:
    """query by metadata of list of strings: 

    { "genre": ["comedy", "documentary"] }

    {"genre":"comedy"}

    {"genre": {"$in":["documentary","action"]}}

    {"$and": [{"genre": "comedy"}, {"genre":"documentary"}]}
    """
    def __init__(self):
        pc = Pinecone(api_key=PINECONE_API_KEY)  
        self.index = pc.Index(INDEX_NAME)  
        
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", 
                                                task_type="clustering", 
                                                google_api_key=FREE_GOOGLE_API_KEY)
        self.vector_store = PineconeVectorStore(pinecone_api_key=PINECONE_API_KEY, 
                                                index=INDEX_NAME, embedding=self.embeddings) #, text_key="summary")

    def add_documents(self, documents):
        self.vector_store.add_documents(documents)
    def add_documents_ids(self, documents, ids):
        self.vector_store.add_documents(documents=documents, ids=ids)
    def remove_documents(self, ids: List):
        self.vector_store.delete(ids)
    
    def get_all_ids(self) -> List[str]:
        """Retrieves all IDs from the index, handling pagination."""
        all_ids = []
        for ids in self.index.list():
            for id in ids:
                all_ids.append(id)
        # print(len(all_ids))
        return all_ids
    def get_by_ids(self, ids:List[str]) -> Dict[str, Dict[str, dict|list]]:
        response = self.index.fetch(ids)
        result = {}
        if 'vectors' in response:  # Check if vectors are present in response
            for record_id, record_data in response['vectors'].items():
                metadata:Dict = record_data['metadata']
                result[record_id] = {'metadata': metadata, 'vector':record_data['values']}
        return result
    def get_all_data(self) -> Dict[str, Dict[str, dict|list]]:
        response = self.index.fetch(self.get_all_ids())
        result = {}
        if 'vectors' in response:  # Check if vectors are present in response
            for record_id, record_data in response['vectors'].items():
                metadata = record_data['metadata']
                result[record_id] = {'metadata': metadata, 'vector':record_data['values']}
        return result

    def get_by_filter(self, filter:Dict[str, str | bool | List] | None = None):
        response = self.index.query(vector=[0]*768, filter=filter,
                                    top_k=UPPER_LIMIT_NO_EMBED,
                                    include_metadata=True)
        matching_ids = [match['id'] for match in response['matches']]
        print(f"ids matching filter: {filter}\n\t{matching_ids}")
        if matching_ids:  # Check if there are IDs to fetch
            return self.get_by_ids(matching_ids)
        else:
            return {} # Return empty if no matches
    
    def retrieve_k(self, query, k=5, filter=None) -> List[Document]:
        """Query embeddings from the Pinecone index."""
        return self.vector_store.similarity_search(query, k=k, filter=filter)
    def retrieve_k_score(self, query, k=5, filter=None) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
    
vs = VS_Manager()
print(vs.get_by_filter({'Financial Goals':{"$in":['Wealth Building','Income Generation']}}))