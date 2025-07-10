from elasticsearch import Elasticsearch
from config.config import ES_HOST, ES_PORT, ES_INDEX_NAME, ES_USERNAME, ES_PASSWORD
import logging
import os
from glob import glob
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ElasticSearchRetriever:
    def __init__(self):
        self.es = Elasticsearch(
            [f"http://{ES_HOST}:{ES_PORT}"],
            basic_auth=(ES_USERNAME, ES_PASSWORD) if ES_USERNAME and ES_PASSWORD else None
        )
        self.index_name = ES_INDEX_NAME

    def create_index(self):
        """
        创建ES索引
        """
        mapping = {
            "mappings": {
                "properties": {
                    "question": {"type": "text", "analyzer": "standard"},
                    "answer": {"type": "text", "analyzer": "standard"},
                    "department": {"type": "keyword"},
                    "source": {"type": "keyword"}
                }
            }
        }

        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name, body=mapping)

    def index_qa_pairs(self, qa_pairs):
        for qa in qa_pairs:
            doc = {
                "question": qa['question'],
                "answer": qa['answer'],
                "department": qa.get('department', ''),
                "source": qa.get('source', '')
            }
            self.es.index(index=self.index_name, body=doc)

        self.es.indices.refresh(index=self.index_name)

    def search(self, query, k=5):
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^2", "answer"],
                    "type": "best_fields"
                }
            },
            "size": k
        }
        response = self.es.search(index=self.index_name, body=search_body)

        results = []
        max_score = response["hits"]["max_score"] if response["hits"]["max_score"] else 1.0
        for hit in response["hits"]["hits"]:
            results.append({
                "question": hit["_source"]["question"],
                "answer": hit["_source"]["answer"],
                "department": hit["_source"].get("department", ""),
                "score": hit["_score"],
                "normalized_score": hit["_score"] / max_score
            })

        return results

    def delete_index(self):
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name) 