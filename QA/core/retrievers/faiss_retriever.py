import faiss
import numpy as np
from core.generators.bert_encoder import BertEncoder
import logging

logger = logging.getLogger(__name__)

class FaissRetriever:
    def __init__(self):
        self.index = None
        self.encoder = BertEncoder()
        self.documents = []
        self.dimension = 768

    def initialize(self, documents):
        """初始化Faiss索引"""
        self.documents = documents
        
        # 检查是否有数据
        if not documents:
            logger.warning("没有文档数据，跳过Faiss索引初始化")
            return
        
        # 生成文档嵌入
        texts = [doc.get('question', '') + ' ' + doc.get('answer', '') for doc in documents]
        embeddings = self.encoder.encode_texts(texts)
        
        # 检查embeddings是否为空
        if len(embeddings) == 0:
            logger.warning("embeddings为空，跳过Faiss索引初始化")
            return
        
        # 确保embeddings是numpy数组
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        # 创建Faiss索引
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 标准化嵌入向量
        faiss.normalize_L2(embeddings)
        
        # 添加到索引
        self.index.add(embeddings)
        
        logger.info(f"Faiss索引初始化: {len(documents)}个文档")

    def search(self, query, k=5):
        """搜索相关文档"""
        if self.index is None or not self.documents:
            logger.warning("索引未初始化或没有文档数据，返回空结果")
            return []

        query_embedding = self.encoder.encode_texts([query])
        
        # 确保query_embedding是numpy数组
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding)
        
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'question': doc.get('question', ''),
                    'answer': doc.get('answer', ''),
                    'similarity': float(score),
                    'rank': i + 1
                })
        
        return results 