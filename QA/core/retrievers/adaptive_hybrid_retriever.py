import logging
from typing import Dict, List, Any, Optional
from .es_retriever import ElasticSearchRetriever
from .faiss_retriever import FaissRetriever
from core.generators.bert_encoder import BertEncoder
from core.detectors.knowledge_conflict_detector import KnowledgeConflictDetector
from core.dialogue.query_analyzer import QueryAnalyzer
import numpy as np

logger = logging.getLogger(__name__)

class AdaptiveHybridRetriever:
    def __init__(self):
        self.es_retriever = ElasticSearchRetriever()
        self.faiss_retriever = FaissRetriever()
        self.bert_encoder = BertEncoder()
        self.conflict_detector = KnowledgeConflictDetector()
        self.query_analyzer = QueryAnalyzer()
        self.documents = []
        
        logger.info("自适应检索器初始化完成")

    def initialize(self, documents: List[Dict[str, Any]]):
        """初始化检索器"""
        try:
            self.documents = documents
            logger.info(f"开始初始化检索器，文档数量: {len(documents)}")
            
            # 初始化Faiss检索器
            logger.info("初始化Faiss检索器...")
            self.faiss_retriever.initialize(documents)
            
            # 初始化ES检索器
            try:
                logger.info("初始化ElasticSearch检索器...")
                self.es_retriever.create_index()
                self.es_retriever.index_qa_pairs(documents)
                logger.info("ES检索器初始化")
            except Exception as e:
                logger.warning(f"ES检索器初始化失败: {e}，将只使用Faiss检索")
            
            logger.info("检索器初始化")
            
        except Exception as e:
            logger.error(f"检索器初始化失败: {e}")
            raise

    def search(self, query: str, top_k: int = 5, context: Optional[Dict] = None) -> Dict[str, Any]:
        """执行自适应混合检索"""
        try:
            # 查询分析
            query_analysis = self.query_analyzer.analyze(query, context)
            
            # 根据查询类型调整检索权重
            weights = self._calculate_weights(query_analysis)
            
            # 执行多种检索
            results = {}
            
            # Faiss检索
            logger.info("执行向量检索...")
            faiss_results = self.faiss_retriever.search(query, top_k * 2)
            results['faiss'] = faiss_results
            
            # ES检索（如果可用）
            try:
                logger.info("执行文本检索...")
                es_results = self.es_retriever.search(query, top_k * 2)
                results['es'] = es_results
            except Exception as e:
                logger.warning(f"ES检索失败: {e}")
                results['es'] = []
            
            # 混合结果
            hybrid_results = self._merge_results(results, weights, top_k)
            
            # 知识冲突检测
            conflict_analysis = self.conflict_detector.detect_conflicts(
                hybrid_results[:top_k], query
            )
            
            return {
                'results': hybrid_results[:top_k],
                'query_analysis': query_analysis,
                'conflict_analysis': conflict_analysis,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return {
                'results': [],
                'query_analysis': {},
                'conflict_analysis': {},
                'weights': {}
            }

    def _calculate_weights(self, query_analysis: Dict[str, Any]) -> Dict[str, float]:
        """根据查询分析结果计算检索权重"""
        query_type = query_analysis.get('query_type', 'general')
        confidence = query_analysis.get('confidence', 0.5)
        
        # 默认权重，自己配置
        weights = {
            'faiss': 0.6,
            'es': 0.4
        }
        
        # 根据查询类型调整权重
        if query_type == 'factual':
            weights['faiss'] = 0.7  # 事实查询更依赖语义匹配
            weights['es'] = 0.3
        elif query_type == 'symptom':
            weights['faiss'] = 0.5
            weights['es'] = 0.5  # 症状查询平衡使用
        elif query_type == 'treatment':
            weights['faiss'] = 0.8  # 治疗查询更依赖精确匹配
            weights['es'] = 0.2
        
        # 根据置信度微调
        if confidence < 0.3:
            # 低置信度时增加ES权重
            weights['faiss'] *= 0.9
            weights['es'] *= 1.1
        
        # 归一化权重
        total = weights['faiss'] + weights['es']
        weights['faiss'] /= total
        weights['es'] /= total
        
        return weights

    def _merge_results(self, results: Dict[str, List], weights: Dict[str, float], top_k: int) -> List[Dict]:
        """合并不同检索器的结果"""
        merged = []
        seen = set()
        
        faiss_results = results.get('faiss', [])
        es_results = results.get('es', [])
        
        # 合并分数
        all_results = []
        
        # 处理Faiss结果
        for i, result in enumerate(faiss_results):
            score = result.get('similarity', 0) * weights['faiss']
            # 添加位置衰减
            score *= (1.0 - i * 0.01)
            
            result_key = self._get_result_key(result)
            if result_key not in seen:
                all_results.append({
                    **result,
                    'final_score': score,
                    'source': 'faiss'
                })
                seen.add(result_key)
        
        # 处理ES
        for i, result in enumerate(es_results):
            normalized_score = result.get('normalized_score', 0)
            score = normalized_score * weights['es']
            # 添加位置衰减
            score *= (1.0 - i * 0.01)
            
            result_key = self._get_result_key(result)
            if result_key not in seen:
                # 转换ES
                standardized_result = {
                    'question': result.get('question', ''),
                    'answer': result.get('answer', ''),
                    'similarity': normalized_score,
                    'final_score': score,
                    'source': 'es'
                }
                all_results.append(standardized_result)
                seen.add(result_key)
            else:
                # 如果已存在，增加分数
                for existing in all_results:
                    if self._get_result_key(existing) == result_key:
                        existing['final_score'] += score * 0.5  # 重复结果给一半权重
                        existing['source'] = 'hybrid'
                        break
        
        # 按最终分数排序
        all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return all_results[:top_k * 2]  # 返回更多结果供后续筛选

    def _get_result_key(self, result: Dict) -> str:
        """生成结果的唯一标识"""
        question = result.get('question', result.get('title', ''))
        answer = result.get('answer', '')
        return f"{question[:50]}_{answer[:50]}"

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息"""
        return {
            'document_count': len(self.documents),
            'faiss_index_size': getattr(self.faiss_retriever, 'index', {}).get('ntotal', 0) if hasattr(self.faiss_retriever, 'index') else 0,
            'es_available': hasattr(self.es_retriever, 'es') and self.es_retriever.es is not None
        } 