import re
import jieba
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    def __init__(self):
        # 查询类型模式
        self.query_type_patterns = {
            'symptom': ['症状', '感觉', '不舒服', '疼痛', '难受', '发热', '咳嗽'],
            'diagnosis': ['诊断', '什么病', '是不是', '可能是', '确诊'],
            'treatment': ['治疗', '怎么办', '用药', '吃什么药', '如何治疗'],
            'prevention': ['预防', '避免', '注意事项', '保健', '预防措施'],
            'factual': ['什么是', '介绍', '了解', '科普', '原因', '为什么']
        }
        
        # 紧急程度关键词
        self.urgency_keywords = {
            'high': ['急性', '严重', '紧急', '立即', '马上', '剧烈'],
            'medium': ['持续', '反复', '经常', '一直'],
            'low': ['偶尔', '轻微', '一般', '普通']
        }
        
        # 医学实体模式
        self.medical_entities = {
            'symptoms': [
                '头痛', '发烧', '咳嗽', '乏力', '失眠', '腹痛', '头晕', '恶心',
                '胸痛', '呼吸困难', '心悸', '出汗', '食欲不振', '体重减轻'
            ],
            'diseases': [
                '高血压', '糖尿病', '感冒', '肺炎', '胃炎', '心脏病', '肾病',
                '肝病', '关节炎', '过敏', '哮喘', '抑郁症', '焦虑症'
            ],
            'body_parts': [
                '头部', '胸部', '腹部', '背部', '四肢', '心脏', '肺部', '肝脏',
                '肾脏', '胃部', '肠道', '关节', '皮肤', '眼睛'
            ]
        }

    def analyze(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """分析查询，返回查询类型、实体、紧急程度等信息"""
        try:
            # 预处理查询
            cleaned_query = self._preprocess_query(query)
            
            # 识别查询类型
            query_type = self._classify_query_type(cleaned_query)
            
            # 提取医学实体
            entities = self._extract_entities(cleaned_query)
            
            # 评估紧急程度
            urgency = self._assess_urgency(cleaned_query)
            
            # 计算置信度
            confidence = self._calculate_confidence(query_type, entities, cleaned_query)
            
            # 生成查询特征
            features = self._extract_features(cleaned_query, entities)
            
            analysis_result = {
                'query_type': query_type,
                'entities': entities,
                'urgency': urgency,
                'confidence': confidence,
                'features': features,
                'processed_query': cleaned_query
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            return {
                'query_type': 'general',
                'entities': {},
                'urgency': 'low',
                'confidence': 0.3,
                'features': {},
                'processed_query': query
            }

    def _preprocess_query(self, query: str) -> str:
        """预处理查询文本"""
        # 去除多余空白
        query = re.sub(r'\s+', ' ', query.strip())
        
        # 转换为小写（对于某些分析有用）
        # 注意：中文不需要小写转换，但可以做其他标准化
        return query

    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        type_scores = {}
        
        for query_type, patterns in self.query_type_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query:
                    score += 1
            type_scores[query_type] = score
        
        # 返回得分最高的类型
        if type_scores and max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        return 'general'

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """提取医学实体"""
        entities = {
            'symptoms': [],
            'diseases': [],
            'body_parts': []
        }
        
        # 使用jieba分词
        words = list(jieba.cut(query))
        
        # 匹配实体
        for entity_type, patterns in self.medical_entities.items():
            for pattern in patterns:
                if pattern in query:
                    entities[entity_type].append(pattern)
        
        return entities

    def _assess_urgency(self, query: str) -> str:
        """评估查询紧急程度"""
        urgency_scores = {
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        for urgency_level, keywords in self.urgency_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    urgency_scores[urgency_level] += 1
        
        # 返回得分最高的紧急程度
        if urgency_scores['high'] > 0:
            return 'high'
        elif urgency_scores['medium'] > 0:
            return 'medium'
        else:
            return 'low'

    def _calculate_confidence(self, query_type: str, entities: Dict, query: str) -> float:
        """计算分析结果的置信度"""
        confidence = 0.5  # 基础置信度
        
        # 如果识别到明确的查询类型，增加置信度
        if query_type != 'general':
            confidence += 0.2
        
        # 如果提取到实体，增加置信度
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        if entity_count > 0:
            confidence += min(entity_count * 0.1, 0.3)
        
        # 查询长度影响
        if len(query) > 10:  # 较长的查询通常更明确
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _extract_features(self, query: str, entities: Dict) -> Dict[str, Any]:
        """提取查询特征"""
        features = {
            'query_length': len(query),
            'word_count': len(list(jieba.cut(query))),
            'entity_count': sum(len(entity_list) for entity_list in entities.values()),
            'has_question_words': self._has_question_words(query),
            'has_negation': self._has_negation(query),
            'is_complex': self._is_complex_query(query)
        }
        
        return features

    def _has_question_words(self, query: str) -> bool:
        """检查是否包含疑问词"""
        question_words = ['什么', '怎么', '如何', '为什么', '哪里', '谁', '吗', '呢']
        return any(word in query for word in question_words)

    def _has_negation(self, query: str) -> bool:
        """检查是否包含否定词"""
        negation_words = ['不', '没', '无', '非', '未']
        return any(word in query for word in negation_words)

    def _is_complex_query(self, query: str) -> bool:
        """判断是否为复杂查询"""
        # 简单规则：包含多个实体或查询较长
        words = list(jieba.cut(query))
        return len(words) > 10 or '并且' in query or '还有' in query or '同时' in query 