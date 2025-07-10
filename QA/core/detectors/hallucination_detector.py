# -*- coding: utf-8 -*-

import re
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
import jieba

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    import statistics
    class NumpyFallback:
        def var(self, data):
            return statistics.variance(data) if len(data) > 1 else 0
        def mean(self, data):
            return statistics.mean(data) if data else 0
        def array(self, data):
            return data
    np = NumpyFallback()
    cosine_similarity = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class HallucinationType(Enum):
    FACTUAL_ERROR = "factual_error"
    INCONSISTENCY = "inconsistency"
    OVERCONFIDENCE = "overconfidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    REPETITION = "repetition"
    IRRELEVANT = "irrelevant"
    UNCERTAINTY = "uncertainty"

@dataclass
class HallucinationResult:
    is_hallucination: bool
    confidence: float
    hallucination_type: Optional[HallucinationType]
    evidence: list
    suggestions: list
    score_breakdown: dict

class EnhancedHallucinationDetector:
    """幻觉检测器"""
    def __init__(self):
        # 初始化语义模型
        self.semantic_model = self._init_semantic_model()
        # CHATGPT生成的，自由换
        # 不确定性表达词汇
        self.uncertainty_phrases = {
            "可能", "也许", "大概", "估计", "推测", "猜测", "不确定", 
            "不太清楚", "可能是", "据我了解", "我认为", "我觉得",
            "应该是", "似乎", "看起来", "听说", "据说", "疑似"
        }

        # 过度自信表达词汇
        self.confidence_phrases = {
            "绝对", "肯定", "一定", "必须", "必然", "毫无疑问", 
            "确实", "当然", "显然", "明显", "100%", "完全确定",
            "永远", "从不", "所有", "任何情况下"
        }

        # 医疗免责声明
        self.medical_disclaimers = {
            "请咨询医生", "建议就医", "专业医疗建议", "医生诊断",
            "不能替代", "仅供参考", "具体情况", "个体差异",
            "因人而异", "遵医嘱"
        }

        # 危险医疗模式
        self.dangerous_medical_patterns = {
            "绝对不能.*运动": "高血压患者完全不能运动是错误的",
            "立即死亡": "使用'立即死亡'等恐吓性语言是不当的",
            "5分钟.*治愈.*癌症": "声称短时间内治愈癌症是虚假的",
            "任何.*特效药.*治愈.*癌症": "声称有特效药能治愈所有癌症是不实的",
            "绝对不能.*药物": "过于绝对化的药物禁忌建议",
            "必须.*停药": "给出绝对化的停药建议是危险的",
            "100%.*治愈": "声称100%治愈率是不现实的",
            "永远不会.*复发": "声称永远不会复发是过度承诺",
            "立即.*壮阳": "声称立即壮阳效果是不科学的",
            "永久.*增大": "声称永久增大是虚假宣传",
            "一次.*根治": "声称一次治疗根治男科疾病是不实的",
            "100%.*提高.*性能力": "声称100%提高性能力是过度承诺",
            "绝对.*无副作用": "声称绝对无副作用是不实的",
            "天然.*无害": "声称天然产品无害是错误的",
            "包治百病": "声称包治百病是虚假宣传",
            "神药": "使用'神药'等夸大宣传是不当的"
        }

        # 医疗误解模式
        self.medical_misconceptions = {
            "高血压.*绝对不能运动": "高血压患者适当运动是有益的",
            "糖尿病.*不能吃水果": "糖尿病患者可以适量吃水果",
            "感冒.*必须用抗生素": "感冒通常是病毒感染，不需要抗生素",
            "发烧.*立即输液": "低热不需要立即输液",
            "所有.*中药.*无副作用": "中药也可能有副作用",
            "癌症.*完全不能治愈": "许多癌症是可以治疗和控制的"
        }

        # 医疗知识图谱
        self.medical_knowledge_graph = {
            "高血压": {
                "type": "disease",
                "symptoms": ["头痛", "头晕", "耳鸣", "心悸"],
                "treatments": ["降压药", "ACEI", "ARB", "利尿剂"],
                "contraindications": ["高盐饮食", "过度饮酒"],
                "compatible_activities": ["散步", "慢跑", "游泳", "太极"],
                "severity_levels": ["轻度", "中度", "重度"],
                "related_diseases": ["冠心病", "糖尿病", "肾病"]
            },
            "糖尿病": {
                "type": "disease",
                "symptoms": ["多饮", "多尿", "多食", "体重减轻"],
                "treatments": ["胰岛素", "二甲双胍", "饮食控制", "运动"],
                "contraindications": ["高糖饮食", "久坐不动"],
                "compatible_foods": ["蔬菜", "瘦肉", "鱼类", "低糖水果"],
                "severity_levels": ["轻度", "中度", "重度"],
                "related_diseases": ["高血压", "冠心病", "肾病"]
            },
            "感冒": {
                "type": "disease",
                "symptoms": ["发热", "咳嗽", "流鼻涕", "咽痛"],
                "treatments": ["休息", "多喝水", "对症治疗"],
                "contraindications": ["抗生素滥用", "过度劳累"],
                "severity_levels": ["轻度", "中度"],
                "related_diseases": ["流感", "咽炎", "支气管炎"]
            },
            "癌症": {
                "type": "disease",
                "symptoms": ["不明原因体重下降", "持续疲劳", "异常肿块"],
                "treatments": ["手术", "化疗", "放疗", "靶向治疗", "免疫治疗"],
                "contraindications": ["延误治疗", "迷信偏方"],
                "severity_levels": ["早期", "中期", "晚期"],
                "related_diseases": ["转移癌", "复发癌"]
            }
        }

        # 同义词词典
        self.medical_synonyms = {
            "高血压": ["高血压病", "血压高", "血压升高"],
            "糖尿病": ["糖尿病", "血糖高", "血糖升高"],
            "感冒": ["感冒", "伤风", "上呼吸道感染"],
            "发烧": ["发热", "体温升高", "发烧"],
            "头痛": ["头疼", "脑袋疼", "头部疼痛"]
        }

    def _init_semantic_model(self):
        """初始化语义模型"""
        try:
            if SentenceTransformer is not None:
                model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("语义模型加载")
                return model
            else:
                logger.warning("SentenceTransformer未安装，使用规则模式")
                return None
        except Exception as e:
            logger.error(f"语义模型初始化失败: {e}")
            return None

    def _extract_medical_entities(self, text: str) -> List[str]:
        """提取医疗实体"""
        entities = []
        
        # 检查知识图谱中的实体
        for entity, info in self.medical_knowledge_graph.items():
            if entity in text:
                entities.append(entity)
                
        # 检查同义词
        for standard_term, synonyms in self.medical_synonyms.items():
            for synonym in synonyms:
                if synonym in text:
                    entities.append(standard_term)
                    break
                    
        return list(set(entities))

    def _verify_medical_facts(self, text: str, entities: List[str]) -> List[Dict]:
        """验证医疗事实"""
        fact_violations = []
        
        for entity in entities:
            if entity in self.medical_knowledge_graph:
                kg_info = self.medical_knowledge_graph[entity]
                
                # 检查矛盾的治疗方法
                for contraindication in kg_info.get('contraindications', []):
                    if contraindication in text:
                        fact_violations.append({
                            'type': 'contraindication_violation',
                            'entity': entity,
                            'violation': contraindication,
                            'description': f"{entity}与{contraindication}存在矛盾"
                        })
                
                # 检查不兼容的活动/食物
                compatible_items = kg_info.get('compatible_activities', []) + kg_info.get('compatible_foods', [])
                if compatible_items:
                    for item in compatible_items:
                        if f"不能{item}" in text or f"禁止{item}" in text:
                            fact_violations.append({
                                'type': 'compatibility_violation',
                                'entity': entity,
                                'violation': item,
                                'description': f"{entity}患者实际上可以{item}"
                            })
        
        return fact_violations

    def detect(self, generated_text: str, source_contexts=None, query=None, probability_info=None):
        """检测幻觉"""
        try:
            scores = {}
            evidence = []
            suggestions = []

            # 提取医疗实体
            medical_entities = self._extract_medical_entities(generated_text)
            
            # 1. 语义一致性检测
            consistency_score, consistency_evidence = self._check_semantic_consistency(
                generated_text, source_contexts
            )
            scores["semantic_consistency"] = consistency_score
            evidence.extend(consistency_evidence)

            # 2. 医疗事实验证
            fact_score, fact_evidence = self._check_medical_facts(
                generated_text, medical_entities
            )
            scores["medical_facts"] = fact_score
            evidence.extend(fact_evidence)

            # 3. 不确定性分析
            uncertainty_score, uncertainty_evidence = self._analyze_uncertainty_advanced(
                generated_text
            )
            scores["uncertainty"] = uncertainty_score
            evidence.extend(uncertainty_evidence)

            # 4. 过度自信检测
            overconfidence_score, overconf_evidence = self._detect_overconfidence(
                generated_text
            )
            scores["overconfidence"] = overconfidence_score
            evidence.extend(overconf_evidence)

            # 5. 语义相关性检测
            relevance_score, relevance_evidence = self._check_semantic_relevance(
                generated_text, query
            )
            scores["semantic_relevance"] = relevance_score
            evidence.extend(relevance_evidence)

            # 6. 重复检测
            repetition_score, repetition_evidence = self._detect_repetition(
                generated_text
            )
            scores["repetition"] = repetition_score
            evidence.extend(repetition_evidence)

            # 7. 医疗安全检测
            safety_score, safety_evidence = self._check_medical_safety(
                generated_text
            )
            scores["medical_safety"] = safety_score
            evidence.extend(safety_evidence)

            # 8. 概率分布检测
            if probability_info:
                prob_score, prob_evidence = self._check_probability_distribution(
                    generated_text, probability_info
                )
                scores["probability_distribution"] = prob_score
                evidence.extend(prob_evidence)

            # 计算最终得分
            final_score = self._calculate_final_score(scores)
            hallucination_type = self._determine_hallucination_type(scores)
            suggestions = self._generate_suggestions(scores, hallucination_type)
            is_hallucination = final_score > 0.3

            return HallucinationResult(
                is_hallucination=is_hallucination,
                confidence=final_score,
                hallucination_type=hallucination_type,
                evidence=evidence,
                suggestions=suggestions,
                score_breakdown=scores
            )

        except Exception as e:
            logger.error(f"幻觉检测失败: {e}")
            return HallucinationResult(
                is_hallucination=False,
                confidence=0.0,
                hallucination_type=None,
                evidence=["检测过程发生错误"],
                suggestions=["建议人工审核"],
                score_breakdown={}
            )

    def _check_semantic_consistency(self, text: str, contexts=None):
        """
        语义一致性检测
        使用语义相似度而非简单字符串匹配
        """
        if not contexts:
            return 0.0, []
        
        evidence = []
        score = 0.0
        
        try:
            if self.semantic_model is not None and cosine_similarity is not None:
                # 使用语义模型计算相似度
                all_texts = [text] + contexts
                embeddings = self.semantic_model.encode(all_texts)
                
                # 计算生成文本与每个上下文的相似度
                text_embedding = embeddings[0:1]
                context_embeddings = embeddings[1:]
                
                similarities = cosine_similarity(text_embedding, context_embeddings)[0]
                avg_similarity = np.mean(similarities)
                
                # 低相似度表示不一致
                if avg_similarity < 0.3:
                    score = 0.8
                    evidence.append(f"语义相似度过低: {avg_similarity:.3f}")
                elif avg_similarity < 0.5:
                    score = 0.5
                    evidence.append(f"语义相似度较低: {avg_similarity:.3f}")
                else:
                    score = 0.0
                    evidence.append(f"语义一致性良好: {avg_similarity:.3f}")
                    
            else:
                # 降级到简单字符串匹配
                for context in contexts:
                    if len(context) > 10:  # 忽略过短的上下文
                        # 检查关键词重叠
                        text_words = set(jieba.cut(text))
                        context_words = set(jieba.cut(context))
                        overlap = len(text_words & context_words)
                        
                        if overlap < 3:
                            score += 0.2
                            evidence.append(f"与源文档关键词重叠过少")
                
                score = min(score, 1.0)
                
        except Exception as e:
            logger.error(f"语义一致性检测失败: {e}")
            score = 0.0
            evidence.append("一致性检测过程出错")
        
        return score, evidence

    def _check_medical_facts(self, text: str, medical_entities: List[str]):
        """
        医疗事实验证
        基于医疗知识图谱验证事实准确性
        """
        evidence = []
        score = 0.0
        
        try:
            # 验证医疗事实
            fact_violations = self._verify_medical_facts(text, medical_entities)
            
            if fact_violations:
                score = min(len(fact_violations) * 0.3, 1.0)
                for violation in fact_violations:
                    evidence.append(f"医疗事实错误: {violation['description']}")
            
            # 检查医疗误解模式
            for pattern, correct_info in self.medical_misconceptions.items():
                if re.search(pattern, text):
                    score += 0.4
                    evidence.append(f"医疗误解: {correct_info}")
            
            score = min(score, 1.0)
            
        except Exception as e:
            logger.error(f"医疗事实验证失败: {e}")
            score = 0.0
            evidence.append("医疗事实验证过程出错")
        
        return score, evidence

    def _analyze_uncertainty_advanced(self, text: str):
        """
        不确定性分析    结合词汇分析和语义理解
        """
        evidence = []
        uncertainty_count = 0
        confidence_count = 0

        # 统计不确定性表达
        for phrase in self.uncertainty_phrases:
            if phrase in text:
                uncertainty_count += 1
                evidence.append(f"包含不确定性表达: '{phrase}'")

        # 统计过度自信表达
        for phrase in self.confidence_phrases:
            if phrase in text:
                confidence_count += 1

        # 计算不确定性得分
        text_length = len(text)
        uncertainty_ratio = uncertainty_count / (text_length / 100) if text_length > 0 else 0
        confidence_ratio = confidence_count / (text_length / 100) if text_length > 0 else 0

        # 适当的不确定性表达
        if uncertainty_count == 0 and confidence_count > 0:
            score = 0.5
            evidence.append("医疗建议缺乏适当的不确定性表达")
        elif uncertainty_count == 0 and text_length > 100:
            score = 0.3
            evidence.append("较长回答缺乏不确定性表达")
        elif uncertainty_count > 8:
            score = 0.6
            evidence.append("过多不确定性表达可能影响信息价值")
        elif confidence_ratio > 0.05:  # 过度自信
            score = 0.4
            evidence.append("过多自信表达在医疗领域存在风险")
        else:
            score = 0.0
            if uncertainty_count > 0:
                evidence.append(f"不确定性表达适当: {uncertainty_count}个")

        return score, evidence

    def _detect_overconfidence(self, text):
        evidence = []
        confidence_count = 0

        for phrase in self.confidence_phrases:
            if phrase in text:
                confidence_count += 1
                evidence.append(f"发现过度自信表达: '{phrase}'")

        score = min(confidence_count * 0.15, 1.0)
        return score, evidence

    def _check_semantic_relevance(self, text: str, query: str):
        """
        语义相关性检测  使用语义模型计算查询与回答的相关性
        """
        if not query:
            return 0.0, []
        
        evidence = []
        score = 0.0
        
        try:
            if self.semantic_model is not None and cosine_similarity is not None:
                # 自由换模型
                embeddings = self.semantic_model.encode([query, text])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                # 分数自己根据业务来测试调
                if similarity < 0.3:
                    score = 0.7
                    evidence.append(f"语义相关性很低: {similarity:.3f}")
                elif similarity < 0.5:
                    score = 0.4
                    evidence.append(f"语义相关性较低: {similarity:.3f}")
                else:
                    score = 0.0
                    evidence.append(f"语义相关性良好: {similarity:.3f}")
                    
            else:
                # 降级到关键词匹配
                query_words = set(jieba.cut(query))
                text_words = set(jieba.cut(text))
                
                overlap = len(query_words & text_words)
                overlap_ratio = overlap / len(query_words) if query_words else 0
                
                if overlap_ratio < 0.2:
                    score = 0.6
                    evidence.append(f"关键词重叠过少: {overlap}/{len(query_words)}")
                elif overlap_ratio < 0.4:
                    score = 0.3
                    evidence.append(f"关键词重叠较少: {overlap}/{len(query_words)}")
                else:
                    score = 0.0
                    evidence.append(f"关键词重叠充足: {overlap}/{len(query_words)}")
                    
        except Exception as e:
            logger.error(f"语义相关性检测失败: {e}")
            score = 0.0
            evidence.append("相关性检测过程出错")
        
        return score, evidence

    def _detect_repetition(self, text):
        evidence = []
        sentences = text.split('。')
        
        repetition_count = 0
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                if sent1 == sent2 and len(sent1) > 5:
                    repetition_count += 1
                    evidence.append(f"发现重复内容: '{sent1}'")
        
        score = min(repetition_count * 0.2, 1.0)
        return score, evidence

    def _check_medical_safety(self, text):
        evidence = []
        score = 0.0
        
        for pattern, description in self.dangerous_medical_patterns.items():
            if re.search(pattern, text):
                score += 0.3
                evidence.append(f"危险医疗信息: {description}")
        
        return min(score, 1.0), evidence

    def _check_probability_distribution(self, text, probability_info):
        try:
            if not probability_info or 'statistics' not in probability_info:
                return 0.0, []
            
            stats = probability_info['statistics']
            evidence = []
            score = 0.0
            
            # 检查平均概率
            if stats.get('average_probability', 1.0) < 0.3:
                score += 0.4
                evidence.append("生成概率偏低")
            
            # 检查困惑度
            if stats.get('perplexity', 1.0) > 50:
                score += 0.3
                evidence.append("困惑度过高")
            
            return min(score, 1.0), evidence
        except:
            return 0.0, []

    def _calculate_final_score(self, scores):
        """
        计算最终幻觉得分（权重）
        """
        if not scores:
            return 0.0
        
        # 权重分配 -根据业务来
        weights = {
            'semantic_consistency': 0.2,    # 语义一致性
            'medical_facts': 0.25,          # 医疗事实准确性（新增，高权重）
            'uncertainty': 0.12,            # 不确定性分析
            'overconfidence': 0.15,         # 过度自信检测
            'semantic_relevance': 0.1,      # 语义相关性
            'repetition': 0.08,             # 重复检测
            'medical_safety': 0.3,          # 医疗安全（最高权重）
            'probability_distribution': 0.1  # 概率分布检测
        }
        
        final_score = 0.0
        total_weight = 0.0
        
        for key, score in scores.items():
            weight = weights.get(key, 0.05)  # 未知类型给予较低权重
            final_score += score * weight
            total_weight += weight
        
        return final_score / total_weight if total_weight > 0 else 0.0

    def _determine_hallucination_type(self, scores):
        """
        确定幻觉类型（映射）
        """
        max_score = 0
        max_type = None
        
        # 类型映射
        type_mapping = {
            'semantic_consistency': HallucinationType.INCONSISTENCY,
            'medical_facts': HallucinationType.FACTUAL_ERROR,
            'overconfidence': HallucinationType.OVERCONFIDENCE,
            'medical_safety': HallucinationType.FACTUAL_ERROR,
            'semantic_relevance': HallucinationType.IRRELEVANT,
            'repetition': HallucinationType.REPETITION,
            'uncertainty': HallucinationType.UNCERTAINTY,
            'probability_distribution': HallucinationType.UNSUPPORTED_CLAIM
        }
        
        for key, score in scores.items():
            if score > max_score and key in type_mapping:
                max_score = score
                max_type = type_mapping[key]
        
        return max_type

    def _generate_suggestions(self, scores, hallucination_type):
        """
        生成改进建议
        """
        suggestions = []
        
        # 语义一致性建议
        if scores.get('semantic_consistency', 0) > 0.3:
            suggestions.append("请检查回答是否与源文档语义一致")
        
        # 医疗事实准确性建议
        if scores.get('medical_facts', 0) > 0.3:
            suggestions.append("请验证医疗事实的准确性，避免与医学知识冲突")
        
        # 过度自信建议
        if scores.get('overconfidence', 0) > 0.3:
            suggestions.append("建议使用更谨慎的表达，避免绝对化陈述")
        
        # 医疗安全建议
        if scores.get('medical_safety', 0) > 0.3:
            suggestions.append("请避免给出可能危险的医疗建议，建议添加免责声明")
        
        # 语义相关性建议
        if scores.get('semantic_relevance', 0) > 0.3:
            suggestions.append("请确保回答与问题语义相关")
        
        # 不确定性建议
        if scores.get('uncertainty', 0) > 0.3:
            suggestions.append("建议在医疗建议中适当表达不确定性")
        
        # 重复性建议
        if scores.get('repetition', 0) > 0.3:
            suggestions.append("请避免重复表达相同内容")
        
        # 概率分布建议
        if scores.get('probability_distribution', 0) > 0.3:
            suggestions.append("生成质量可能不稳定，建议重新生成")
        
        # 根据幻觉类型提供专门建议
        if hallucination_type == HallucinationType.FACTUAL_ERROR:
            suggestions.append("建议核实医疗事实的准确性")
        elif hallucination_type == HallucinationType.OVERCONFIDENCE:
            suggestions.append("建议使用'可能'、'通常'等谨慎表达")
        elif hallucination_type == HallucinationType.INCONSISTENCY:
            suggestions.append("建议检查回答与参考资料的一致性")
        
        if not suggestions:
            suggestions.append("回答质量良好，符合医疗AI标准")
        
        return suggestions[:4]  # 返回最多4个建议