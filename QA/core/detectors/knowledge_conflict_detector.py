import re
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import torch

logger = logging.getLogger(__name__)

class ConflictType(Enum):
    """冲突类型枚举"""
    VIEWPOINT_CONFLICT = "viewpoint_conflict"    # 观点冲突
    TEMPORAL_CONFLICT = "temporal_conflict"      # 时效性冲突
    AUTHORITY_CONFLICT = "authority_conflict"    # 权威性冲突
    SEVERITY_CONFLICT = "severity_conflict"      # 严重程度冲突
    TREATMENT_CONFLICT = "treatment_conflict"    # 治疗方案冲突
    DOSAGE_CONFLICT = "dosage_conflict"         # 剂量冲突

@dataclass
class ConflictEvidence:
    """冲突证据"""
    source_id: str
    content: str
    position: str  # 立场/观点
    confidence: float
    authority_score: float
    timestamp: Optional[str] = None

@dataclass
class ConflictDetail:
    """冲突详情"""
    conflict_type: ConflictType
    description: str
    evidence_list: List[ConflictEvidence]
    severity: str  # low, medium, high
    recommendation: str

@dataclass
class ConflictAnalysisResult:
    """冲突分析结果"""
    has_conflicts: bool
    conflicts: List[ConflictDetail]
    reconciled_info: Dict[str, Any]
    confidence_score: float
    safety_warning: Optional[str] = None

class KnowledgeConflictDetector:
    """知识冲突检测器 - ，使用语义理解+规则匹配"""
    
    def __init__(self):
        # 初始化语义模型
        try:
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.use_semantic = True
            logger.info("语义模型加载")
        except Exception as e:
            logger.warning(f"语义模型加载失败，使用规则模式: {e}")
            self.use_semantic = False
        
        # 医疗知识图谱 - 实体关系定义
        self.medical_kg = {
            '高血压': {
                'contradictions': ['低血压', '血压正常'],
                'treatments': ['降压药', 'ACEI', 'ARB', '利尿剂'],
                'contraindications': ['升压药', '高盐饮食过量'],
                'severity_levels': ['轻度', '中度', '重度']
            },
            '糖尿病': {
                'contradictions': ['血糖正常', '低血糖'],
                'treatments': ['胰岛素', '二甲双胍', '磺脲类'],
                'contraindications': ['高糖饮食', '大量饮酒'],
                'severity_levels': ['前期', '轻度', '中重度']
            }
        }
        
        # 强化的对立观点模式 - 增加语义权重
        self.opposing_patterns = [
            {
                'positive': r'(建议|推荐|应该|最好|可以)',
                'negative': r'(不建议|不推荐|不应该|避免|禁止|不可以)',
                'weight': 0.9,
                'context': '治疗建议'
            },
            {
                'positive': r'(安全|无害|无风险)',
                'negative': r'(危险|有害|有风险|禁用)',
                'weight': 0.95,
                'context': '安全性评估'
            },
            {
                'positive': r'(有效|管用|好用|疗效好)',
                'negative': r'(无效|不管用|没用|疗效差)',
                'weight': 0.8,
                'context': '疗效评价'
            },
            {
                'positive': r'(轻微|不严重|问题不大|良性)',
                'negative': r'(严重|危险|需要注意|恶性)',
                'weight': 0.85,
                'context': '严重程度'
            }
        ]
        
        # 增强的剂量冲突检测模式
        self.dosage_patterns = [
            {
                'pattern': r'(\d+(?:\.\d+)?)\s*(mg|g|ml|片|粒|次)',
                'type': 'basic_dosage',
                'weight': 1.0
            },
            {
                'pattern': r'(每天|每日|一天)\s*(\d+)\s*(次|片|粒)',
                'type': 'frequency',
                'weight': 0.9
            },
            {
                'pattern': r'(一次|每次)\s*(\d+(?:\.\d+)?)\s*(mg|g|ml|片|粒)',
                'type': 'single_dose',
                'weight': 0.85
            },
            {
                'pattern': r'(\d+)\s*-\s*(\d+)\s*(mg|g|ml|片|粒)',
                'type': 'range_dose',
                'weight': 0.8
            }
        ]
        
        # 剂量单位转换表
        self.dosage_conversion = {
            'mg': 1,
            'g': 1000,
            'ml': 1,  # 液体剂量
            '片': 1,  # 固体制剂
            '粒': 1   # 胶囊等
        }
        
        # 时间相关词汇
        self.temporal_words = [
            '最新', '新版', '更新', '修订', '2023', '2024', '近期', '目前',
            '以前', '过去', '原来', '传统', '老版', '旧版'
        ]
        
        # 分层权威性标识 - 增加权重和可信度评估
        self.authority_indicators = {
            '国际权威': {
                'keywords': ['WHO', '世卫组织', 'FDA', 'EMA', 'NICE指南', 'Cochrane'],
                'weight': 1.0,
                'confidence': 0.95
            },
            '国家权威': {
                'keywords': ['卫生部', '药监局', '国家卫健委', '中华医学会', '国家标准'],
                'weight': 0.9,
                'confidence': 0.9
            },
            '学术权威': {
                'keywords': ['临床试验', 'RCT', '荟萃分析', '系统评价', 'Lancet', 'NEJM'],
                'weight': 0.85,
                'confidence': 0.85
            },
            '专业权威': {
                'keywords': ['专家共识', '指南', '诊疗规范', '院士', '主任医师'],
                'weight': 0.75,
                'confidence': 0.8
            },
            '一般来源': {
                'keywords': ['研究显示', '数据表明', '报告指出', '专家', '教授'],
                'weight': 0.6,
                'confidence': 0.7
            }
        }
        
        # 严重程度词汇
        self.severity_words = {
            'high': ['危险', '严重', '紧急', '致命', '危及', '急性', '恶性'],
            'medium': ['注意', '小心', '谨慎', '可能', '风险', '不良'],
            'low': ['轻微', '一般', '常见', '正常', '无害', '良性']
        }
    
    def detect_conflicts(self, retrieval_results: List[Dict[str, Any]], 
                        query: str = "") -> ConflictAnalysisResult:
        """检测检索结果中的冲突"""
        try:
            if len(retrieval_results) < 2:
                return ConflictAnalysisResult(
                    has_conflicts=False,
                    conflicts=[],
                    reconciled_info={},
                    confidence_score=1.0
                )
            
            # 提取内容用于分析
            contents = []
            for i, result in enumerate(retrieval_results):
                content = {
                    'id': str(i),
                    'question': result.get('question', ''),
                    'answer': result.get('answer', ''),
                    'full_text': f"{result.get('question', '')} {result.get('answer', '')}",
                    'score': result.get('score', 0.0),
                    'source': result.get('source', 'unknown')
                }
                contents.append(content)
            
            # 检测各类冲突
            conflicts = []
            
            # 1. 观点冲突检测
            viewpoint_conflicts = self._detect_viewpoint_conflicts(contents)
            conflicts.extend(viewpoint_conflicts)
            
            # 2. 剂量冲突检测
            dosage_conflicts = self._detect_dosage_conflicts(contents)
            conflicts.extend(dosage_conflicts)
            
            # 3. 时效性冲突检测
            temporal_conflicts = self._detect_temporal_conflicts(contents)
            conflicts.extend(temporal_conflicts)
            
            # 4. 权威性冲突检测
            authority_conflicts = self._detect_authority_conflicts(contents)
            conflicts.extend(authority_conflicts)
            
            # 5. 严重程度冲突检测
            severity_conflicts = self._detect_severity_conflicts(contents)
            conflicts.extend(severity_conflicts)
            
            # 生成协调信息
            reconciled_info = self._generate_reconciled_info(contents, conflicts)
            
            # 计算整体置信度
            confidence_score = self._calculate_confidence(contents, conflicts)
            
            # 生成安全警告
            safety_warning = self._generate_safety_warning(conflicts)
            
            return ConflictAnalysisResult(
                has_conflicts=len(conflicts) > 0,
                conflicts=conflicts,
                reconciled_info=reconciled_info,
                confidence_score=confidence_score,
                safety_warning=safety_warning
            )
            
        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
            return ConflictAnalysisResult(
                has_conflicts=False,
                conflicts=[],
                reconciled_info={},
                confidence_score=0.5,
                safety_warning="冲突检测过程中出现错误，请谨慎参考信息"
            )
    
    def _detect_viewpoint_conflicts(self, contents: List[Dict]) -> List[ConflictDetail]:
        """检测观点冲突 - 使用语义理解+规则匹配"""
        conflicts = []
        
        # 语义冲突检测
        if self.use_semantic and len(contents) > 1:
            semantic_conflicts = self._detect_semantic_conflicts(contents)
            conflicts.extend(semantic_conflicts)
        
        # 规则模式冲突检测
        for pattern_info in self.opposing_patterns:
            positive_matches = []
            negative_matches = []
            
            for content in contents:
                text = content['full_text']
                
                # 使用新的模式结构
                positive_pattern = pattern_info['positive']
                negative_pattern = pattern_info['negative']
                pattern_weight = pattern_info['weight']
                context_type = pattern_info['context']
                
                if re.search(positive_pattern, text):
                    authority_score = self._calculate_authority_score(text)
                    confidence = min(0.8 * pattern_weight * authority_score, 0.95)
                    
                    positive_matches.append(ConflictEvidence(
                        source_id=content['id'],
                        content=text[:200],
                        position=f"支持观点({context_type})",
                        confidence=confidence,
                        authority_score=authority_score
                    ))
                
                if re.search(negative_pattern, text):
                    authority_score = self._calculate_authority_score(text)
                    confidence = min(0.8 * pattern_weight * authority_score, 0.95)
                    
                    negative_matches.append(ConflictEvidence(
                        source_id=content['id'],
                        content=text[:200],
                        position=f"反对观点({context_type})",
                        confidence=confidence,
                        authority_score=authority_score
                    ))
            
            if positive_matches and negative_matches:
                # 计算冲突严重程度
                avg_authority = sum(e.authority_score for e in positive_matches + negative_matches) / len(positive_matches + negative_matches)
                severity = self._calculate_conflict_severity(pattern_weight, avg_authority)
                
                conflict = ConflictDetail(
                    conflict_type=ConflictType.VIEWPOINT_CONFLICT,
                    description=f"发现{context_type}观点冲突：支持vs反对",
                    evidence_list=positive_matches + negative_matches,
                    severity=severity,
                    recommendation=self._generate_viewpoint_recommendation(context_type, avg_authority)
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_dosage_conflicts(self, contents: List[Dict]) -> List[ConflictDetail]:
        """检测剂量冲突"""
        conflicts = []
        dosage_info = defaultdict(list)
        
        for content in contents:
            text = content['full_text']
            
            for pattern in self.dosage_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    if isinstance(match, tuple):
                        dosage_key = match[-1]  # 单位
                        dosage_value = match[0] if len(match) > 1 else match
                    else:
                        dosage_key = "generic"
                        dosage_value = match
                    
                    dosage_info[dosage_key].append({
                        'value': dosage_value,
                        'source_id': content['id'],
                        'content': text[:200],
                        'full_match': str(match)
                    })
        
        # 检查同种剂量单位是否有差异
        for unit, dosages in dosage_info.items():
            if len(dosages) > 1:
                values = [d['value'] for d in dosages]
                if len(set(values)) > 1:  # 有不同的值
                    evidence = [
                        ConflictEvidence(
                            source_id=d['source_id'],
                            content=d['content'],
                            position=f"剂量: {d['full_match']}",
                            confidence=0.9,
                            authority_score=0.7
                        ) for d in dosages
                    ]
                    
                    conflict = ConflictDetail(
                        conflict_type=ConflictType.DOSAGE_CONFLICT,
                        description=f"发现{unit}剂量冲突: {', '.join(set(values))}",
                        evidence_list=evidence,
                        severity="high",
                        recommendation="剂量冲突，必须咨询医生确认正确剂量"
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_temporal_conflicts(self, contents: List[Dict]) -> List[ConflictDetail]:
        """检测时效性冲突"""
        conflicts = []
        new_contents = []
        old_contents = []
        
        for content in contents:
            text = content['full_text']
            
            has_new_indicators = any(word in text for word in self.temporal_words[:8])
            has_old_indicators = any(word in text for word in self.temporal_words[8:])
            
            if has_new_indicators:
                new_contents.append(content)
            elif has_old_indicators:
                old_contents.append(content)
        
        if new_contents and old_contents:
            evidence = []
            
            for content in new_contents:
                evidence.append(ConflictEvidence(
                    source_id=content['id'],
                    content=content['full_text'][:200],
                    position="新版本/最新",
                    confidence=0.7,
                    authority_score=0.8
                ))
            
            for content in old_contents:
                evidence.append(ConflictEvidence(
                    source_id=content['id'],
                    content=content['full_text'][:200],
                    position="旧版本/传统",
                    confidence=0.7,
                    authority_score=0.5
                ))
            
            conflict = ConflictDetail(
                conflict_type=ConflictType.TEMPORAL_CONFLICT,
                description="发现新旧版本信息冲突",
                evidence_list=evidence,
                severity="medium",
                recommendation="建议参考最新版本信息"
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_authority_conflicts(self, contents: List[Dict]) -> List[ConflictDetail]:
        """检测权威性冲突"""
        conflicts = []
        authority_scores = []
        
        for content in contents:
            text = content['full_text']
            score = self._calculate_authority_score(text)
            authority_scores.append((content, score))
        
        # 如果权威性得分差异较大，且内容有冲突
        if len(authority_scores) > 1:
            scores = [score for _, score in authority_scores]
            max_score = max(scores)
            min_score = min(scores)
            
            if max_score - min_score > 0.4:  # 权威性差异阈值
                high_authority = [content for content, score in authority_scores if score >= max_score - 0.1]
                low_authority = [content for content, score in authority_scores if score <= min_score + 0.1]
                
                if high_authority and low_authority:
                    evidence = []
                    
                    for content in high_authority:
                        evidence.append(ConflictEvidence(
                            source_id=content['id'],
                            content=content['full_text'][:200],
                            position="高权威性",
                            confidence=0.8,
                            authority_score=max_score
                        ))
                    
                    for content in low_authority:
                        evidence.append(ConflictEvidence(
                            source_id=content['id'],
                            content=content['full_text'][:200],
                            position="低权威性",
                            confidence=0.6,
                            authority_score=min_score
                        ))
                    
                    conflict = ConflictDetail(
                        conflict_type=ConflictType.AUTHORITY_CONFLICT,
                        description="发现权威性差异较大的信息",
                        evidence_list=evidence,
                        severity="low",
                        recommendation="建议优先参考权威性较高的信息"
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_severity_conflicts(self, contents: List[Dict]) -> List[ConflictDetail]:
        """检测严重程度冲突"""
        conflicts = []
        severity_assessments = []
        
        for content in contents:
            text = content['full_text']
            severity = self._assess_severity(text)
            if severity:
                severity_assessments.append((content, severity))
        
        if len(severity_assessments) > 1:
            severities = [sev for _, sev in severity_assessments]
            unique_severities = set(severities)
            
            if len(unique_severities) > 1:
                evidence = []
                for content, severity in severity_assessments:
                    evidence.append(ConflictEvidence(
                        source_id=content['id'],
                        content=content['full_text'][:200],
                        position=f"严重程度: {severity}",
                        confidence=0.7,
                        authority_score=0.6
                    ))
                
                conflict = ConflictDetail(
                    conflict_type=ConflictType.SEVERITY_CONFLICT,
                    description=f"发现严重程度评估不一致: {', '.join(unique_severities)}",
                    evidence_list=evidence,
                    severity="medium",
                    recommendation="建议以最严重的评估为准，谨慎对待"
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def _detect_semantic_conflicts(self, contents: List[Dict]) -> List[ConflictDetail]:
        """使用语义相似度检测观点冲突"""
        conflicts = []
        
        if len(contents) < 2:
            return conflicts
        
        try:
            # 提取答案文本
            answers = [content.get('answer', '') for content in contents]
            
            # 计算语义向量
            embeddings = self.semantic_model.encode(answers)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(embeddings)
            
            # 找出语义相似度低的文档对（可能存在冲突）
            threshold = 0.3  # 低相似度阈值
            
            for i in range(len(contents)):
                for j in range(i + 1, len(contents)):
                    similarity = similarity_matrix[i][j]
                    
                    if similarity < threshold:
                        # 进一步检查是否为真正的冲突
                        if self._is_semantic_conflict(answers[i], answers[j]):
                            evidence = [
                                ConflictEvidence(
                                    source_id=contents[i]['id'],
                                    content=answers[i][:200],
                                    position="观点A",
                                    confidence=0.7,
                                    authority_score=self._calculate_authority_score(answers[i])
                                ),
                                ConflictEvidence(
                                    source_id=contents[j]['id'],
                                    content=answers[j][:200],
                                    position="观点B",
                                    confidence=0.7,
                                    authority_score=self._calculate_authority_score(answers[j])
                                )
                            ]
                            
                            conflict = ConflictDetail(
                                conflict_type=ConflictType.VIEWPOINT_CONFLICT,
                                description=f"语义冲突检测：相似度{similarity:.2f}",
                                evidence_list=evidence,
                                severity="medium",
                                recommendation="发现语义层面的观点分歧，建议仔细比较两种观点的适用性"
                            )
                            conflicts.append(conflict)
        except Exception as e:
            logger.error(f"语义冲突检测失败: {e}")
        
        return conflicts
    
    def _is_semantic_conflict(self, text1: str, text2: str) -> bool:
        """判断两个文本是否存在语义冲突"""
        # 使用医疗知识图谱检查冲突（Ner）
        entities1 = self._extract_medical_entities(text1)
        entities2 = self._extract_medical_entities(text2)
        
        for entity1 in entities1:
            if entity1 in self.medical_kg:
                contradictions = self.medical_kg[entity1].get('contradictions', [])
                for entity2 in entities2:
                    if entity2 in contradictions:
                        return True
        
        return False
    
    def _extract_medical_entities(self, text: str) -> List[str]:
        """从文本中提取医疗实体"""
        entities = []
        words = jieba.cut(text)
        
        for word in words:
            if word in self.medical_kg:
                entities.append(word)
        
        return entities
    
    def _calculate_conflict_severity(self, pattern_weight: float, authority_score: float) -> str:
        """计算冲突严重程度"""
        severity_score = pattern_weight * authority_score
        
        if severity_score >= 0.8:
            return "high"
        elif severity_score >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_viewpoint_recommendation(self, context_type: str, authority_score: float) -> str:
        """生成观点冲突的建议"""
        base_recommendations = {
            '治疗建议': "发现治疗方案冲突，建议：1)咨询专业医生 2)考虑个体差异 3)遵循最新指南",
            '安全性评估': "发现安全性评价分歧，建议：1)以保守评估为准 2)咨询药师 3)密切监测",
            '疗效评价': "发现疗效评价不一，建议：1)参考循证医学证据 2)个体化治疗 3)定期评估",
            '严重程度': "发现严重程度判断分歧，建议：1)按最严重情况处理 2)及时就医 3)密切观察"
        }
        
        base_rec = base_recommendations.get(context_type, "建议综合考虑多方观点")
        
        if authority_score > 0.8:
            return f"{base_rec} [权威性较高，建议优先参考]"
        else:
            return f"{base_rec} [权威性一般，建议谨慎判断]"
    
    def _calculate_authority_score(self, text: str) -> float:
        """计算权威性得分 - 分层权重算法"""
        max_score = 0.0
        max_confidence = 0.0
        
        for level, info in self.authority_indicators.items():
            keywords = info['keywords']
            weight = info['weight']
            confidence = info['confidence']
            
            for keyword in keywords:
                if keyword in text:
                    score = weight * confidence
                    if score > max_score:
                        max_score = score
                        max_confidence = confidence
        
        # 默认基础权威性
        if max_score == 0.0:
            max_score = 0.3
            max_confidence = 0.5
        
        return max_score
    
    def _assess_severity(self, text: str) -> Optional[str]:
        """评估严重程度"""
        for severity, words in self.severity_words.items():
            if any(word in text for word in words):
                return severity
        return None
    
    def _generate_reconciled_info(self, contents: List[Dict], 
                                conflicts: List[ConflictDetail]) -> Dict[str, Any]:
        """生成协调信息"""
        reconciled = {
            'summary': '基于多源信息的综合分析',
            'key_points': [],
            'consensus': [],
            'conflicts_summary': [],
            'recommendations': []
        }
        
        if not conflicts:
            reconciled['summary'] = '信息来源一致，无明显冲突'
            return reconciled
        
        # 总结冲突
        conflict_types = [c.conflict_type.value for c in conflicts]
        reconciled['conflicts_summary'] = list(set(conflict_types))
        
        # 收集建议
        recommendations = []
        for conflict in conflicts:
            recommendations.append(conflict.recommendation)
        reconciled['recommendations'] = recommendations
        
        # 生成关键点
        high_authority_contents = [
            c for c in contents 
            if self._calculate_authority_score(c['full_text']) > 0.5
        ]
        
        if high_authority_contents:
            reconciled['key_points'] = [
                f"来源 {c['id']}: {c['answer'][:100]}..." 
                for c in high_authority_contents[:3]
            ]
        
        return reconciled
    
    def _calculate_confidence(self, contents: List[Dict], 
                            conflicts: List[ConflictDetail]) -> float:
        """计算整体置信度"""
        if not conflicts:
            return 0.9
        
        # 基础置信度
        base_confidence = 0.7
        
        # 根据冲突严重程度调整
        high_severity_count = sum(1 for c in conflicts if c.severity == "high")
        medium_severity_count = sum(1 for c in conflicts if c.severity == "medium")
        
        confidence = base_confidence
        confidence -= high_severity_count * 0.2
        confidence -= medium_severity_count * 0.1
        
        return max(confidence, 0.1)
    
    def _generate_safety_warning(self, conflicts: List[ConflictDetail]) -> Optional[str]:
        """生成安全警告"""
        high_severity_conflicts = [c for c in conflicts if c.severity == "high"]
        
        if high_severity_conflicts:
            return "[警告!] 发现高风险冲突信息，强烈建议咨询专业医生"
        
        medium_severity_conflicts = [c for c in conflicts if c.severity == "medium"]
        if len(medium_severity_conflicts) >= 2:
            return "[警告!] 发现多个中等风险冲突，建议谨慎参考"
        
        return None 