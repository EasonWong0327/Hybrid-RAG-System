import re
import jieba
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class IntentType(Enum):
    """意图类型枚举"""
    SYMPTOM_INQUIRY = "symptom_inquiry"
    DIAGNOSIS_REQUEST = "diagnosis_request" 
    TREATMENT_INQUIRY = "treatment_inquiry"
    PREVENTION = "prevention"
    GENERAL_INFO = "general_info"
    FOLLOW_UP = "follow_up"
    CLARIFICATION = "clarification"

@dataclass
class EntityMention:
    """实体提及"""
    text: str
    entity_type: str
    confidence: float
    start_pos: int = 0
    end_pos: int = 0
    synonyms: List[str] = field(default_factory=list)

@dataclass
class DialogueSlot:
    """对话槽位"""
    name: str
    value: Any = None
    confidence: float = 0.0
    is_confirmed: bool = False
    update_turn: int = 0

@dataclass
class DialogueState:
    """对话状态"""
    intent: IntentType = IntentType.GENERAL_INFO
    entities: Dict[str, List[EntityMention]] = field(default_factory=dict)
    slots: Dict[str, DialogueSlot] = field(default_factory=dict)
    context_stack: deque = field(default_factory=lambda: deque(maxlen=5))
    turn_count: int = 0
    confidence: float = 0.0

class SmartDST:
    """智能对话状态跟踪器  NER+意图理解+状态管理"""
    
    def __init__(self):
        # 医疗实体词典，公司会加载数据库
        self.medical_entities = {
            'symptoms': {
                '头痛': ['头疼', '脑袋疼', '头部疼痛', '偏头痛'],
                '发烧': ['发热', '体温升高', '低烧', '高烧', '体温异常'],
                '咳嗽': ['咳痰', '干咳', '咳嗽不止', '咳咳'],
                '乏力': ['无力', '疲劳', '疲倦', '没精神', '浑身无力'],
                '失眠': ['睡不着', '入睡困难', '多梦', '早醒'],
                '腹痛': ['肚子疼', '胃疼', '腹部疼痛', '肚痛'],
                '头晕': ['眩晕', '头昏', '头昏眼花', '晕眩'],
                '恶心': ['想吐', '反胃', '犯恶心'],
                '胸痛': ['胸闷', '胸部疼痛', '心口疼'],
                '呼吸困难': ['气短', '喘不过气', '呼吸急促', '憋气']
            },
            'diseases': {
                '高血压': ['血压高', '高压', '血压升高'],
                '糖尿病': ['血糖高', '糖尿', 'DM', '血糖异常'],
                '感冒': ['风寒', '感冒发烧', '上呼吸道感染', '伤风'],
                '肺炎': ['肺部感染', '支气管肺炎', '肺部炎症'],
                '胃炎': ['胃病', '胃部炎症', '急性胃炎', '慢性胃炎'],
                '心脏病': ['心脏疾病', '心血管疾病', '冠心病']
            },
            'body_parts': {
                '头部': ['脑袋', '头', '颅部'],
                '胸部': ['胸腔', '胸口', '前胸'],
                '腹部': ['肚子', '腹腔', '小腹', '上腹', '下腹'],
                '背部': ['后背', '脊背', '腰背'],
                '心脏': ['心脏部位', '心口'],
                '肺部': ['肺', '肺脏', '呼吸系统']
            },
            'treatments': {
                '药物治疗': ['吃药', '用药', '服药', '药疗'],
                '手术': ['开刀', '手术治疗', '外科手术'],
                '物理治疗': ['理疗', '康复理疗', '物理疗法'],
                '运动疗法': ['锻炼', '运动康复', '体育疗法']
            },
            'medications': {
                '阿司匹林': ['拜阿司匹灵', 'ASA'],
                '青霉素': ['盘尼西林', '青霉'],
                '头孢': ['头孢菌素', '先锋霉素'],
                '降压药': ['血管紧张素', 'ACEI', 'ARB']
            }
        }
        
        # 槽位定义 对话状态跟踪的核心（根据自己来）
        self.slot_definitions = {
            'main_symptom': DialogueSlot('主要症状'),
            'symptom_duration': DialogueSlot('症状持续时间'),
            'symptom_severity': DialogueSlot('症状严重程度'),
            'affected_body_part': DialogueSlot('受影响部位'),
            'suspected_disease': DialogueSlot('疑似疾病'),
            'current_medications': DialogueSlot('当前用药'),
            'medical_history': DialogueSlot('既往病史'),
            'age_group': DialogueSlot('年龄组'),
            'gender': DialogueSlot('性别'),
            'urgency_level': DialogueSlot('紧急程度')
        }
        
        # TF-IDF 意图分类（简单来）
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # 中文没有标准停用词
            ngram_range=(1, 2)
        )
        
        # 意图分类训练数据（实际应用中应该用更大的数据集）
        self.intent_training_data = {
            IntentType.SYMPTOM_INQUIRY: [
                "我头痛怎么办", "感觉胸闷", "最近总是咳嗽", "肚子疼得厉害",
                "头晕恶心", "发烧了", "失眠睡不着", "乏力无精神"
            ],
            IntentType.DIAGNOSIS_REQUEST: [
                "这是什么病", "我得的是什么病", "能帮我诊断一下吗", "这些症状是什么疾病",
                "我可能得了什么病", "这种情况严重吗", "需要检查什么"
            ],
            IntentType.TREATMENT_INQUIRY: [
                "怎么治疗", "用什么药", "如何缓解", "治疗方案", "吃什么药好",
                "怎样才能好", "有什么办法", "如何用药"
            ],
            IntentType.PREVENTION: [
                "如何预防", "怎么避免", "注意什么", "预防措施", "保健方法",
                "日常护理", "生活注意事项"
            ],
            IntentType.FOLLOW_UP: [
                "还有其他症状", "另外", "还想问", "补充一下", "忘了说",
                "对了", "还有就是"
            ]
        }
        
        # 初始化意图分类器
        self._initialize_intent_classifier()
        
        # 对话状态管理
        self.dialogue_states = {}  # session_id -> DialogueState
        
        # 实体链接和归一化
        self.entity_synonyms = self._build_entity_synonyms()
        
        logger.info("DST系统初始化")
    
    def _initialize_intent_classifier(self):
        """初始化意图分类器"""
        try:
            # 准备训练数据
            texts = []
            labels = []
            
            for intent_type, examples in self.intent_training_data.items():
                texts.extend(examples)
                labels.extend([intent_type] * len(examples))
            
            # 训练TF-IDF
            self.tfidf_vectorizer.fit(texts)
            
            # 为每个意图计算中心向量
            self.intent_centroids = {}
            for intent_type in IntentType:
                intent_texts = self.intent_training_data.get(intent_type, [])
                if intent_texts:
                    vectors = self.tfidf_vectorizer.transform(intent_texts)
                    centroid = np.mean(vectors.toarray(), axis=0)
                    self.intent_centroids[intent_type] = centroid
                    
            logger.info(f"意图分类器初始化: {len(self.intent_centroids)}种意图")
            
        except Exception as e:
            logger.error(f"意图分类器初始化失败: {e}")
            # 降级为规则模式
            self.use_ml_intent = False
    
    def _build_entity_synonyms(self) -> Dict[str, str]:
        """构建实体同义词映射"""
        synonyms = {}
        
        for entity_type, entities in self.medical_entities.items():
            for main_entity, synonym_list in entities.items():
                # 主实体映射到自己
                synonyms[main_entity] = main_entity
                # 同义词映射到主实体
                for synonym in synonym_list:
                    synonyms[synonym] = main_entity
                    
        return synonyms

    def process_turn(self, query: str, conversation_history: List = None, context: Any = None) -> tuple:
        """处理对话轮次 - 完整的DST状态跟踪"""
        try:
            # 获取或创建对话状态
            session_id = getattr(context, 'session_id', 'default') if context else 'default'
            dialogue_state = self._get_or_create_dialogue_state(session_id)
            
            # 更新轮次计数
            dialogue_state.turn_count += 1
            
            # 高级实体识别
            entities = self._advanced_entity_recognition(query)
            
            # 机器学习意图识别
            intent = self._ml_intent_recognition(query, dialogue_state)
            
            # 更新对话状态
            dialogue_state = self._update_dialogue_state(
                dialogue_state, query, entities, intent, conversation_history
            )
            
            # 槽位填充
            dialogue_state = self._slot_filling(dialogue_state, entities, query)
            
            # 生成增强查询
            enhanced_query = self._generate_enhanced_query(
                query, dialogue_state, conversation_history
            )
            
            # 构建返回结果
            dst_result = {
                'entities': self._format_entities_output(entities),
                'intent': intent.value,
                'dialogue_state': dialogue_state,
                'filled_slots': {k: v.value for k, v in dialogue_state.slots.items() if v.value is not None},
                'confidence': dialogue_state.confidence,
                'turn_count': dialogue_state.turn_count,
                'context_summary': self._generate_context_summary(dialogue_state)
            }
            
            # 保存状态
            self.dialogue_states[session_id] = dialogue_state
            
            return enhanced_query, dst_result
            
        except Exception as e:
            logger.error(f"DST处理失败: {e}")
            return query, {
                'entities': {},
                'intent': IntentType.GENERAL_INFO.value,
                'dialogue_state': None,
                'confidence': 0.3,
                'error': str(e)
            }

    def _get_or_create_dialogue_state(self, session_id: str) -> DialogueState:
        """获取或创建对话状态"""
        if session_id not in self.dialogue_states:
            self.dialogue_states[session_id] = DialogueState()
        return self.dialogue_states[session_id]
    
    def _advanced_entity_recognition(self, query: str) -> Dict[str, List[EntityMention]]:
        """高级实体识别 - 支持同义词和上下文"""
        entities = defaultdict(list)
        
        # jiba
        words = list(jieba.cut(query))
        text = "".join(words)  # 重新组合用于位置计算
        
        for entity_type, entity_dict in self.medical_entities.items():
            for main_entity, synonyms in entity_dict.items():
                # 检查主实体
                all_variants = [main_entity] + synonyms
                
                for variant in all_variants:
                    if variant in query:
                        start_pos = query.find(variant)
                        end_pos = start_pos + len(variant)
                        
                        # 计算置信度
                        confidence = 1.0 if variant == main_entity else 0.8
                        
                        entity_mention = EntityMention(
                            text=variant,
                            entity_type=entity_type,
                            confidence=confidence,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            synonyms=synonyms
                        )
                        
                        entities[entity_type].append(entity_mention)
                        break  # 避免重复匹配
        
        return dict(entities)
    
    def _ml_intent_recognition(self, query: str, dialogue_state: DialogueState) -> IntentType:
        """ml意图识别"""
        try:
            # 向量化查询
            query_vector = self.tfidf_vectorizer.transform([query]).toarray()[0]
            
            # 计算与各意图中心的相似度
            similarities = {}
            for intent_type, centroid in self.intent_centroids.items():
                similarity = cosine_similarity([query_vector], [centroid])[0][0]
                similarities[intent_type] = similarity
            
            # 考虑对话上下文
            if dialogue_state.turn_count > 1:
                # 如果是后续轮次，可能是澄清或补充
                follow_up_indicators = ['还有', '另外', '补充', '对了', '忘了说']
                if any(indicator in query for indicator in follow_up_indicators):
                    similarities[IntentType.FOLLOW_UP] *= 1.5
            
            # 返回最高相似度的意图
            best_intent = max(similarities.keys(), key=lambda k: similarities[k])
            
            # 置信度阈值检查
            if similarities[best_intent] < 0.3:
                return IntentType.GENERAL_INFO
                
            return best_intent
            
        except Exception as e:
            logger.error(f"意图识别失败: {e}")
            return IntentType.GENERAL_INFO
    
    def _update_dialogue_state(self, state: DialogueState, query: str, 
                              entities: Dict, intent: IntentType, 
                              history: List = None) -> DialogueState:
        """更新对话状态"""
        state.intent = intent
        
        # 更新实体信息
        for entity_type, entity_list in entities.items():
            if entity_list:
                state.entities[entity_type] = entity_list
        
        # 添加到上下文栈
        state.context_stack.append({
            'query': query,
            'intent': intent,
            'entities': entities,
            'turn': state.turn_count
        })
        
        # 计算整体置信度
        entity_confidence = np.mean([
            np.mean([e.confidence for e in entity_list]) 
            for entity_list in entities.values() if entity_list
        ]) if entities else 0.5
        
        state.confidence = (entity_confidence + 0.8) / 2  # 结合实体和意图置信度
        
        return state
    
    def _slot_filling(self, state: DialogueState, entities: Dict, query: str) -> DialogueState:
        """槽位填充算法"""
        current_turn = state.turn_count
        
        # 根据实体类型填充槽位
        if 'symptoms' in entities and entities['symptoms']:
            main_symptom = entities['symptoms'][0].text  # 取第一个症状作为主症状
            state.slots['main_symptom'] = DialogueSlot(
                name='main_symptom',
                value=main_symptom,
                confidence=entities['symptoms'][0].confidence,
                is_confirmed=True,
                update_turn=current_turn
            )
        
        if 'diseases' in entities and entities['diseases']:
            suspected_disease = entities['diseases'][0].text
            state.slots['suspected_disease'] = DialogueSlot(
                name='suspected_disease',
                value=suspected_disease,
                confidence=entities['diseases'][0].confidence,
                is_confirmed=False,  # 疾病需要确认
                update_turn=current_turn
            )
        
        if 'body_parts' in entities and entities['body_parts']:
            affected_part = entities['body_parts'][0].text
            state.slots['affected_body_part'] = DialogueSlot(
                name='affected_body_part',
                value=affected_part,
                confidence=entities['body_parts'][0].confidence,
                is_confirmed=True,
                update_turn=current_turn
            )
        
        # 时间模式识别
        duration_patterns = [
            (r'(\d+)天', '天'),
            (r'(\d+)周', '周'),
            (r'(\d+)个月', '月'),
            (r'最近|近期', '近期'),
            (r'一直|总是', '持续')
        ]
        
        for pattern, unit in duration_patterns:
            if re.search(pattern, query):
                match = re.search(pattern, query)
                if match:
                    if match.groups():
                        duration = f"{match.group(1)}{unit}"
                    else:
                        duration = unit
                    
                    state.slots['symptom_duration'] = DialogueSlot(
                        name='symptom_duration',
                        value=duration,
                        confidence=0.8,
                        is_confirmed=True,
                        update_turn=current_turn
                    )
                    break
        
        # 严重程度识别
        severity_patterns = {
            '轻微': ['轻微', '稍微', '一点点', '不严重'],
            '中等': ['比较', '有点', '中等'],
            '严重': ['严重', '剧烈', '厉害', '难受']
        }
        
        for severity, keywords in severity_patterns.items():
            if any(keyword in query for keyword in keywords):
                state.slots['symptom_severity'] = DialogueSlot(
                    name='symptom_severity',
                    value=severity,
                    confidence=0.7,
                    is_confirmed=True,
                    update_turn=current_turn
                )
                break
        
        return state
    
    def _generate_enhanced_query(self, query: str, state: DialogueState, 
                                history: List = None) -> str:
        """生成增强查询"""
        enhancements = [query]
        
        # 添加槽位信息
        filled_slots = [slot for slot in state.slots.values() if slot.value is not None]
        if filled_slots:
            slot_info = []
            for slot in filled_slots:
                slot_info.append(f"{slot.name}:{slot.value}")
            
            if slot_info:
                enhancements.append(f"相关信息: {', '.join(slot_info)}")
        
        # 添加对话历史中的重要信息
        if len(state.context_stack) > 1:
            prev_entities = []
            for ctx in list(state.context_stack)[-3:-1]:  # 最近2轮
                for entity_type, entity_list in ctx.get('entities', {}).items():
                    for entity in entity_list:
                        prev_entities.append(entity.text)
            
            if prev_entities:
                enhancements.append(f"历史提及: {', '.join(set(prev_entities))}")
        
        return " | ".join(enhancements)
    
    def _format_entities_output(self, entities: Dict[str, List[EntityMention]]) -> Dict:
        """格式化实体输出"""
        formatted = {}
        for entity_type, entity_list in entities.items():
            formatted[entity_type] = [
                {
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'synonyms': entity.synonyms
                }
                for entity in entity_list
            ]
        return formatted
    
    def _generate_context_summary(self, state: DialogueState) -> str:
        """生成上下文摘要"""
        summary_parts = []
        
        # 主要症状
        if 'main_symptom' in state.slots and state.slots['main_symptom'].value:
            summary_parts.append(f"主症状: {state.slots['main_symptom'].value}")
        
        # 持续时间
        if 'symptom_duration' in state.slots and state.slots['symptom_duration'].value:
            summary_parts.append(f"持续: {state.slots['symptom_duration'].value}")
        
        # 严重程度
        if 'symptom_severity' in state.slots and state.slots['symptom_severity'].value:
            summary_parts.append(f"程度: {state.slots['symptom_severity'].value}")
        
        # 当前意图
        summary_parts.append(f"意图: {state.intent.value}")
        
        return " | ".join(summary_parts) if summary_parts else "新对话"

    def _recognize_intent(self, query: str) -> str:
        """识别意图"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern in query:
                    score += 1
            intent_scores[intent] = score
        
        # 返回得分最高的意图
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return 'general_info'

    def _track_topic(self, query: str, entities: Dict, context: Any) -> str:
        """跟踪主题"""
        # 简化的主题跟踪
        if entities['diseases']:
            return entities['diseases'][0]
        elif entities['symptoms']:
            return f"症状_{entities['symptoms'][0]}"
        elif entities['body_parts']:
            return f"部位_{entities['body_parts'][0]}"
        
        return "一般咨询"

    def _build_medical_context(self, entities: Dict, intent: str, context: Any) -> Dict:
        """构建医疗上下文"""
        medical_context = {
            'primary_concern': None,
            'related_symptoms': [],
            'target_body_part': None,
            'treatment_preference': None
        }
        
        # 主要关注点
        if entities['diseases']:
            medical_context['primary_concern'] = entities['diseases'][0]
        elif entities['symptoms']:
            medical_context['primary_concern'] = entities['symptoms'][0]
        
        # 相关症状
        medical_context['related_symptoms'] = entities['symptoms']
        
        # 目标身体部位
        if entities['body_parts']:
            medical_context['target_body_part'] = entities['body_parts'][0]
        
        # 治疗偏好
        if entities['treatments']:
            medical_context['treatment_preference'] = entities['treatments'][0]
        
        return medical_context

    def _enhance_query(self, query: str, entities: Dict, intent: str, context: Any) -> str:
        """增强查询"""
        enhanced_parts = [query]
        
        # 根据意图添加上下文
        if intent == 'symptom_inquiry' and entities['symptoms']:
            enhanced_parts.append(f"症状相关: {', '.join(entities['symptoms'])}")
        
        elif intent == 'treatment_inquiry' and entities['diseases']:
            enhanced_parts.append(f"治疗相关: {', '.join(entities['diseases'])}")
        
        # 添加身体部位信息
        if entities['body_parts']:
            enhanced_parts.append(f"部位: {', '.join(entities['body_parts'])}")
        
        return " ".join(enhanced_parts)

    def get_conversation_state(self, session_id: str) -> Dict:
        """获取对话状态"""
        return {
            'session_id': session_id,
            'status': 'active',
            'last_intent': 'general_info',
            'active_entities': {},
            'conversation_flow': []
        } 