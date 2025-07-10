from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import time
from datetime import datetime


from core.retrievers.adaptive_hybrid_retriever import AdaptiveHybridRetriever
from core.generators.qwen3_generator import Qwen3Generator
from core.dialogue.dialogue_manager import DialogueManager, MessageType
from core.dialogue.smart_dst import SmartDST
from config.config import *
from core.detectors.toxicity_detector import ToxicityDetector
from core.detectors.hallucination_detector import EnhancedHallucinationDetector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic模型
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    use_context: bool = True
    max_tokens: int = 512

class SessionRequest(BaseModel):
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    toxicity: Dict[str, Any]
    hallucination: Dict[str, Any]
    query_analysis: Dict[str, Any]
    conflict_analysis: Dict[str, Any]
    response_time: float
    timestamp: str


app = FastAPI(
    title="混合RAG系统",
    description="基于Qwen3-4B的医疗问答RAG系统",
    version="1.0.0"
)

# 全局组件init
adaptive_retriever = None
qwen3_generator = None
dialogue_manager = None
smart_dst = None
toxicity_detector = None
hallucination_detector = None

def initialize_system():
    """初始化系统组件"""
    global adaptive_retriever, qwen3_generator, dialogue_manager, smart_dst, toxicity_detector, hallucination_detector

    try:
        logger.info("初始化RAG系统...")

        # 加载数据
        from utils.data_get import QAPairs
        qa_loader = QAPairs()
        qa_pairs = qa_loader.get_all_qa_pairs()
        logger.info(f"加载数据: {len(qa_pairs)} 条")

        # 初始化组件
        adaptive_retriever = AdaptiveHybridRetriever()
        adaptive_retriever.initialize(qa_pairs)
        
        qwen3_generator = Qwen3Generator()
        
        dialogue_manager = DialogueManager(
            max_history=MAX_HISTORY_TURNS,
            session_timeout=SESSION_TIMEOUT
        )
        
        smart_dst = SmartDST()
        toxicity_detector = ToxicityDetector()
        hallucination_detector = EnhancedHallucinationDetector()

        logger.info("系统初始化完成")

    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        raise

# 依赖注入
def get_adaptive_retriever() -> AdaptiveHybridRetriever:
    if adaptive_retriever is None:
        raise HTTPException(status_code=500, detail="检索器未初始化")
    return adaptive_retriever

def get_qwen3_generator() -> Qwen3Generator:
    if qwen3_generator is None:
        raise HTTPException(status_code=500, detail="生成器未初始化")
    return qwen3_generator

def get_dialogue_manager() -> DialogueManager:
    if dialogue_manager is None:
        raise HTTPException(status_code=500, detail="对话管理器未初始化")
    return dialogue_manager

def get_smart_dst() -> SmartDST:
    if smart_dst is None:
        raise HTTPException(status_code=500, detail="DST未初始化")
    return smart_dst

def get_toxicity_detector() -> ToxicityDetector:
    if toxicity_detector is None:
        raise HTTPException(status_code=500, detail="毒性检测器未初始化")
    return toxicity_detector

def get_hallucination_detector() -> EnhancedHallucinationDetector:
    if hallucination_detector is None:
        raise HTTPException(status_code=500, detail="幻觉检测器未初始化")
    return hallucination_detector


# API

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    initialize_system()

@app.get("/")
async def root():
    """API根路径"""
    return {
        "message": "Hybrid-RAG系统API",
        "version": "1.0.0",
        "status": "running",
        "config": get_config_summary()
    }

@app.post("/sessions", response_model=Dict[str, str])
async def create_session(
    request: SessionRequest,
    dm: DialogueManager = Depends(get_dialogue_manager)
):

    try:
        session_id = dm.create_session(user_id=request.user_id)
        return {"session_id": session_id, "status": "created"}
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        raise HTTPException(status_code=500, detail="创建会话失败")

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: AdaptiveHybridRetriever = Depends(get_adaptive_retriever),
    generator: Qwen3Generator = Depends(get_qwen3_generator),
    dm: DialogueManager = Depends(get_dialogue_manager),
    dst: SmartDST = Depends(get_smart_dst),
    hallucination_det: EnhancedHallucinationDetector = Depends(get_hallucination_detector)
):
    """处理用户查询"""
    start_time = time.time()

    try:
        # 会话管理
        session_id = request.session_id
        if not session_id:
            session_id = dm.create_session()

        # 添加用户消息
        dm.add_message(session_id, MessageType.USER, request.query)

        # DST处理
        conversation_history = dm.get_conversation_history(session_id, max_turns=5)
        context = dm.get_context(session_id)
        enhanced_query, dst_result = dst.process_turn(request.query, conversation_history, context)

        # 更新上下文
        dm.update_context(
            session_id,
            current_topic=dst_result.get("topic"),
            entities=dst_result.get("entities", {}),
            medical_context=dst_result.get("medical_context", {})
        )

        # 混合检索
        retrieval_context = {}
        if request.use_context:
            context = dm.get_context(session_id)
            if context:
                retrieval_context = {
                    "current_topic": context.current_topic,
                    "entities": dict(context.entities),
                    "intent_history": context.intent_history[-3:]  # 最近3个意图，自己看着改
                }

        # 执行自适应检索
        retrieval_result = retriever.search(
            query=enhanced_query,
            top_k=TOP_K,
            context=retrieval_context
        )

        # 获取检索结果
        results = retrieval_result.get('results', [])
        query_analysis = retrieval_result.get('query_analysis', {})
        conflict_analysis = retrieval_result.get('conflict_analysis', {})

        # 构建提示词
        context_str = "\n".join([
            f"[相关内容{i+1}] {result['question']}: {result['answer']}"
            for i, result in enumerate(results[:3])
        ])

        # 根据冲突分析调整提示词
        conflict_warning = ""
        if hasattr(conflict_analysis, 'has_conflicts') and conflict_analysis.has_conflicts:
            conflict_warning = f"\n\n[警告] 检测到信息冲突：{conflict_analysis.safety_warning or ''}"
            if hasattr(conflict_analysis, 'reconciled_info') and conflict_analysis.reconciled_info.get('recommendations'):
                conflict_warning += f"\n建议：{'; '.join(conflict_analysis.reconciled_info['recommendations'][:2])}"

        # 自由改，根据自己来
        prompt = f"""基于以下医学知识回答用户问题：

{context_str}{conflict_warning}

用户问题：{request.query}

查询分析：类型={query_analysis.get('query_type', 'unknown')}, 置信度={query_analysis.get('confidence', 0):.2f}

请根据提供的医学知识给出专业、准确的回答。如果检测到信息冲突，请在回答中说明并提供平衡的观点。如果知识库中没有相关信息，请说明这一点。"""

        # 生成回答（启用概率分布检测）
        generation_result = generator.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            return_probabilities=True
        )

        answer = generation_result["response"]
        toxicity = generation_result["toxicity"]
        probability_info = generation_result.get("probability_info")

        # 幻觉检测
        source_contexts = [result['answer'] for result in results[:3]]
        hallucination_result = hallucination_det.detect(
            generated_text=answer,
            source_contexts=source_contexts,
            query=request.query,
            probability_info=probability_info
        )

        # 添加系统回答
        dm.add_message(session_id, MessageType.ASSISTANT, answer)

        # 响应时间
        response_time = time.time() - start_time

        # 转换conflict_analysis为字典格式
        conflict_analysis_dict = {}
        if hasattr(conflict_analysis, 'has_conflicts'):
            conflict_analysis_dict = {
                "has_conflicts": conflict_analysis.has_conflicts,
                "conflicts": [
                    {
                        "type": conflict.conflict_type.value,
                        "description": conflict.description,
                        "severity": conflict.severity,
                        "recommendation": conflict.recommendation
                    } for conflict in conflict_analysis.conflicts
                ],
                "reconciled_info": conflict_analysis.reconciled_info,
                "confidence_score": conflict_analysis.confidence_score,
                "safety_warning": conflict_analysis.safety_warning
            }

        return QueryResponse(
            answer=answer,
            sources=[{
                "question": result['question'],
                "answer": result['answer'],
                "similarity": result['similarity'],
                "source": result.get('source', 'unknown')
            } for result in results[:3]],
            session_id=session_id,
            toxicity={
                "is_toxic": toxicity.get("is_toxic", False),
                "score": toxicity.get("confidence", 0.0),
                "categories": toxicity.get("detected_categories", []),
                "toxicity_type": toxicity.get("toxicity_type", "正常内容"),
                "severity": toxicity.get("severity", "none")
            },
            hallucination={
                "is_hallucination": hallucination_result.is_hallucination,
                "confidence": hallucination_result.confidence,
                "evidence": hallucination_result.evidence[:3],
                "suggestions": hallucination_result.suggestions[:3]
            },
            query_analysis=query_analysis,
            conflict_analysis=conflict_analysis_dict,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"查询处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str,
    dm: DialogueManager = Depends(get_dialogue_manager)
):
    """获取会话信息"""
    try:
        context = dm.get_context(session_id)
        history = dm.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "context": context.__dict__ if context else None,
            "history": [{"type": msg.type.value, "content": msg.content} for msg in history],
            "message_count": len(history)
        }
    except Exception as e:
        logger.error(f"获取会话信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class HallucinationCheckRequest(BaseModel):
    text: str
    source_contexts: Optional[List[str]] = None
    query: Optional[str] = None


@app.post("/check_hallucination")
async def check_hallucination(
    request: HallucinationCheckRequest,
    hallucination_det: EnhancedHallucinationDetector = Depends(get_hallucination_detector)
):
    """独立幻觉检测接口，这个我会单独写一个更详细的project，来检测幻觉"""
    try:
        result = hallucination_det.detect(
            generated_text=request.text,
            source_contexts=request.source_contexts,
            query=request.query
        )
        return {
            "is_hallucination": result.is_hallucination,
            "confidence": result.confidence,
            "hallucination_type": result.hallucination_type.value if result.hallucination_type else None,
            "evidence": result.evidence,
            "suggestions": result.suggestions,
            "score_breakdown": result.score_breakdown
        }
    except Exception as e:
        logger.error(f"幻觉检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查组件状态
        components = {
            "adaptive_retriever": adaptive_retriever is not None,
            "qwen3_generator": qwen3_generator is not None,
            "dialogue_manager": dialogue_manager is not None,
            "smart_dst": smart_dst is not None,
            "toxicity_detector": toxicity_detector is not None,
            "hallucination_detector": hallucination_detector is not None
        }
        
        all_ready = all(components.values())
        
        return {
            "status": "healthy" if all_ready else "initializing",
            "components": components,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {"status": "error", "error": str(e)}



def run_app():
    """运行应用"""
    logger.info("启动混合RAG系统...")
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    ) 