"""
核心模块包
包含检索器、生成器、检测器和对话管理等核心功能
"""

__version__ = "1.0.0"

# 导入主要组件
from .retrievers.adaptive_hybrid_retriever import AdaptiveHybridRetriever
from .detectors.knowledge_conflict_detector import KnowledgeConflictDetector
from .dialogue.query_analyzer import QueryAnalyzer
from .generators.qwen3_generator import Qwen3Generator
from .dialogue.dialogue_manager import DialogueManager

__all__ = [
    'AdaptiveHybridRetriever',
    'KnowledgeConflictDetector', 
    'QueryAnalyzer',
    'Qwen3Generator',
    'DialogueManager'
] 