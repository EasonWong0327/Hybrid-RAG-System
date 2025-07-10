import uuid
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MessageType(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

@dataclass
class Message:
    type: MessageType
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class ConversationContext:
    session_id: str
    user_id: Optional[str] = None
    current_topic: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    intent_history: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

class DialogueManager:
    def __init__(self, max_history: int = 10, session_timeout: int = 3600):
        self.max_history = max_history
        self.session_timeout = session_timeout
        self.sessions: Dict[str, List[Message]] = {}
        self.contexts: Dict[str, ConversationContext] = {}

    def create_session(self, user_id: Optional[str] = None) -> str:
        """创建新会话"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        self.contexts[session_id] = ConversationContext(
            session_id=session_id,
            user_id=user_id
        )
        logger.info(f"创建会话: {session_id}")
        return session_id

    def add_message(self, session_id: str, message_type: MessageType, content: str):
        """添加消息到会话"""
        if session_id not in self.sessions:
            session_id = self.create_session()
        
        message = Message(type=message_type, content=content)
        self.sessions[session_id].append(message)
        
        # 限制历史长度
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
        
        # 更新活跃时间
        if session_id in self.contexts:
            self.contexts[session_id].last_active = time.time()

    def get_conversation_history(self, session_id: str, max_turns: int = None) -> List[Message]:
        """获取会话历史"""
        if session_id not in self.sessions:
            return []
        
        messages = self.sessions[session_id]
        if max_turns:
            return messages[-max_turns:]
        return messages

    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """获取会话上下文"""
        return self.contexts.get(session_id)

    def update_context(self, session_id: str, **kwargs):
        """更新会话上下文"""
        if session_id in self.contexts:
            context = self.contexts[session_id]
            for key, value in kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)
            context.last_active = time.time()

    def clear_session(self, session_id: str):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.contexts:
            del self.contexts[session_id]
        logger.info(f"清除会话: {session_id}")

    def cleanup_expired_sessions(self):
        """清理过期会话"""
        current_time = time.time()
        expired_sessions = []
        
        for session_id, context in self.contexts.items():
            if current_time - context.last_active > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.clear_session(session_id)
        
        if expired_sessions:
            logger.info(f"清理过期会话: {len(expired_sessions)}")

    def get_session_count(self) -> int:
        """获取活跃会话数量"""
        return len(self.sessions)

    def format_conversation_for_model(self, session_id: str, max_turns: int = 5) -> str:
        """格式化对话历史供模型使用"""
        messages = self.get_conversation_history(session_id, max_turns)
        
        formatted = []
        for msg in messages:
            role = "用户" if msg.type == MessageType.USER else "助手"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted) 