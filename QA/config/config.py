import os
from dotenv import load_dotenv

load_dotenv()

# 数据配置
TEST_NUM = 10000 # 测试时使用的数据量

# 科室数据配置 -
# 根据自己垂直领域，可随意替换数据，注意：1、数据列名保持一致，2、改rag_api.py中的prompt
ENABLED_DEPARTMENTS = [
    "Andriatria_男科",  # 男科
    # "Oncology_肿瘤科",  # 肿瘤科 - 已注释，不加载
    # "Pediatric_儿科",   # 儿科 - 已注释，不加载  
    # "Surgical_外科",    # 外科 - 已注释，不加载
]

# BERT模型配置
BERT_MODEL_NAME = "bert-base-chinese"
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32

# Faiss配置
FAISS_INDEX_TYPE = "IVFPQ"
FAISS_NLIST = 30
VECTOR_DIM = 768

# ElasticSearch配置（docker 提前部署）
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX_NAME = "qa_index"
ES_USERNAME = os.getenv("ES_USERNAME", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")

# 混合检索配置
VECTOR_WEIGHT = 0.6
TEXT_WEIGHT = 0.4
TOP_K = 5

# API配置
API_HOST = "0.0.0.0"
API_PORT = 8000
DEBUG = True

# LLM模型配置
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-4B")
LLM_DEVICE = os.getenv("LLM_DEVICE", "cuda")
LLM_MAX_LENGTH = int(os.getenv("LLM_MAX_LENGTH", "2048"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# 对话管理配置
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "20"))
SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT", "3600")) # 1小时

# 上下文感知检索配置
CURRENT_QUERY_WEIGHT = 1.0
CONTEXT_QUERY_WEIGHT = 0.6
HISTORY_QUERY_WEIGHT = 0.4
TOPIC_RELEVANCE_BOOST = 0.2
ENTITY_RELEVANCE_BOOST = 0.3
MAX_CONTEXT_BOOST = 0.5

# 安全检测配置
TOXICITY_DETECTION_ENABLED = True
TOXICITY_THRESHOLD = 0.6

# 数据目录配置
DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
MEDICAL_DATASET_DIR = os.path.join(DATA_DIR, "Chinese-medical-dialogue-data")

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# 缓存配置
CACHE_ENABLED = True
CACHE_MAX_SIZE = 1000
CACHE_TTL = 1800

# 工具函数

def get_config_summary():
 """获取配置摘要"""
 return {
 "api": f"{API_HOST}:{API_PORT}",
 "llm_model": LLM_MODEL_NAME,
 "cache_enabled": CACHE_ENABLED,
 "toxicity_detection": TOXICITY_DETECTION_ENABLED,
 "vector_weight": VECTOR_WEIGHT,
 "text_weight": TEXT_WEIGHT
 } 