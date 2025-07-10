#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 解决OpenMP冲突问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
混合RAG系统 - 主启动文件
基于Qwen3-4B的医疗问答系统
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.rag_api import run_app
from config.config import get_config_summary

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """打印启动横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║ 混合RAG系统 ║
    ║ 基于Qwen3-4B的医疗问答系统 ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """检查关键依赖"""
    try:
        import torch
        import faiss
        import elasticsearch
        import transformers
        logger.info("依赖检查通过")
        return True
    except ImportError as e:
        logger.error(f"[错误] 缺少依赖: {e}")
        return False

def main():
    """主函数"""
    print_banner()

    # 检查依赖
    if not check_dependencies():
        logger.error("请安装所需依赖后重试")
        sys.exit(1)

    # 显示配置信息
    config = get_config_summary()
    logger.info(" 系统配置:")
    for key, value in config.items():
        logger.info(f" {key}: {value}")

    # 启动API服务
    try:
        logger.info(" 启动简化版RAG系统...")
        run_app()
    except KeyboardInterrupt:
        logger.info(" 系统正常退出")
    except Exception as e:
        logger.error(f"[错误] 系统启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 