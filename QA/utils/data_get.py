import os
import pandas as pd
import logging
from glob import glob
from typing import List, Dict, Any
from config.config import ENABLED_DEPARTMENTS, TEST_NUM

logger = logging.getLogger(__name__)

class QAPairs:
    def __init__(self):
        self.data_dir = "data/Chinese-medical-dialogue-data"
        self.data = None
        self.qa_pairs = []

    def load_data(self):
        """加载医疗对话数据"""
        try:
            if not os.path.exists(self.data_dir):
                logger.warning(f"数据目录不存在: {self.data_dir}")
                return []

            # 根据配置加载指定科室的数据
            csv_files = []
            for department in ENABLED_DEPARTMENTS:
                dept_files = glob(os.path.join(self.data_dir, f"{department}/*.csv"))
                csv_files.extend(dept_files)
            
            logger.info(f"找到 {len(csv_files)} 个数据文件，加载的科室: {ENABLED_DEPARTMENTS}")

            all_data = []
            for file_path in csv_files:
                try:
                    # 尝试不同的编码
                    for encoding in ['utf-8', 'gbk', 'gb2312']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding)
                            logger.info(f"加载文件: {file_path}")
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        logger.error(f"无法读取文件: {file_path}")
                        continue

                    # 处理数据
                    if 'ask' in df.columns and 'answer' in df.columns:
                        for _, row in df.iterrows():
                            if pd.notna(row['ask']) and pd.notna(row['answer']):
                                qa_pair = {
                                    'question': str(row['ask']).strip(),
                                    'answer': str(row['answer']).strip(),
                                    'department': self._extract_department(file_path),
                                    'source': os.path.basename(file_path)
                                }
                                all_data.append(qa_pair)

                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {e}")

            logger.info(f"总共加载了 {len(all_data)} 条问答数据")
            
            # 限制数据量
            if TEST_NUM and len(all_data) > TEST_NUM:
                all_data = all_data[:TEST_NUM]
                logger.info(f"根据TEST_NUM配置，只保留前 {TEST_NUM} 条数据")
            
            self.qa_pairs = all_data
            return all_data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return []

    def _extract_department(self, file_path: str) -> str:
        """从文件路径提取科室信息"""
        try:
            # 从路径中提取科室名称
            parts = file_path.split(os.sep)
            for part in parts:
                if '_' in part and any(char in part for char in ['科', '内', '外', '儿', '妇']):
                    return part.split('_')[1] if '_' in part else part
            return "未知科室"
        except:
            return "未知科室"

    def get_all_qa_pairs(self) -> List[Dict[str, Any]]:
        """获取所有问答对"""
        if not self.qa_pairs:
            self.load_data()
        return self.qa_pairs

    def get_qa_pairs_by_department(self, department: str) -> List[Dict[str, Any]]:
        """根据科室获取问答对"""
        if not self.qa_pairs:
            self.load_data()
        
        return [qa for qa in self.qa_pairs if department in qa.get('department', '')]

    def get_sample_data(self, n: int = 100) -> List[Dict[str, Any]]:
        """获取样本数据"""
        if not self.qa_pairs:
            self.load_data()
        
        return self.qa_pairs[:n] if len(self.qa_pairs) >= n else self.qa_pairs

    def search_qa_pairs(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索包含关键词的问答对"""
        if not self.qa_pairs:
            self.load_data()
        
        results = []
        for qa in self.qa_pairs:
            if keyword in qa['question'] or keyword in qa['answer']:
                results.append(qa)
        
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.qa_pairs:
            self.load_data()
        
        departments = {}
        for qa in self.qa_pairs:
            dept = qa.get('department', '未知科室')
            departments[dept] = departments.get(dept, 0) + 1
        
        return {
            'total_qa_pairs': len(self.qa_pairs),
            'departments': departments,
            'avg_question_length': sum(len(qa['question']) for qa in self.qa_pairs) / len(self.qa_pairs) if self.qa_pairs else 0,
            'avg_answer_length': sum(len(qa['answer']) for qa in self.qa_pairs) / len(self.qa_pairs) if self.qa_pairs else 0
        } 