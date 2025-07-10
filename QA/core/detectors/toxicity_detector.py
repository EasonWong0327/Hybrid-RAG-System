import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
import joblib
import os
import json
from typing import List, Dict, Any, Tuple, Optional
import logging

class ToxicityClassifier(nn.Module):
    """有毒内容分类器,这个比较简单，一个多分类模型，可以私下训练好，然后加载进来"""
    
    def __init__(self, bert_model_name: str = "bert-base-chinese", 
                 num_classes: int = 6, dropout: float = 0.3):
        super(ToxicityClassifier, self).__init__()
        
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # 冻结BERT参数的前几层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in self.bert.encoder.layer[:8]:  # 冻结前8层
            for param in layer.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        return logits

class ToxicityDetector:
    """有毒内容检测器"""
    
    def __init__(self, model_path: str = "./models/toxicity_detector", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # 毒性定义
        self.toxicity_labels = {
            0: "正常内容",
            1: "轻微不当",
            2: "歧视性言论", 
            3: "仇恨言论",
            4: "暴力内容",
            5: "严重有害"
        }
        
        # 严重程度
        self.severity_thresholds = {
            "low": 0.3,      # 轻微
            "medium": 0.6,   # 中等
            "high": 0.8      # 严重
        }
        
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> bool:
        """加载训练好的模型"""
        try:
            tokenizer_path = os.path.join(self.model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
                self.logger.warning("未找到保存的tokenizer，使用默认tokenizer")

            model_file = os.path.join(self.model_path, "model.pth")
            if os.path.exists(model_file):
                self.model = ToxicityClassifier()
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.logger.info("毒性检测模型加载")
                return True
            else:
                self.logger.warning("未找到训练好的模型，使用规则方法")
                return False
                
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            return False
    
    def detect(self, text: str) -> Dict[str, Any]:
        """检测文本毒性"""
        if self.model and self.tokenizer:
            return self._model_detect(text)
        else:
            return self._rule_based_detect(text)
    
    def _model_detect(self, text: str) -> Dict[str, Any]:
        """基于模型的检测"""
        try:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs)
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
            predicted_class = np.argmax(probabilities)
            confidence = float(probabilities[predicted_class])
            
            severity = self._get_severity(confidence, predicted_class)
            
            return {
                "is_toxic": predicted_class > 0,
                "toxicity_type": self.toxicity_labels[predicted_class],
                "confidence": confidence,
                "severity": severity,
                "class_probabilities": {
                    self.toxicity_labels[i]: float(prob) 
                    for i, prob in enumerate(probabilities)
                },
                "detection_method": "model"
            }
            
        except Exception as e:
            self.logger.error(f"模型检测失败: {e}")
            return self._rule_based_detect(text)
    
    def _rule_based_detect(self, text: str) -> Dict[str, Any]:
        """基于规则的检测（没model就会用这里，可以先运行train_toxicity_detector训练model）"""
        # 定义有害关键词
        harmful_keywords = {
            "仇恨言论": ["死", "杀", "恨", "该死", "去死"],
            "歧视性言论": ["歧视", "种族", "性别歧视", "地域黑"],
            "暴力内容": ["暴力", "打击", "攻击", "伤害", "暴打"],
            "不当内容": ["傻逼", "白痴", "智障", "脑残"]
        }
        
        detected_types = []
        max_severity = 0
        
        text_lower = text.lower()
        
        for category, keywords in harmful_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected_types.append(category)
                    if category == "仇恨言论":
                        max_severity = max(max_severity, 0.9)
                    elif category == "暴力内容":
                        max_severity = max(max_severity, 0.8)
                    elif category == "歧视性言论":
                        max_severity = max(max_severity, 0.7)
                    else:
                        max_severity = max(max_severity, 0.5)
                    break
        
        is_toxic = len(detected_types) > 0
        toxicity_type = detected_types[0] if detected_types else "正常内容"
        severity = self._get_severity(max_severity, 1 if is_toxic else 0)
        
        return {
            "is_toxic": is_toxic,
            "toxicity_type": toxicity_type,
            "confidence": max_severity,
            "severity": severity,
            "detected_categories": detected_types,
            "detection_method": "rule_based"
        }
    
    def _get_severity(self, confidence: float, predicted_class: int) -> str:
        """获取严重程度"""
        if predicted_class == 0:
            return "none"
        elif confidence >= self.severity_thresholds["high"] or predicted_class >= 4:
            return "high"
        elif confidence >= self.severity_thresholds["medium"] or predicted_class >= 2:
            return "medium"
        else:
            return "low"
    
    def batch_detect(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量检测"""
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results
    
    def should_block(self, detection_result: Dict[str, Any]) -> bool:
        """判断是否应该阻止输出"""
        return (detection_result["is_toxic"] and 
                detection_result["severity"] in ["medium", "high"])
