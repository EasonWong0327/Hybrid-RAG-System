import torch
import torch.nn as nn
import os
import sys
from transformers import AutoTokenizer
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from core.detectors.toxicity_detector import ToxicityClassifier

class ToxicityTrainer:
    """毒性检测模型训练器"""
    
    def __init__(self, model_save_path: str = "./models/toxicity_detector"):
        self.model_save_path = model_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_sample_data(self) -> Tuple[List[str], List[int]]:
        # 模拟的训练数据
        texts = [
            "医生建议患者定期检查血压",
            "这个治疗方案不太好",
            "某地区的人都不讲卫生",
            "希望你早点死掉",
            "要狠狠地打击这种行为",
            "去死吧你这个垃圾"
        ]
        
        labels = [0, 1, 2, 3, 4, 5]
        
        expanded_texts = []
        expanded_labels = []
        
        for text, label in zip(texts, labels):
            # 莫喷，纯复制 ==
            for _ in range(100):
                expanded_texts.append(text)
                expanded_labels.append(label)
        
        return expanded_texts, expanded_labels
    
    def train(self, texts: List[str], labels: List[int], 
              epochs: int = 10, batch_size: int = 16, learning_rate: float = 2e-5):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        
        model = ToxicityClassifier()
        model.to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                batch_labels = torch.tensor(batch_labels).to(self.device)

                optimizer.zero_grad()
                logits = model(**inputs)
                loss = criterion(logits, batch_labels)

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(texts) // batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # saved
        os.makedirs(self.model_save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.model_save_path, "model.pth"))
        tokenizer.save_pretrained(os.path.join(self.model_save_path, "tokenizer"))
        
        print(f"模型保存到: {self.model_save_path}")

def main():
    print("开始训练毒性检测模型...")

    trainer = ToxicityTrainer()
    texts, labels = trainer.prepare_sample_data()
    print(f"训练数据准备完成，共 {len(texts)} 条样本")

    trainer.train(texts, labels, epochs=5, batch_size=8, learning_rate=2e-5)
    
    print("训练完成！")

if __name__ == "__main__":
    main()