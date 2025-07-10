import torch
from transformers import BertModel, BertTokenizer
from config.config import BERT_MODEL_NAME, MAX_SEQ_LENGTH, BATCH_SIZE

class BertEncoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.model = BertModel.from_pretrained(BERT_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts):
        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            
            # tokenizer
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)
        
        return vectors

    def encode_single(self, text):
        return self.encode([text])[0]
    
    def encode_texts(self, texts):
        """别名方法，为了兼容性"""
        return self.encode(texts) 