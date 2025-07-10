import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.detectors.toxicity_detector import ToxicityDetector
from config.config import LLM_MODEL_NAME, LLM_DEVICE
import logging

logger = logging.getLogger(__name__)

class Qwen3Generator:
    def __init__(self, model_name=None):
        self.model_name = model_name or "Qwen/Qwen3-4B"
        self.device = torch.device(LLM_DEVICE if LLM_DEVICE != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = 2048
        self.temperature = 0.7

        logger.info(f"使用模型: {self.model_name}, 设备: {self.device}")
        
        self._load_model()
        
        self.toxicity_detector = ToxicityDetector()
        self.toxicity_detector.load_model()

    def _load_model(self):
        try:
            logger.info(f"加载模型: {self.model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
                use_fast=False
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            if self.device.type != "cuda":
                self.model = self.model.to(self.device)

            self.model.eval()
            logger.info("模型加载")

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            if "model type `Qwen3`" in str(e):
                # 我测试着Qwen3对torch版本和tf版本要求挺高的
                logger.error("需要升级transformers版本: pip install transformers --upgrade")
            raise

    def generate(self, prompt, max_new_tokens=512, temperature=None, return_probabilities=False):
        try:
            messages = [
                {"role": "system", "content": "你是一个专业的医疗助手，请根据提供的医学知识回答用户问题。请直接给出简洁专业的回答，不要包含思考过程或特殊标签。"},
                {"role": "user", "content": prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                text, 
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length - max_new_tokens
            ).to(self.device)

            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature or self.temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "return_dict_in_generate": True,
                "output_scores": return_probabilities,
            }

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # 清理思考标签
            generated_text = self._clean_response(generated_text)

            probability_info = None
            if return_probabilities and hasattr(outputs, 'scores') and outputs.scores:
                probability_info = self._calculate_probability_info(outputs, inputs)

            toxicity_result = self.toxicity_detector.detect(generated_text)

            result = {
                "response": generated_text,
                "toxicity": toxicity_result,
                "model_info": {
                    "model_name": self.model_name,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature or self.temperature
                }
            }

            if probability_info:
                result["probability_info"] = probability_info

            return result

        except Exception as e:
            logger.error(f"生成回答失败: {e}")
            return {
                "response": "抱歉，我无法生成回答，请稍后再试。",
                "error": str(e),
                "toxicity": {"is_toxic": False, "confidence": 0.0}
            }

    def _clean_response(self, text: str) -> str:
        """清理响应文本，移除不必要的标签和格式"""
        import re
        
        # 移除<think>...</think>标签及其内容
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 移除单独的<think>标签（如果没有闭合）
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        
        # 移除其他可能的标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 清理多余的空白字符
        text = re.sub(r'\n\s*\n', '\n\n', text)  # 多个空行变为两个
        text = re.sub(r'[ \t]+', ' ', text)      # 多个空格变为一个
        
        return text.strip()

    def _calculate_probability_info(self, outputs, inputs):
        """计算概率分布信息"""
        try:
            scores = outputs.scores
            generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:]

            token_probabilities = []
            log_probs = []

            for i, (score, token_id) in enumerate(zip(scores, generated_tokens)):
                probs = torch.softmax(score[0], dim=-1)
                token_prob = probs[token_id].item()
                log_prob = torch.log(probs[token_id]).item()
                token_text = self.tokenizer.decode([token_id])

                token_probabilities.append({
                    "token": token_text,
                    "probability": token_prob,
                    "log_probability": log_prob,
                    "position": i
                })

                log_probs.append(log_prob)

            avg_prob = sum(tp["probability"] for tp in token_probabilities) / len(token_probabilities)
            min_prob = min(tp["probability"] for tp in token_probabilities)
            perplexity = torch.exp(-torch.tensor(log_probs).mean()).item()

            return {
                "token_probabilities": token_probabilities,
                "statistics": {
                    "average_probability": avg_prob,
                    "minimum_probability": min_prob,
                    "perplexity": perplexity,
                    "total_tokens": len(token_probabilities)
                }
            }

        except Exception as e:
            logger.error(f"计算概率信息失败: {e}")
            return None

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "temperature": self.temperature
        } 