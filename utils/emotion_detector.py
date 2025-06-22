from transformers import pipeline
import aiohttp
import asyncio
from typing import Tuple

class EmotionDetector:
    def __init__(self):
        # 使用更轻量级的模型
        self.image_classifier = pipeline(
            "image-classification", 
            model="DunnBC22/cnn_image_classification_emotion_detection"
        )
        self.text_classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion"
        )
    
    async def detect(self, image_data: bytes) -> Tuple[str, float]:
        """识别图片情感"""
        try:
            # 转换图片格式
            image = Image.open(io.BytesIO(image_data))
            image = image.convert("RGB")
            
            # 情感分析
            results = self.image_classifier(image)
            top_result = results[0]
            return top_result["label"], top_result["score"]
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return "neutral", 0.5
    
    async def detect_text_emotion(self, text: str) -> str:
        """识别文本情感"""
        if not text:
            return "neutral"
            
        try:
            results = self.text_classifier(text)
            label = results[0]["label"].lower()
            
            # 映射到我们的情感分类
            label_mapping = {
                "happy": "happy",
                "joy": "happy",
                "sadness": "sad",
                "anger": "angry",
                "surprise": "surprised",
                "fear": "sad",
                "love": "happy",
                "neutral": "neutral"
            }
            return label_mapping.get(label, "neutral")
        except Exception as e:
            logger.error(f"文本情感分析失败: {e}")
            return "neutral"
