import numpy as np
from PIL import Image
from transformers import pipeline
import asyncio
from typing import Tuple

class EmotionDetector:
    def __init__(self):
        # 初始化情感分析模型
        self.image_classifier = pipeline(
            "image-classification", 
            model="rafalosa/diffusion-emotion"
        )
        self.text_classifier = pipeline(
            "text-classification",
            model="finiteautomata/bertweet-base-sentiment-analysis"
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
            print(f"Error in emotion detection: {e}")
            return "neutral", 0.5
    
    async def detect_text_emotion(self, text: str) -> str:
        """识别文本情感"""
        if not text:
            return "neutral"
            
        try:
            results = self.text_classifier(text)
            label = results[0]["label"].lower()
            
            # 映射到我们的情感分类
            if label in ["happy", "joy"]:
                return "happy"
            elif label in ["sad", "sorrow"]:
                return "sad"
            elif label in ["angry", "disgust"]:
                return "angry"
            elif label in ["surprise"]:
                return "surprised"
            else:
                return "neutral"
                
        except Exception as e:
            print(f"Error in text emotion detection: {e}")
            return "neutral"
