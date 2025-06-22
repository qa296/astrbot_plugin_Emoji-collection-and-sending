import os
import random
import asyncio
import io
from typing import Optional, Tuple
from PIL import Image

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp
from astrbot.core import AstrBotConfig
from astrbot.api.provider import LLMResponse  # 添加这行导入

# 情感分类标签
EMOTION_LABELS = {
    "happy": "高兴",
    "sad": "悲伤", 
    "angry": "生气",
    "surprised": "惊讶",
    "neutral": "中性"
}

@register("emotion_memes", "YourName", "智能情感表情包插件", "1.0.0")
class EmotionMemesPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.auto_collect = config.get("auto_collect", True)
        self.emotion_threshold = config.get("emotion_threshold", 0.7)
        self.send_probability = config.get("send_probability", 0.3)
        
        # 初始化数据目录
        self.data_dir = os.path.join("data", "emotion_memes")
        os.makedirs(self.data_dir, exist_ok=True)
        for emotion in EMOTION_LABELS.values():
            os.makedirs(os.path.join(self.data_dir, emotion), exist_ok=True)
            
        # 初始化情感分析模型
        from utils.emotion_detector import EmotionDetector
        self.detector = EmotionDetector()
        
        logger.info("情感表情包插件已加载")

    async def terminate(self):
        logger.info("情感表情包插件已卸载")

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def auto_collect_memes(self, event: AstrMessageEvent):
        """自动收集表情包"""
        if not self.auto_collect:
            return
            
        # 检查消息中是否包含图片
        for component in event.message_obj.message:
            if isinstance(component, Comp.Image):
                await self.process_and_save_image(component, event)
    
    async def process_and_save_image(self, image_component: Comp.Image, event: AstrMessageEvent):
        """处理并保存图片"""
        try:
            # 下载图片
            image_data = await self.download_image(image_component)
            if not image_data:
                return
                
            # 情感分析
            emotion, confidence = await self.detect_emotion(image_data)
            if confidence < self.emotion_threshold:
                return
                
            # 保存图片到对应情感目录
            emotion_dir = os.path.join(self.data_dir, EMOTION_LABELS[emotion])
            existing_files = os.listdir(emotion_dir)
            filename = f"meme_{len(existing_files)+1}.jpg"
            filepath = os.path.join(emotion_dir, filename)
            
            with open(filepath, "wb") as f:
                f.write(image_data)
                
            logger.info(f"已保存表情包到 {filepath} (情感: {emotion}, 置信度: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"处理图片失败: {e}")

    async def download_image(self, image_component: Comp.Image) -> Optional[bytes]:
        """下载图片"""
        try:
            if hasattr(image_component, 'file') and image_component.file:
                # 本地文件
                with open(image_component.file, "rb") as f:
                    return f.read()
            elif hasattr(image_component, 'url') and image_component.url:
                # 远程URL
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_component.url) as resp:
                        if resp.status == 200:
                            return await resp.read()
            return None
        except Exception as e:
            logger.error(f"下载图片失败: {e}")
            return None

    @filter.on_llm_response()
    async def send_emotion_meme(self, event: AstrMessageEvent, resp: LLMResponse):
        """在LLM回复后发送合适的情感表情包"""
        if random.random() > self.send_probability:
            return
            
        # 分析消息情感
        text = resp.result_chain.get_plain_text()
        if not text:
            return
            
        text_emotion = await self.detector.detect_text_emotion(text)
        
        # 从对应目录随机选择表情包
        emotion_dir = os.path.join(self.data_dir, EMOTION_LABELS[text_emotion])
        if os.path.exists(emotion_dir):
            memes = [f for f in os.listdir(emotion_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            if memes:
                selected_meme = random.choice(memes)
                meme_path = os.path.join(emotion_dir, selected_meme)
                yield event.image_result(meme_path)

    @filter.command("emotion_stats")
    async def show_stats(self, event: AstrMessageEvent):
        """显示表情包统计信息"""
        stats = []
        for emotion in EMOTION_LABELS.values():
            count = len(os.listdir(os.path.join(self.data_dir, emotion)))
            stats.append(f"{emotion}: {count}张")
            
        yield event.plain_result("表情包统计:\n" + "\n".join(stats))
