import os
import json
import asyncio
import random
import aiohttp
from PIL import Image
import io
from typing import Dict, List, Optional, Tuple, Any

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp
from astrbot.api.event.filter import PlatformAdapterType
from astrbot.core import AstrBotConfig
from astrbot.api.provider import LLMResponse

# 情感分类映射
DEFAULT_EMOTIONS = {
    "高兴": "happy",
    "悲伤": "sad",
    "生气": "angry",
    "惊讶": "surprised",
    "喜爱": "love",
    "厌恶": "disgust"
}

@register("emoji-collection", "tianyu", "基于多模态AI的表情收集与发送插件", "1.0.0")
class EmojiCollectionPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.ollama_api_url = self.config.get("ollama_api_url", "http://localhost:11434")
        self.ollama_model = self.config.get("ollama_model", "llava")
        self.auto_collect = self.config.get("auto_collect", True)
        self.emotion_map = self.config.get("default_emotions", DEFAULT_EMOTIONS)
        
        # 创建存储目录
        self.base_dir = os.path.join("data", "plugins_data", "emoji-collection")
        os.makedirs(self.base_dir, exist_ok=True)
        
        # 加载情感数据库
        self.emotion_db_path = os.path.join(self.base_dir, "emotion_db.json")
        self.emotion_db = self.load_emotion_db()
        
        logger.info(f"EmojiCollection插件已加载，使用模型: {self.ollama_model}")

    def load_emotion_db(self) -> Dict[str, List[str]]:
        """加载情感数据库"""
        if os.path.exists(self.emotion_db_path):
            with open(self.emotion_db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {emotion: [] for emotion in self.emotion_map.keys()}

    def save_emotion_db(self):
        """保存情感数据库"""
        with open(self.emotion_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.emotion_db, f, ensure_ascii=False, indent=2)

    def get_emotion_dir(self, emotion: str) -> str:
        """获取情感对应的目录"""
        emotion_dir = os.path.join(self.base_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        return emotion_dir

    async def analyze_emotion(self, image_data: bytes) -> Optional[str]:
        """使用Ollama分析图片情感"""
        try:
            # 准备请求数据
            data = {
                "model": self.ollama_model,
                "prompt": "这张图片表达了什么情感？请从以下选项中选择一个: " + 
                          ", ".join(self.emotion_map.keys()) + 
                          "。只回答情感名称，不要解释。",
                "images": [image_data],
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_api_url}/api/generate",
                    json=data
                ) as response:
                    if response.status != 200:
                        logger.error(f"Ollama API错误: {response.status}")
                        return None
                    
                    result = await response.json()
                    response_text = result.get("response", "").strip()
                    
                    # 检查响应是否包含有效情感
                    for emotion in self.emotion_map.keys():
                        if emotion in response_text:
                            return emotion
                    
                    logger.warning(f"无法识别的响应: {response_text}")
                    return None
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            return None

    async def process_image(self, event: AstrMessageEvent, emotion: Optional[str] = None):
        """处理收到的图片"""
        # 获取图片数据
        image_seg = next((seg for seg in event.get_messages() if isinstance(seg, Comp.Image)), None)
        if not image_seg:
            return "未找到图片"
        
        # 下载图片
        image_url = image_seg.url
        if image_url.startswith("https:"):
            image_url = image_url.replace("https:", "http:")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                if response.status != 200:
                    return "图片下载失败"
                
                image_data = await response.read()
                
                # 如果未指定情感，使用AI分析
                if emotion is None:
                    emotion = await self.analyze_emotion(image_data)
                    if not emotion:
                        return "无法识别图片情感"
                
                # 检查情感是否有效
                if emotion not in self.emotion_db:
                    return f"无效的情感: {emotion}"
                
                # 保存图片
                emotion_dir = self.get_emotion_dir(emotion)
                file_count = len(os.listdir(emotion_dir)) + 1
                file_ext = self.detect_image_format(image_data)
                file_name = f"emoji_{file_count}.{file_ext}"
                file_path = os.path.join(emotion_dir, file_name)
                
                with open(file_path, 'wb') as f:
                    f.write(image_data)
                
                # 更新数据库
                self.emotion_db[emotion].append(file_path)
                self.save_emotion_db()
                
                return f"已添加 {emotion} 表情: {file_name}"

    def detect_image_format(self, image_data: bytes) -> str:
        """检测图片格式"""
        try:
            img = Image.open(io.BytesIO(image_data))
            return img.format.lower() if img.format else "jpg"
        except:
            return "jpg"

    @filter.command("emoadd")
    async def emoadd_command(self, event: AstrMessageEvent, emotion: Optional[str] = None):
        """添加表情图片"""
        result = await self.process_image(event, emotion)
        yield event.plain_result(result)

    @filter.command("emolist")
    async def emolist_command(self, event: AstrMessageEvent, emotion: str):
        """列出表情图片"""
        if emotion not in self.emotion_db:
            yield event.plain_result(f"无效的情感: {emotion}")
            return
        
        files = self.emotion_db[emotion]
        if not files:
            yield event.plain_result(f"没有找到 {emotion} 表情")
            return
        
        file_list = "\n".join([os.path.basename(f) for f in files])
        yield event.plain_result(f"{emotion} 表情列表:\n{file_list}")

    @filter.command("emosend")
    async def emosend_command(self, event: AstrMessageEvent, emotion: str):
        """发送随机表情"""
        if emotion not in self.emotion_db:
            yield event.plain_result(f"无效的情感: {emotion}")
            return
        
        files = self.emotion_db[emotion]
        if not files:
            yield event.plain_result(f"没有找到 {emotion} 表情")
            return
        
        # 随机选择一个表情
        selected_file = random.choice(files)
        yield event.chain_result([
            Comp.Plain(f"{emotion}表情:"),
            Comp.Image.fromFileSystem(selected_file)
        ])

    @filter.command("emohelp")
    async def emohelp_command(self, event: AstrMessageEvent):
        """显示帮助信息"""
        help_text = (
            "表情收集与发送插件指令:\n"
            "/emoadd [情感] - 添加表情图片(可选情感)\n"
            "/emolist <情感> - 列出指定情感的表情\n"
            "/emosend <情感> - 发送随机表情\n"
            "/emohelp - 显示帮助信息\n"
            f"支持的情感: {', '.join(self.emotion_map.keys())}"
        )
        yield event.plain_result(help_text)

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """自动收集群消息中的表情"""
        if not self.auto_collect:
            return
        
        # 检查是否包含图片
        if any(isinstance(seg, Comp.Image) for seg in event.get_messages()):
            result = await self.process_image(event)
            logger.info(f"自动收集表情: {result}")

    @filter.on_llm_response()
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """在LLM响应后自动发送表情"""
        try:
            # 从LLM响应中提取情感关键词
            # 安全获取纯文本内容
            response_text = ""
            if resp.result_chain:
                for comp in resp.result_chain.chain:
                    if isinstance(comp, Comp.Plain):
                        response_text += comp.text
            
            # 如果没有文本内容，则跳过
            if not response_text:
                return
            
            for emotion in self.emotion_map.keys():
                if emotion in response_text and random.random() < 0.7:  # 70%概率发送表情
                    files = self.emotion_db.get(emotion, [])
                    if files:
                        selected_file = random.choice(files)
                        # 避免在短时间内连续发送多条消息
                        await asyncio.sleep(0.5)
                        await event.send(event.chain_result([
                            Comp.Image.fromFileSystem(selected_file)
                        ]))
                    break
        except Exception as e:
            logger.error(f"自动发送表情失败: {str(e)}")

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("表情收集插件已卸载")
