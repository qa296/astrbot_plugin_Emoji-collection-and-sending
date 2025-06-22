from PIL import Image
import io

async def resize_image(image_data: bytes, max_size: int = 512) -> bytes:
    """调整图片大小"""
    try:
        img = Image.open(io.BytesIO(image_data))
        
        # 保持宽高比调整大小
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            
        # 转换为JPEG格式
        output = io.BytesIO()
        img.convert("RGB").save(output, format="JPEG", quality=85)
        return output.getvalue()
        
    except Exception as e:
        print(f"Error resizing image: {e}")
        return image_data  # 如果处理失败返回原图
