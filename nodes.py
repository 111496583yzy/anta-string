import re

class StringConcat:
    """
    将多个文本输入拼接成一个输出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "text1": ("STRING", {"multiline": True, "default": ""}),
                "text2": ("STRING", {"multiline": True, "default": ""}),
                "text3": ("STRING", {"multiline": True, "default": ""}),
                "text4": ("STRING", {"multiline": True, "default": ""}),
                "text5": ("STRING", {"multiline": True, "default": ""}),
                "separator": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "concat"
    CATEGORY = "text"

    def concat(self, text1="", text2="", text3="", text4="", text5="", separator=""):
        # 收集所有非空文本
        texts = [t for t in [text1, text2, text3, text4, text5] if t.strip()]
        # 使用分隔符拼接
        result = separator.join(texts)
        return (result,)


class StringSplit:
    """
    将包含图片1:~图片5:格式的文本拆分成5个输出
    输入格式示例: 图片1:这是第一张图片的描述。图片2:这是第二张图片。
    支持中文冒号：和英文冒号:
    输出会保留 图片N：格式前缀（统一使用中文冒号），并自动补充句号
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("图片1", "图片2", "图片3", "图片4", "图片5")
    FUNCTION = "split"
    CATEGORY = "text"

    def split(self, text=""):
        # 初始化5个输出为空字符串
        results = ["", "", "", "", ""]
        
        # 使用正则表达式匹配 图片1: 到 图片5: 的内容
        # 支持中文冒号：和英文冒号:
        for i in range(1, 6):
            # 匹配 "图片N:" 或 "图片N：" 后面的内容，直到遇到下一个"图片"标记或结尾
            pattern = rf'图片{i}[:：]\s*(.*?)(?=图片[1-5][:：]|$)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                # 移除末尾可能存在的句号（统一处理）
                content = content.rstrip('。.')
                if content:
                    # 输出格式: 图片N：内容。 (统一使用中文冒号，自动补充句号)
                    results[i-1] = f"图片{i}：{content}。"
        
        return tuple(results)


class ImageDominantColor:
    """
    提取图片主色调，输出纯色图片和十六进制色值字符串
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "hex_color")
    FUNCTION = "get_dominant_color"
    CATEGORY = "image"

    def get_dominant_color(self, image):
        # image is a torch tensor: [batch, height, width, channels] (0-1 float)
        # We process the first image in the batch
        import torch
        from PIL import Image
        import numpy as np

        # Convert tensor to PIL Image (take first image of batch)
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Resize to speed up calculation and reduce noise
        img_small = img.resize((150, 150))
        
        # Get colors (ignoring alpha if present, assuming RGB from ComfyUI)
        if img_small.mode != "RGB":
            img_small = img_small.convert("RGB")
            
        # Efficiently find dominant color using quantization
        # This forces the image to 1 color, finding the most representative one
        result = img_small.quantize(colors=1)
        dominant_color = result.getpalette()[:3]
        
        # Hex string
        hex_color = "#{:02x}{:02x}{:02x}".format(dominant_color[0], dominant_color[1], dominant_color[2])
        
        # Create pure color image matching input dimensions (OR user-defined, but for now matching batch behavior)
        # We'll return a single 512x512 pure color image to ensure it's usable, 
        # or we could match input size. Let's make it typical generic size or match input?
        # Matching input might be huge. Let's stick to a standard size or the size of input.
        # ComfyUI usually expects standard tensors.
        
        # Create a solid color image for output
        # Using a reasonable size 512x512
        output_img = Image.new("RGB", (512, 512), tuple(dominant_color))
        
        # Convert back to torch tensor
        output_img_np = np.array(output_img).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_img_np)[None,] # Add batch dimension
        
        return (output_tensor, hex_color)


NODE_CLASS_MAPPINGS = {
    "StringConcat": StringConcat,
    "StringSplit": StringSplit,
    "ImageDominantColor": ImageDominantColor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringConcat": "String Concat (文本拼接)",
    "StringSplit": "String Split (文本拆分)",
    "ImageDominantColor": "Image Dominant Color (主色提取)",
}
