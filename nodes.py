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
    提取图片主色调，支持提取多种颜色（最多32种）
    输出：色板图片、HEX颜色列表字符串、以及前5种颜色的独立HEX输出
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_colors": ("INT", {"default": 5, "min": 1, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image_palette", "hex_list", "hex_1", "hex_2", "hex_3", "hex_4", "hex_5")
    FUNCTION = "get_dominant_colors"
    CATEGORY = "image"

    def get_dominant_colors(self, image, max_colors=5):
        # image is a torch tensor: [batch, height, width, channels] (0-1 float)
        # We process the first image in the batch
        import torch
        from PIL import Image, ImageDraw
        import numpy as np

        # Convert tensor to PIL Image (take first image of batch)
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Resize to speed up calculation and reduce noise
        img_small = img.resize((150, 150))
        
        # Get colors (ignoring alpha if present, assuming RGB from ComfyUI)
        if img_small.mode != "RGB":
            img_small = img_small.convert("RGB")
            
        # Efficiently find dominant colors using quantization
        # This reduces the image to 'max_colors' colors
        result = img_small.quantize(colors=max_colors)
        
        # Get palette (r, g, b, r, g, b, ...)
        palette = result.getpalette() or []
        # Quantize might return a palette of 256 entries (768 ints), trimming to used colors
        # We need to find which colors are actually used and sort by frequency
        # Convert quantized image to data to count frequencies
        color_counts = result.getcolors(maxcolors=256)
        if color_counts:
            # Sort by count (descending)
            color_counts.sort(key=lambda x: x[0], reverse=True)
            # Limit to max_colors found (in case < max_colors)
            dominant_colors = []
            for count, index in color_counts[:max_colors]:
                # Extract RGB from palette
                r = palette[index * 3]
                g = palette[index * 3 + 1]
                b = palette[index * 3 + 2]
                dominant_colors.append((r, g, b))
        else:
            # Fallback if getcolors fails (rare)
            dominant_colors = [(0, 0, 0)]

        # Prepare Hex strings
        hex_colors = []
        for c in dominant_colors:
            hex_colors.append("#{:02x}{:02x}{:02x}".format(c[0], c[1], c[2]))
            
        # 1. Output Hex List (comma separated)
        hex_list_str = ", ".join(hex_colors)
        
        # 2. Output Individual Hex Ports (Top 5)
        # Pad with empty string if fewer than 5 colors found
        hex_ports = hex_colors[:5] + [""] * (5 - len(hex_colors[:5]))
        
        # 3. Create Palette Image
        # Create a horizontal strip or grid showing all colors
        # 512 width, height depends on count? Or fixed 512x512 with stripes?
        # Let's do horizontal stripes of equal height for clarity
        w, h = 512, 512
        stripe_height = h // len(dominant_colors)
        palette_img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(palette_img)
        
        for idx, color in enumerate(dominant_colors):
            y0 = idx * stripe_height
            # Make the last one fill the remaining space to avoid gaps
            y1 = (idx + 1) * stripe_height if idx < len(dominant_colors) - 1 else h
            draw.rectangle([0, y0, w, y1], fill=color)
            
        # Convert back to torch tensor
        palette_img_np = np.array(palette_img).astype(np.float32) / 255.0
        palette_tensor = torch.from_numpy(palette_img_np)[None,] # Add batch dimension
        
        return (palette_tensor, hex_list_str, hex_ports[0], hex_ports[1], hex_ports[2], hex_ports[3], hex_ports[4])


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
