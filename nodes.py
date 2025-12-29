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

    RETURN_TYPES = ("IMAGE", "STRING",) + ("STRING",) * 32
    RETURN_NAMES = ("image_palette", "hex_list") + tuple(f"hex_{i}" for i in range(1, 33))
    FUNCTION = "get_dominant_colors"
    CATEGORY = "image"

    def get_dominant_colors(self, image, max_colors=5):
        # image is a torch tensor: [batch, height, width, channels] (0-1 float)
        # We process the first image in the batch
        import torch
        from PIL import Image, ImageDraw
        import numpy as np
        import math

        # Convert tensor to PIL Image (take first image of batch)
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Resize to speed up calculation
        img_small = img.resize((150, 150))
        
        if img_small.mode != "RGB":
            img_small = img_small.convert("RGB")
            
        print(f"DEBUG: ImageDominantColor requested {max_colors} colors")

            
        # 1. Adaptively determine candidate count
        # We need more candidates to allow filtering (e.g., getting past the 10 shades of brown)
        candidate_count = max(max_colors * 5, 48)
        
        # 2. Quantize to get candidate palette
        result = img_small.quantize(colors=candidate_count)
        palette = result.getpalette() or []
        color_counts = result.getcolors(maxcolors=256)
        
        dominant_colors = []
        if color_counts:
            # Sort by count (descending)
            color_counts.sort(key=lambda x: x[0], reverse=True)
            
            # Extract all candidate colors [ (r,g,b), ... ]
            candidates = []
            for count, index in color_counts:
                r = palette[index * 3]
                g = palette[index * 3 + 1]
                b = palette[index * 3 + 2]
                candidates.append((r, g, b))
                
            # 3. Diversity Filtering
            # Iterate candidates and pick only if distinct enough from already selected
            min_dist_sq = 25 * 25  # Threshold approx 25 in RGB space
            
            selected_colors = []
            
            for c in candidates:
                if len(selected_colors) >= max_colors:
                    break
                    
                is_distinct = True
                for sc in selected_colors:
                    # Euclidean distance squared: (r1-r2)^2 + ...
                    dist_sq = (c[0]-sc[0])**2 + (c[1]-sc[1])**2 + (c[2]-sc[2])**2
                    if dist_sq < min_dist_sq:
                        is_distinct = False
                        break
                
                if is_distinct:
                    selected_colors.append(c)
            
            # If we didn't get enough colors due to strict filtering, append the rest from top frequency
            # ignoring distance check (to fill the request)
            if len(selected_colors) < max_colors:
                for c in candidates:
                    if len(selected_colors) >= max_colors:
                        break
                    # Avoid exact duplicates
                    if c not in selected_colors:
                        selected_colors.append(c)
            
            dominant_colors = selected_colors
            
        else:
            dominant_colors = [(0, 0, 0)]

        # Prepare Hex strings
        hex_colors = []
        for c in dominant_colors:
            hex_colors.append("#{:02x}{:02x}{:02x}".format(c[0], c[1], c[2]))
            
        # 1. Output Hex List (comma separated)
        hex_list_str = ", ".join(hex_colors)
        
        # 2. Output Individual Hex Ports (32 fixed ports)
        # Pad with empty string if fewer than 32 colors found
        total_ports = 32
        hex_ports = hex_colors[:total_ports] + [""] * (total_ports - len(hex_colors[:total_ports]))
        
        # 3. Create Palette Image
        w, h = 512, 512
        stripe_height = h // len(dominant_colors) if len(dominant_colors) > 0 else h
        palette_img = Image.new("RGB", (w, h), (0, 0, 0))
        draw = ImageDraw.Draw(palette_img)
        
        for idx, color in enumerate(dominant_colors):
            y0 = idx * stripe_height
            y1 = (idx + 1) * stripe_height if idx < len(dominant_colors) - 1 else h
            draw.rectangle([0, y0, w, y1], fill=color)
            
        # Convert back to torch tensor
        palette_img_np = np.array(palette_img).astype(np.float32) / 255.0
        palette_tensor = torch.from_numpy(palette_img_np)[None,] 
        
        return (palette_tensor, hex_list_str) + tuple(hex_ports)


class MediaInfo:
    """
    获取媒体信息（图片、视频、音频）
    支持输入：图片Tensor、音频Dict、VHS视频信息Dict、文件路径
    输出：宽、高、时长、FPS、采样率等详细信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "file_path": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "video_info": ("VHS_VIDEOINFO",),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT", "FLOAT", "INT", "INT", "INT")
    RETURN_NAMES = ("info_text", "width", "height", "batch_size", "fps", "duration", "frame_count", "sample_rate", "channels")
    FUNCTION = "get_info"
    CATEGORY = "utils"

    def get_info(self, file_path="", image=None, audio=None, video_info=None):
        import json
        import os
        
        info = {}
        
        # Defaults
        width = 0
        height = 0
        batch_size = 0
        fps = 0.0
        duration = 0.0
        frame_count = 0
        sample_rate = 0
        channels = 0
        
        # 1. Image Tensor Info
        if image is not None:
            # [Batch, Height, Width, Channels]
            shape = image.shape
            batch_size = shape[0]
            height = shape[1]
            width = shape[2]
            info["image_tensor"] = {
                "batch_size": batch_size,
                "height": height,
                "width": width,
                "channels": shape[3] if len(shape) > 3 else 1
            }
            
        # 2. Audio Info
        if audio is not None:
            # ComfyUI audio dict: {'waveform': tensor, 'sample_rate': int}
            if 'waveform' in audio:
                waveform = audio['waveform']
                # [Batch, Channels, Samples] or [Channels, Samples]
                channels = waveform.shape[1] if len(waveform.shape) > 1 else 1
                total_samples = waveform.shape[-1]
                sample_rate = audio.get('sample_rate', 44100)
                duration = total_samples / sample_rate if sample_rate > 0 else 0
                
                info["audio_tensor"] = {
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "total_samples": total_samples,
                    "duration": round(duration, 4)
                }

        # 3. VHS Video Info
        if video_info is not None:
            # video_info is a dict from ComfyUI-VideoHelperSuite
            # Expected keys: source_fps, source_width, source_height, source_duration, source_frame_count
            fps = video_info.get("source_fps", video_info.get("loaded_fps", 0.0))
            width = video_info.get("source_width", video_info.get("loaded_width", width))
            height = video_info.get("source_height", video_info.get("loaded_height", height))
            duration = video_info.get("source_duration", video_info.get("loaded_duration", duration))
            frame_count = video_info.get("source_frame_count", video_info.get("loaded_frame_count", 0))
            
            info["video_info"] = video_info

        # 4. File Path Info (Fallback/Supplement)
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            info["file"] = {
                "path": file_path,
                "size_bytes": file_size,
                "extension": file_ext
            }
            
            # Simple Image Check using PIL
            if file_ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff']:
                try:
                    from PIL import Image
                    with Image.open(file_path) as img:
                        if width == 0: width = img.width
                        if height == 0: height = img.height
                        info["file_image"] = {
                            "format": img.format,
                            "mode": img.mode,
                            "width": img.width,
                            "height": img.height
                        }
                except:
                    pass
        
        # Construct summary string
        return (json.dumps(info, indent=4, ensure_ascii=False), width, height, batch_size, fps, duration, frame_count, sample_rate, channels)


NODE_CLASS_MAPPINGS = {
    "StringConcat": StringConcat,
    "StringSplit": StringSplit,
    "ImageDominantColor": ImageDominantColor,
    "MediaInfo": MediaInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringConcat": "String Concat (文本拼接)",
    "StringSplit": "String Split (文本拆分)",
    "ImageDominantColor": "Image Dominant Color (主色提取)",
    "MediaInfo": "Media Info (媒体信息)",
}
