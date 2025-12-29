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


NODE_CLASS_MAPPINGS = {
    "StringConcat": StringConcat,
    "StringSplit": StringSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringConcat": "String Concat (文本拼接)",
    "StringSplit": "String Split (文本拆分)",
}
