import time

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
import os
import json
import requests
from camel.utils.commons import logger
from volcenginesdkarkruntime import Ark


class MultimodelProcessor:
    """
    支持多种模型生成图像的处理器类
    提供不同平台模型的图像生成能力，包括OpenAI DALL-E、Qwen等
    """
    def __init__(self, output_dir: str = "data/images"):
        """
        初始化多模型处理器
        
        Args:
            output_dir: 生成图像的保存目录
        """
        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)
        # 获取当前文件所在目录
        current_dir = os.path.dirname(current_file_path)
        # 获取项目根目录（向上两级，因为当前文件在 oasis/multimodel/ 目录下）
        project_root = os.path.dirname(os.path.dirname(current_dir))
        # 构建完整的图像存储路径（项目根目录 + data/images）
        self.output_dir = os.path.join(project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储已初始化的模型
        self.models = {}
        
        # 默认系统消息
        self.default_system_message = "You are a helpful assistant that can generate images."
    
    def _create_default_model(self):
        """创建默认模型"""
        return ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        )
    
    def _save_image(self, image_data: bytes, filename: str) -> str:
        """
        保存图像数据到文件
        
        Args:
            image_data: 图像二进制数据
            filename: 保存的文件名
        
        Returns:
            保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_data)
        return filepath
    
    def _download_image(self, image_url: str) -> bytes:
        """
        下载图像文件
        
        Args:
            image_url: 图像URL
        
        Returns:
            图像二进制数据
        """
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Error downloading image: {str(e)}")
            raise
    
    def qwen_generate_image(self, 
                           prompt: str, 
                           size: str = "1140*1472",
                           quality: str = "standard") -> str:
        """
        使用Qwen模型生成图像，处理特定的Qwen返回格式
        改造后只返回一张图片的路径
        
        Args:
            prompt: 图像生成提示词
            size: 图像尺寸
            quality: 图像质量
            
        Returns:
            生成的图像文件路径，如果发生错误则返回空字符串
        """
        try:
            # 导入dashscope库
            import dashscope
            from dashscope import MultiModalConversation
            
            # 设置API URL（北京地域）
            dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"text": prompt}
                    ]
                }
            ]
            
            api_key = "sk-234578785f38454d82decb1b7023c900",

            # 调用Qwen模型生成图像（只生成一张）
            response = MultiModalConversation.call(
                api_key=api_key,
                model="qwen-image-plus",
                messages=messages,
                result_format='message',
                stream=False,
                watermark=True,
                prompt_extend=True,
                negative_prompt='',
                size=size
            )
            
            # 处理响应
            if response.status_code == 200:
                # 打印完整响应，便于调试
                logger.debug(json.dumps(response, ensure_ascii=False))
                
                # 解析响应，提取图像URL
                try:
                    # 获取output部分
                    if hasattr(response, 'output') and response.output:
                        output = response.output
                        
                        # 获取choices数组
                        if hasattr(output, 'choices') and output.choices:
                            for choice in output.choices:
                                # 获取message
                                if hasattr(choice, 'message') and choice.message:
                                    message = choice.message
                                    
                                    # 获取content数组
                                    if hasattr(message, 'content') and message.content:
                                        for content_item in message.content:
                                            # 提取image URL
                                            if isinstance(content_item, dict) and 'image' in content_item:
                                                image_url = content_item['image']
                                                
                                                try:
                                                    # 下载图像
                                                    image_data = self._download_image(image_url)
                                                    # 生成文件名
                                                    filename = f"qwen_image_{str(hash(prompt + image_url))[:8]}.png"
                                                    # 保存图像
                                                    filepath = self._save_image(image_data, filename)
                                                    # 直接返回图像路径
                                                    return filepath
                                                except Exception as e:
                                                    logger.debug(f"Error saving image: {str(e)}")
                                                    return ""
                except Exception as e:
                    logger.debug(f"Error parsing Qwen response: {str(e)}")
                    return ""
            else:
                # 处理错误响应
                error_msg = f"Qwen API error: status_code={response.status_code}"
                if hasattr(response, 'code') and response.code:
                    error_msg += f", code={response.code}"
                if hasattr(response, 'message') and response.message:
                    error_msg += f", message={response.message}"
                
                logger.debug(error_msg)
                return ""
            
            # 如果没有找到图像URL，返回空字符串
            logger.error("Warning: No image URL found in Qwen response")
            return ""
        except ImportError as e:
            logger.error(f"Dashscope library not found: {str(e)}")
            return ""
        except Exception as e:
            print(f"Error generating image with Qwen: {str(e)}")
            return ""

    def ark_generate_image(self,
                            prompt: str,
                            size: str = "1140*1472",
                            quality: str = "standard") -> str:
        client = Ark(
            # 此为默认路径，您可根据业务所在地域进行配置
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
            api_key="159a1c88-e782-423f-80e8-32c7d3fb832d"
        )
        print("----- create request -----")
        create_result = client.content_generation.tasks.create(
            model="doubao-seedance-1-0-pro-250528",  # 模型 Model ID 已为您填入
            content=[
                {
                    # 文本提示词与参数组合
                    "type": "text",
                    "text": "无人机以极快速度穿越复杂障碍或自然奇观，带来沉浸式飞行体验  --resolution 1080p  --duration 5 --camerafixed false --watermark true"
                },
                {  # 若仅需使用文本生成视频功能，可对该大括号内的内容进行注释处理，并删除上一行中大括号后的逗号。
                    # 首帧图片URL
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ark-project.tos-cn-beijing.volces.com/doc_image/seepro_i2v.png"
                    }
                }
            ]
        )
        print(create_result)
        # 轮询查询部分
        print("----- polling task status -----")
        task_id = create_result.id
        while True:
            get_result = client.content_generation.tasks.get(task_id=task_id)
            status = get_result.status
            if status == "succeeded":
                print("----- task succeeded -----")
                print(get_result)
                break
            elif status == "failed":
                print("----- task failed -----")
                print(f"Error: {get_result.error}")
                break
            else:
                print(f"Current status: {status}, Retrying after 3 seconds...")
                time.sleep(3)

    def generate_image(self, 
                      prompt: str, 
                      model_type: str = "openai", 
                      **kwargs) -> str:
        """
        统一的图像生成接口，根据model_type调用不同的生成方法
        
        Args:
            prompt: 图像生成提示词
            model_type: 模型类型，支持 "openai", "qwen"等
            **kwargs: 传递给具体生成方法的参数
        
        Returns:
            生成的图像文件路径，如果发生错误则返回空字符串
        """
        pre_prompt = "I'm posting a message and need an image to accompany it. The content is: {}"
        
        full_prompt = pre_prompt.format(prompt)
        
        if model_type.lower() == "openai":
            return ""
        elif model_type.lower() == "qwen":
            return self.qwen_generate_image(full_prompt, **kwargs)
        elif model_type.lower() == "ark":
            return self.ark_generate_image(full_prompt, **kwargs)
        else:
            logger.debug(f"Warning: Unsupported model type: {model_type}")
            return ""


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = MultimodelProcessor()  # 现在不需要显式指定output_dir
    
    # 示例：使用Qwen生成图像
    prompt = "A beautiful landscape with mountains and lake at sunset"
    print(f"Generating image with Qwen for prompt: {prompt}")
    qwen_result = processor.generate_image(
        prompt,
        model_type="ark",
        size="1140*1472",
        quality="standard"
    )
    print(f"Qwen generation result: {qwen_result}")