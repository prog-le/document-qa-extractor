import json
import logging
from typing import List, Dict, Any, Optional, Union
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 问答对提取的提示模板
QA_PAIRS_HUMAN_PROMPT = """  
请按以下格式整理学习成果:  
<Context>  
文本  
</Context>  
[  
{{"question": "问题1","answer":"答案1"}},  
{{"question": "问题2","answer":"答案2"}},  
]  
------  
  
我们开始吧!  
  
<Context>  
{text}  
<Context/>  
"""

def process_with_llm(text: str, llm_provider: str = 'ollama', model_name: str = 'llama3', 
                    max_retries: int = 3, api_key: Optional[str] = None,
                    callback: Optional[callable] = None) -> Optional[List[Dict[str, str]]]:
    """
    使用大模型处理文本并提取问答对
    
    Args:
        text: 要处理的文本内容
        llm_provider: 大模型提供商
        model_name: 模型名称
        max_retries: 最大重试次数
        api_key: API密钥（对于需要认证的服务）
        callback: 回调函数，当获取到结果时调用
        
    Returns:
        问答对列表，如果处理失败则返回None
    """
    retries = 0
    # 创建进度条
    with tqdm(total=max_retries, desc=f"调用{llm_provider}模型", position=1) as pbar:
        while retries < max_retries:
            try:
                # 准备提示
                prompt = QA_PAIRS_HUMAN_PROMPT.format(text=text)
                
                # 根据提供商选择不同的API
                result = None
                
                if llm_provider == 'ollama':
                    result = process_with_ollama(prompt, model_name)
                elif llm_provider == 'bailian':
                    result = process_with_bailian(prompt, model_name, api_key)
                elif llm_provider == 'volcengine':
                    result = process_with_volcengine(prompt, model_name, api_key)
                elif llm_provider == 'deepseek':
                    result = process_with_deepseek(prompt, model_name, api_key)
                else:
                    logger.error(f"不支持的大模型提供商: {llm_provider}")
                    return None
                
                # 如果有回调函数，调用它
                if callback and result:
                    callback(result)
                
                # 更新进度条到最大值，表示成功
                pbar.update(max_retries)  # 直接更新到完成
                pbar.close()  # 关闭进度条
                
                return result
            except Exception as e:
                retries += 1
                logger.warning(f"调用大模型失败 (尝试 {retries}/{max_retries}): {str(e)}")
                
                # 更新进度条
                pbar.update(1)
                
                if retries < max_retries:
                    # 指数退避策略
                    wait_time = 2 ** retries
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"达到最大重试次数，放弃处理")
                    pbar.close()  # 关闭进度条
                    return None

def process_with_ollama(prompt: str, model_name: str) -> List[Dict[str, str]]:
    """使用Ollama处理文本"""
    try:
        import requests
        
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            # 添加超时设置
            "timeout": 120
        }
        
        try:
            response = requests.post(url, json=data, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            if 'response' not in result:
                logger.error(f"Ollama API返回格式异常: {result}")
                return []
                
            return extract_qa_pairs_from_response(result.get('response', ''))
        except requests.exceptions.ConnectionError:
            logger.error("无法连接到Ollama服务，请确保服务已启动")
            return []
        except requests.exceptions.Timeout:
            logger.error("Ollama API请求超时")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"Ollama API HTTP错误: {str(e)}")
            return []
        except json.JSONDecodeError:
            logger.error("Ollama API返回的不是有效的JSON格式")
            return []
    except ImportError:
        logger.error("请安装requests库: pip install requests")
        raise
    except Exception as e:
        logger.error(f"Ollama API调用失败: {str(e)}")
        raise

def process_with_bailian(prompt: str, model_name: str, api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """使用百炼API处理文本"""
    try:
        from openai import OpenAI
        
        if not api_key:
            logger.error("使用百炼API需要提供API密钥")
            return []
        
        # 设置客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 默认使用qwen-omni-turbo模型，除非指定其他模型
        model_to_use = model_name if model_name else "qwen-omni-turbo"
        
        logger.info(f"使用百炼API调用模型: {model_to_use}")
        
        try:
            # 创建聊天完成请求
            completion = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                modalities=["text"],
                # 百炼API要求stream为True，但我们需要完整响应
                stream=False
            )
            
            # 获取响应内容
            if completion.choices and len(completion.choices) > 0:
                response_text = completion.choices[0].message.content
                return extract_qa_pairs_from_response(response_text)
            else:
                logger.warning("百炼API返回的响应没有内容")
                return []
                
        except Exception as e:
            logger.error(f"百炼API请求失败: {str(e)}")
            return []
            
    except ImportError:
        logger.error("请安装openai库: pip install openai>=1.0.0")
        raise
    except Exception as e:
        logger.error(f"百炼API调用失败: {str(e)}")
        raise

def process_with_volcengine(prompt: str, model_name: str, api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """使用火山引擎API处理文本"""
    try:
        import requests
        
        if not api_key:
            logger.error("使用火山引擎API需要提供API密钥")
            return []
        
        # 默认使用doubao-pro模型，除非指定其他模型
        model_to_use = model_name if model_name else "doubao-pro-32k-240615"
        
        # 火山引擎API端点
        api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        
        logger.info(f"使用火山引擎API调用模型: {model_to_use}")
        
        # 准备请求头和请求体
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model_to_use,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的问答提取助手，请根据提供的文本提取问答对。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        try:
            # 发送请求
            response = requests.post(api_url, headers=headers, json=data, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # 解析响应
            if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0]:
                response_text = result['choices'][0]['message']['content']
                return extract_qa_pairs_from_response(response_text)
            else:
                logger.warning("火山引擎API返回的响应格式异常")
                logger.debug(f"API响应: {result}")
                return []
                
        except requests.exceptions.ConnectionError:
            logger.error("无法连接到火山引擎API服务")
            return []
        except requests.exceptions.Timeout:
            logger.error("火山引擎API请求超时")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"火山引擎API HTTP错误: {str(e)}")
            if response.status_code == 401:
                logger.error("API密钥无效或已过期")
            return []
        except json.JSONDecodeError:
            logger.error("火山引擎API返回的不是有效的JSON格式")
            return []
        except Exception as e:
            logger.error(f"火山引擎API请求处理失败: {str(e)}")
            return []
            
    except ImportError:
        logger.error("请安装requests库: pip install requests")
        raise
    except Exception as e:
        logger.error(f"火山引擎API调用失败: {str(e)}")
        raise

def process_with_deepseek(prompt: str, model_name: str, api_key: Optional[str] = None) -> List[Dict[str, str]]:
    """使用DeepSeek API处理文本"""
    try:
        from openai import OpenAI
        
        if not api_key:
            logger.error("使用DeepSeek API需要提供API密钥")
            return []
        
        # 设置客户端
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # 默认使用deepseek-chat模型，除非指定其他模型
        model_to_use = model_name if model_name else "deepseek-chat"
        
        logger.info(f"使用DeepSeek API调用模型: {model_to_use}")
        
        try:
            # 创建聊天完成请求
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": "你是一个专业的问答提取助手，请根据提供的文本提取问答对。"},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            # 获取响应内容
            if response.choices and len(response.choices) > 0:
                response_text = response.choices[0].message.content
                return extract_qa_pairs_from_response(response_text)
            else:
                logger.warning("DeepSeek API返回的响应没有内容")
                return []
                
        except Exception as e:
            logger.error(f"DeepSeek API请求失败: {str(e)}")
            return []
            
    except ImportError:
        logger.error("请安装openai库: pip install openai>=1.0.0")
        raise
    except Exception as e:
        logger.error(f"DeepSeek API调用失败: {str(e)}")
        raise

def extract_qa_pairs_from_response(response_text: str) -> List[Dict[str, str]]:
    """从大模型响应中提取问答对"""
    try:
        # 检查并修复可能的编码问题
        if isinstance(response_text, str) and any(ord(c) > 127 for c in response_text) and '鍝' in response_text:
            # 可能是UTF-8字符串被错误地用GBK解码
            try:
                # 先将字符串编码为bytes，假设它是latin-1编码
                bytes_data = response_text.encode('latin-1')
                # 然后尝试用UTF-8解码
                response_text = bytes_data.decode('utf-8', errors='replace')
            except Exception as e:
                logger.warning(f"修复编码问题失败: {str(e)}")
        
        # 尝试找到JSON数组部分
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_text[start_idx:end_idx]
            qa_pairs = json.loads(json_str)
            
            # 验证格式
            for qa in qa_pairs:
                if not isinstance(qa, dict) or 'question' not in qa or 'answer' not in qa:
                    logger.warning("大模型返回的JSON格式不符合预期")
                    return []
            
            return qa_pairs
        else:
            logger.warning("无法在响应中找到JSON数组")
            return []
    except json.JSONDecodeError:
        logger.error("JSON解析失败，大模型返回的不是有效的JSON格式")
        logger.debug(f"原始响应: {response_text}")
        return []
    except Exception as e:
        logger.error(f"提取问答对时出错: {str(e)}")
        return [] 