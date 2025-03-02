import os
import json
import argparse
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv
from datetime import datetime
from tqdm.auto import tqdm

# 文件解析模块
from file_parsers import parse_file

# 大模型接口模块
from llm_interface import process_with_llm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

def setup_tqdm():
    """配置tqdm进度条"""
    # 设置tqdm的默认格式
    tqdm.monitor_interval = 0  # 禁用监视器线程
    # 返回一个tqdm实例，用于测试是否支持彩色输出
    return tqdm(range(1), leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')

# 初始化tqdm
setup_tqdm()

def get_files_in_directory(directory: str, extensions: List[str]) -> List[str]:
    """获取指定目录下特定扩展名的所有文件"""
    all_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                all_files.append(os.path.join(root, file))
    
    return all_files

def save_json_results(results: List[Dict[str, Any]], output_dir: str, filename: str, append: bool = False) -> None:
    """
    将结果保存为JSON文件
    
    Args:
        results: 要保存的结果
        output_dir: 输出目录
        filename: 文件名
        append: 是否追加到现有文件
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # 在保存前检查并修复可能的编码问题
    sanitized_results = sanitize_json_data(results)
    
    if append and os.path.exists(output_path):
        try:
            # 读取现有文件
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                
            # 确保existing_data是列表
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
                
            # 合并结果
            combined_results = existing_data + sanitized_results
            
            # 写入合并后的结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"追加到现有文件时出错 {output_path}: {str(e)}")
            # 如果追加失败，创建新文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_results, f, ensure_ascii=False, indent=2)
    else:
        # 直接写入新文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sanitized_results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"结果已保存到: {output_path}")

def sanitize_json_data(data):
    """
    清理JSON数据，修复可能的编码问题
    """
    if isinstance(data, dict):
        return {k: sanitize_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    elif isinstance(data, str):
        # 尝试修复可能的编码问题
        try:
            # 如果字符串看起来像是乱码，尝试修复
            if any(ord(c) > 127 for c in data) and '鍝' in data:
                # 可能是UTF-8字符串被错误地用GBK解码
                # 先将字符串编码为bytes，假设它是latin-1编码
                bytes_data = data.encode('latin-1')
                # 然后尝试用UTF-8解码
                return bytes_data.decode('utf-8', errors='replace')
            return data
        except Exception:
            # 如果修复失败，返回原始字符串
            return data
    else:
        return data

def split_text_into_chunks(text: str, chunk_size: int = 8000, overlap: int = 500) -> List[str]:
    """
    将长文本分割成多个重叠的块
    
    Args:
        text: 要分割的文本
        chunk_size: 每个块的最大字符数
        overlap: 相邻块之间的重叠字符数
        
    Returns:
        文本块列表
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # 确定当前块的结束位置
        end = min(start + chunk_size, len(text))
        
        # 如果不是最后一个块，尝试在句子或段落边界处分割
        if end < len(text):
            # 尝试在段落边界处分割
            paragraph_end = text.rfind('\n\n', start, end)
            if paragraph_end != -1 and paragraph_end > start + chunk_size // 2:
                end = paragraph_end + 2  # 包含换行符
            else:
                # 尝试在句子边界处分割
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('。', start, end),
                    text.rfind('？', start, end),
                    text.rfind('！', start, end)
                )
                if sentence_end != -1 and sentence_end > start + chunk_size // 2:
                    end = sentence_end + 2  # 包含标点和空格
        
        # 添加当前块
        chunks.append(text[start:end])
        
        # 更新下一个块的起始位置，考虑重叠
        start = max(start + 1, end - overlap)
    
    return chunks

def segment_chinese_text(text: str, max_segment_size: int = 8000) -> List[str]:
    """
    使用jieba库对中文文本进行智能分段
    
    Args:
        text: 要分段的中文文本
        max_segment_size: 每个段落的最大字符数
        
    Returns:
        分段后的文本列表
    """
    try:
        import jieba
        
        # 检测文本是否包含大量中文字符
        chinese_char_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_char_count < len(text) * 0.3:  # 如果中文字符少于30%，使用普通分段
            logger.info("文本中文字符比例较低，使用普通分段方法")
            return split_text_into_chunks(text, chunk_size=max_segment_size)
        
        logger.info("检测到中文文本，使用jieba进行中文智能分段")
        
        # 分词
        words = list(jieba.cut(text))
        
        # 根据分词结果和标点符号进行分段
        segments = []
        current_segment = ""
        
        # 中文句子结束标记
        sentence_ends = ['。', '！', '？', '；', '\n\n']
        
        for word in words:
            # 添加当前词
            if len(current_segment) + len(word) <= max_segment_size:
                current_segment += word
            else:
                # 如果添加当前词会超出最大长度，先保存当前段落
                segments.append(current_segment)
                current_segment = word
            
            # 检查是否是句子结束
            if any(current_segment.endswith(end) for end in sentence_ends):
                # 如果当前段落已经足够长，就在句子结束处分段
                if len(current_segment) >= max_segment_size // 2:
                    segments.append(current_segment)
                    current_segment = ""
        
        # 添加最后一个段落
        if current_segment:
            segments.append(current_segment)
        
        # 如果分段结果为空，回退到普通分段方法
        if not segments:
            logger.warning("中文智能分段失败，回退到普通分段方法")
            return split_text_into_chunks(text, chunk_size=max_segment_size)
        
        logger.info(f"中文智能分段完成，共分为 {len(segments)} 个段落")
        return segments
        
    except ImportError:
        logger.warning("未安装jieba库，使用普通分段方法")
        return split_text_into_chunks(text, chunk_size=max_segment_size)
    except Exception as e:
        logger.error(f"中文智能分段出错: {str(e)}")
        return split_text_into_chunks(text, chunk_size=max_segment_size)

def adaptive_split_text(text: str, min_chunk_size: int = 4000, max_chunk_size: int = 12000, 
                       overlap: int = 500, max_chunks: int = 10) -> List[str]:
    """
    根据文本内容自适应分割文本
    
    Args:
        text: 要分割的文本
        min_chunk_size: 最小块大小
        max_chunk_size: 最大块大小
        overlap: 相邻块之间的重叠字符数
        max_chunks: 最大分块数量
        
    Returns:
        文本块列表
    """
    # 如果文本较短，直接返回
    if len(text) <= max_chunk_size:
        return [text]
    
    # 估计理想的块大小，使分块数量不超过max_chunks
    ideal_chunk_size = max(min_chunk_size, min(max_chunk_size, len(text) // max_chunks))
    
    logger.info(f"文本长度: {len(text)}字符，自适应块大小: {ideal_chunk_size}字符")
    
    # 检测是否是中文文本
    chinese_char_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_char_count > len(text) * 0.3:  # 如果中文字符超过30%
        logger.info("检测到中文文本，使用jieba进行中文智能分段")
        chunks = segment_chinese_text(text, max_segment_size=ideal_chunk_size)
    else:
        # 使用理想块大小进行分割
        chunks = split_text_into_chunks(text, chunk_size=ideal_chunk_size, overlap=overlap)
    
    # 如果分块数量仍然过多，合并一些块
    if len(chunks) > max_chunks:
        logger.info(f"分块数量 ({len(chunks)}) 超过最大限制 ({max_chunks})，尝试合并块")
        merged_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) <= max_chunk_size:
                current_chunk += chunk if not current_chunk else "\n\n" + chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        logger.info(f"合并后的分块数量: {len(merged_chunks)}")
        return merged_chunks
    
    return chunks

def main():
    try:
        parser = argparse.ArgumentParser(description='文档解析与问答提取工具')
        parser.add_argument('--directory', '-d', type=str, help='要处理的文件目录')
        parser.add_argument('--output', '-o', type=str, help='输出结果的目录')
        parser.add_argument('--llm', '-l', type=str, 
                            choices=['bailian', 'volcengine', 'deepseek', 'ollama'], 
                            help='使用的大模型')
        parser.add_argument('--model', '-m', type=str, help='模型名称')
        parser.add_argument('--api-key', '-k', type=str, help='API密钥')
        parser.add_argument('--chunk-size', '-c', type=int, default=8000, help='文本分块大小')
        parser.add_argument('--overlap', type=int, default=500, help='文本分块重叠大小')
        parser.add_argument('--no-chunking', action='store_true', help='不进行文本分块，直接处理整个文件')
        
        args = parser.parse_args()
        
        # 从环境变量或命令行参数获取配置
        directory = args.directory or os.getenv('DOCUMENT_DIRECTORY')
        output_dir = args.output or os.getenv('OUTPUT_DIRECTORY', 'output')
        llm_provider = args.llm or os.getenv('LLM_PROVIDER', 'ollama')
        model_name = args.model or os.getenv('MODEL_NAME', 'llama3')
        api_key = args.api_key or os.getenv('API_KEY')
        chunk_size = args.chunk_size or int(os.getenv('CHUNK_SIZE', '8000'))
        overlap = args.overlap or int(os.getenv('OVERLAP', '500'))
        no_chunking = args.no_chunking or (os.getenv('NO_CHUNKING', 'false').lower() == 'true')
        
        # 验证必要参数
        if not directory:
            logger.error("未指定文件目录，请使用--directory参数或设置DOCUMENT_DIRECTORY环境变量")
            return 1
        
        if not os.path.exists(directory):
            logger.error(f"指定的目录不存在: {directory}")
            return 1
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有支持的文件
        supported_extensions = ['.txt', '.pdf', '.docx', '.doc']
        files = get_files_in_directory(directory, supported_extensions)
        
        if not files:
            logger.warning(f"在目录 {directory} 中未找到支持的文件")
            return 0
        
        logger.info(f"找到 {len(files)} 个文件需要处理")
        
        # 清空之前的进度条
        print("\n" * 5)
        
        # 存储所有结果
        all_results = []
        
        # 使用tqdm创建主进度条
        with tqdm(total=len(files), desc="处理文件", unit="文件") as main_pbar:
            for i, file_path in enumerate(files):
                logger.info(f"处理文件 [{i+1}/{len(files)}]: {file_path}")
                
                try:
                    # 解析文件内容
                    logger.info("正在解析文件...")
                    content = parse_file(file_path)
                    
                    if not content:
                        logger.warning(f"无法解析文件: {file_path}")
                        continue
                    
                    # 为每个文件创建单独的结果文件
                    file_name = os.path.basename(file_path)
                    result_file = f"{file_name}.json"
                    
                    # 检查是否需要分块处理
                    if len(content) > chunk_size and not no_chunking:
                        logger.info(f"文件 {file_path} 内容较长，进行自适应分块处理")
                        
                        # 根据文件大小估计最大分块数
                        file_size_mb = len(content) / 1000000  # 估计文件大小（MB）
                        max_chunks = max(3, min(20, int(file_size_mb * 2)))  # 根据文件大小动态调整最大分块数
                        
                        # 检测是否包含大量中文
                        chinese_char_count = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
                        is_chinese_text = chinese_char_count > len(content) * 0.3
                        
                        if is_chinese_text:
                            logger.info("检测到中文文档，将使用中文智能分段")
                        
                        logger.info("正在进行文本分段...")
                        chunks = adaptive_split_text(
                            content, 
                            min_chunk_size=chunk_size // 2,
                            max_chunk_size=chunk_size,
                            overlap=overlap,
                            max_chunks=max_chunks
                        )
                        
                        logger.info(f"文件被分为 {len(chunks)} 个块进行处理")
                        all_qa_pairs = []
                        
                        # 为每个块创建临时结果文件
                        temp_result_file = f"{file_name}_temp.json"
                        
                        # 定义内部函数，使用nonlocal正确引用外部变量
                        seen_questions = set()  # 用于跟踪已经处理过的问题

                        def save_chunk_result(chunk_qa_pairs):
                            """回调函数，用于保存块处理结果"""
                            nonlocal all_qa_pairs, j, chunks, file_path, output_dir, temp_result_file, seen_questions, chunk_pbar
                            
                            # 去重：只添加未见过的问题
                            new_pairs = []
                            for pair in chunk_qa_pairs:
                                question = pair.get('question', '')
                                if question and question not in seen_questions:
                                    seen_questions.add(question)
                                    new_pairs.append(pair)
                            
                            if new_pairs:
                                all_qa_pairs.extend(new_pairs)
                                
                                # 保存当前的所有结果
                                temp_result = {
                                    "file_path": file_path,
                                    "qa_pairs": all_qa_pairs,
                                    "processed_chunks": j + 1,
                                    "total_chunks": len(chunks),
                                    "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                }
                                save_json_results([temp_result], output_dir, temp_result_file)
                            
                            # 更新进度条
                            chunk_pbar.update(1)
                        
                        # 使用tqdm创建块处理进度条
                        with tqdm(total=len(chunks), desc=f"处理 {len(chunks)} 个文本块", position=0) as chunk_pbar:
                            for j, chunk in enumerate(chunks):
                                logger.info(f"处理第 {j+1}/{len(chunks)} 块")
                                chunk_qa_pairs = process_with_llm(
                                    chunk, 
                                    llm_provider=llm_provider, 
                                    model_name=model_name, 
                                    api_key=api_key,
                                    callback=save_chunk_result
                                )
                                # 如果回调未被调用（例如，当结果为None时），手动处理结果
                                if chunk_qa_pairs and not any(pair in all_qa_pairs for pair in chunk_qa_pairs):
                                    all_qa_pairs.extend(chunk_qa_pairs)
                                    chunk_pbar.update(1)
                        
                        qa_pairs = all_qa_pairs
                        
                        # 处理完所有块后，删除临时文件
                        temp_file_path = os.path.join(output_dir, temp_result_file)
                        if os.path.exists(temp_file_path):
                            try:
                                os.remove(temp_file_path)
                            except Exception as e:
                                logger.warning(f"无法删除临时文件 {temp_file_path}: {str(e)}")
                    else:
                        # 如果文件较小或者指定不分块，直接处理整个内容
                        if len(content) > chunk_size:
                            logger.info(f"文件 {file_path} 内容较长 ({len(content)} 字符)，但根据设置直接处理整个文件")
                        else:
                            logger.info(f"文件 {file_path} 内容较短 ({len(content)} 字符)，直接处理")
                        
                        # 使用大模型处理内容
                        logger.info("正在处理文本...")
                        qa_pairs = process_with_llm(content, llm_provider=llm_provider, 
                                                  model_name=model_name, api_key=api_key)
                    
                    if qa_pairs:
                        # 添加文件信息
                        result = {
                            "file_path": file_path,
                            "qa_pairs": qa_pairs,
                            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        all_results.append(result)
                        
                        # 为每个文件单独保存结果
                        logger.info("正在保存结果...")
                        save_json_results([result], output_dir, result_file)
                        
                        # 每处理完一个文件，就更新总结果文件
                        save_json_results(all_results, output_dir, "all_results.json")
                        
                        logger.info(f"文件 {file_path} 处理完成，已保存结果")
                    else:
                        logger.warning(f"大模型未能提取问答对: {file_path}")
                    
                    # 更新主进度条
                    main_pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"处理文件时出错 {file_path}: {str(e)}")
                    # 记录错误到专门的错误日志文件
                    error_log = {
                        "file_path": file_path,
                        "error": str(e),
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    save_json_results([error_log], output_dir, "errors.json", append=True)
                    
                    # 即使出错也更新主进度条
                    main_pbar.update(1)

        # 不需要在循环结束后再次保存所有结果，因为已经在每个文件处理后保存了
        logger.info("处理完成")
    except KeyboardInterrupt:
        logger.info("用户中断处理")
    except Exception as e:
        logger.critical(f"程序执行过程中发生严重错误: {str(e)}", exc_info=True)
        return 1
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 