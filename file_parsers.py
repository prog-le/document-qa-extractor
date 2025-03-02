import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def parse_file(file_path: str) -> Optional[str]:
    """
    根据文件类型解析文件内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件内容字符串，如果解析失败则返回None
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.txt':
            return parse_txt(file_path)
        elif file_extension == '.pdf':
            return parse_pdf(file_path)
        elif file_extension in ['.docx', '.doc']:
            # 优先使用pypandoc解析
            content = parse_doc_with_pandoc(file_path)
            if content:
                return content
            # 如果pypandoc失败，回退到传统方法
            return parse_doc(file_path)
        else:
            logger.warning(f"不支持的文件类型: {file_extension}")
            return None
    except Exception as e:
        logger.error(f"解析文件 {file_path} 时出错: {str(e)}")
        return None

def parse_txt(file_path: str) -> str:
    """解析TXT文件"""
    try:
        # 尝试使用UTF-8编码读取
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # 如果UTF-8失败，尝试使用GBK编码（常用于中文Windows系统）
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # 最后尝试使用latin-1编码（可以读取任何文件，但可能导致中文乱码）
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                logger.warning(f"文件 {file_path} 使用了latin-1编码，可能导致中文显示异常")
                return content
            except Exception as e:
                logger.error(f"解析TXT文件失败: {str(e)}")
                return ""

def parse_pdf(file_path: str) -> str:
    """解析PDF文件"""
    try:
        import PyPDF2
        
        content = []
        with open(file_path, 'rb') as f:
            try:
                pdf_reader = PyPDF2.PdfReader(f)
                # 添加页数检查
                if len(pdf_reader.pages) == 0:
                    logger.warning(f"PDF文件 {file_path} 没有页面")
                    return ""
                    
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    # 检查提取的文本是否为空
                    if text:
                        content.append(text)
                    else:
                        logger.warning(f"PDF文件 {file_path} 第 {page_num+1} 页无法提取文本")
            except PyPDF2.errors.PdfReadError as e:
                logger.error(f"PDF文件 {file_path} 读取错误: {str(e)}")
                return ""
        
        if not content:
            logger.warning(f"PDF文件 {file_path} 未能提取到任何文本")
            return ""
            
        return '\n'.join(content)
    except ImportError:
        logger.error("请安装PyPDF2库: pip install PyPDF2")
        raise

def parse_doc_with_pandoc(file_path: str) -> Optional[str]:
    """使用pypandoc解析DOC/DOCX文件"""
    try:
        import pypandoc
        
        logger.info(f"使用pypandoc解析文件: {file_path}")
        
        # 将文档转换为纯文本
        content = pypandoc.convert_file(file_path, 'plain', format=None)
        
        if content:
            return content
        else:
            logger.warning(f"pypandoc未能提取到内容: {file_path}")
            return None
    except ImportError:
        logger.warning("未安装pypandoc库，请安装: pip install pypandoc")
        return None
    except Exception as e:
        logger.warning(f"使用pypandoc解析失败: {str(e)}")
        return None

def parse_doc(file_path: str) -> str:
    """解析DOC/DOCX文件（传统方法，作为备选）"""
    if file_path.endswith('.docx'):
        try:
            import docx
            
            doc = docx.Document(file_path)
            content = []
            for para in doc.paragraphs:
                content.append(para.text)
            
            return '\n'.join(content)
        except ImportError:
            logger.error("请安装python-docx库: pip install python-docx")
            raise
    else:  # .doc 文件
        # 尝试多种方法处理.doc文件
        methods_tried = []
        
        # 方法1: 使用docx2txt (更轻量级的替代方案)
        try:
            import docx2txt
            methods_tried.append("docx2txt")
            text = docx2txt.process(file_path)
            if text:
                return text
        except ImportError:
            logger.warning("docx2txt未安装，尝试其他方法")
        except Exception as e:
            logger.warning(f"使用docx2txt处理失败: {str(e)}")
        
        # 方法2: 使用antiword (如果系统中已安装)
        try:
            import subprocess
            methods_tried.append("antiword")
            result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception as e:
            logger.warning(f"使用antiword处理失败: {str(e)}")
        
        # 方法3: 使用catdoc (另一个常用工具)
        try:
            import subprocess
            methods_tried.append("catdoc")
            result = subprocess.run(['catdoc', file_path], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception as e:
            logger.warning(f"使用catdoc处理失败: {str(e)}")
        
        # 方法4: 最后尝试textract (如果已安装)
        try:
            import textract
            methods_tried.append("textract")
            text = textract.process(file_path).decode('utf-8')
            if text:
                return text
        except Exception as e:
            logger.warning(f"使用textract处理失败: {str(e)}")
        
        # 所有方法都失败
        logger.error(f"无法解析.doc文件 {file_path}，尝试了以下方法: {', '.join(methods_tried)}")
        logger.error("请安装以下工具之一: docx2txt, antiword, catdoc 或 textract")
        logger.error("例如: pip install docx2txt")
        logger.error("或者: sudo apt-get install antiword / brew install antiword")
        return "" 