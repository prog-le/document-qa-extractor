# 文档解析与问答提取工具

这是一个强大的文档解析与问答提取工具，能够自动从各种文档中提取问答对，支持多种大模型API和文件格式。

## 本项目用途
本项目旨在提供一个高效、易用的文档解析和问答提取解决方案，适用于研究人员、开发者和数据科学家等需要从文档中提取信息的用户。通过集成多种大模型API，用户可以快速获取高质量的问答对，提升工作效率。
例如：大模型训练数据集、知识竞赛出题、文档自动问答、智能客服等场景。

## 功能特点

- **多种文件格式支持**：PDF、TXT、DOCX、DOC
- **多种大模型集成**：
  - Ollama (本地模型)
  - 百炼 (阿里云)
  - 火山引擎
  - DeepSeek
- **智能文本分段**：
  - 自适应分块大小
  - 中文智能分段 (基于jieba分词)
  - 保持语义完整性
- **实时结果保存**：处理过程中实时保存结果，防止中断导致数据丢失
- **去重机制**：避免重复的问答对
- **编码问题自动修复**：自动检测和修复中文编码问题
- **进度可视化**：使用进度条显示各环节处理进度

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/prog-le/document-qa-extractor.git
cd document-qa-extractor
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 文件并重命名为 `.env`，然后根据需求进行配置：

```
# 文档目录和输出设置
DOCUMENT_DIRECTORY=/path/to/your/documents
OUTPUT_DIRECTORY=output

# 大模型设置
LLM_PROVIDER=ollama  # 可选: bailian, volcengine, deepseek, ollama
MODEL_NAME=llama3
API_KEY=your_api_key_here  # 对于需要API密钥的服务

# 文本处理设置
CHUNK_SIZE=8000
OVERLAP=500
NO_CHUNKING=false  # 设置为true时不进行文本分块
```

## 使用方法

### 基本用法

```bash
python document_qa_extractor.py
```

这将使用 `.env` 文件中的配置处理文档。

### 命令行参数

您也可以通过命令行参数覆盖环境变量：

```bash
python document_qa_extractor.py --directory /path/to/docs --output ./results --llm bailian --model qwen-omni-turbo --api-key your_api_key
```

### 参数说明

- `--directory` 或 `-d`: 要处理的文件目录
- `--output` 或 `-o`: 输出结果的目录
- `--llm` 或 `-l`: 使用的大模型提供商 (bailian, volcengine, deepseek, ollama)
- `--model` 或 `-m`: 模型名称
- `--api-key` 或 `-k`: API密钥
- `--chunk-size` 或 `-c`: 文本分块大小 (默认8000)
- `--overlap`: 文本分块重叠大小 (默认500)
- `--no-chunking`: 不进行文本分块，直接处理整个文件

## 大模型配置

### Ollama (本地模型)

```
LLM_PROVIDER=ollama
MODEL_NAME=llama3
# API_KEY不需要
```

确保已安装Ollama并下载了相应模型：
```bash
ollama pull llama3
```

### 百炼 (阿里云)

```
LLM_PROVIDER=bailian
MODEL_NAME=qwen-omni-turbo
API_KEY=your_bailian_api_key_here
```

### 火山引擎

```
LLM_PROVIDER=volcengine
MODEL_NAME=doubao-pro-32k-240615
API_KEY=your_volcengine_api_key_here
```

### DeepSeek

```
LLM_PROVIDER=deepseek
MODEL_NAME=deepseek-chat
API_KEY=your_deepseek_api_key_here
```

## 文本分段策略

### 普通分段

对于一般文本，系统会在段落或句子边界处进行分段，尽量保持语义完整性。

### 中文智能分段

对于中文文本（中文字符比例超过30%），系统会使用jieba分词库进行智能分段，在句子结束处（句号、感叹号、问号等）分段，确保语义完整性。

### 不分段处理

如果您希望直接处理整个文件而不进行分段，可以使用 `--no-chunking` 参数或在 `.env` 文件中设置 `NO_CHUNKING=true`。

## 输出结果

系统会生成以下输出文件：

1. `{filename}.json`: 每个处理的文件都会生成一个对应的JSON结果文件
2. `all_results.json`: 包含所有处理结果的汇总文件
3. `errors.json`: 记录处理过程中的错误信息

输出格式示例：

```json
[
  {
    "file_path": "/path/to/document.pdf",
    "qa_pairs": [
      {
        "question": "什么是文档解析与问答提取工具？",
        "answer": "文档解析与问答提取工具是一个能够自动从各种文档中提取问答对的工具，支持多种大模型API和文件格式。"
      },
      {
        "question": "该工具支持哪些文件格式？",
        "answer": "该工具支持PDF、TXT、DOCX、DOC等多种文件格式。"
      }
    ],
    "processed_time": "2023-06-01 12:34:56"
  }
]
```
更多例子见 `output` 目录。

## 常见问题解决

### 文件解析问题

1. **PDF文件无法解析**:
   - 确保PDF文件不是扫描版或图片版
   - 尝试使用其他工具预处理PDF

2. **DOC文件处理失败**:
   - 确保已安装Pandoc: `pandoc --version`
   - 如果Pandoc安装正确但仍然失败，系统会自动尝试其他解析方法

### 大模型API问题

1. **API密钥错误**:
   - 检查API密钥是否正确
   - 确认API密钥是否过期

2. **模型名称错误**:
   - 参考各大模型提供商的官方文档，确认正确的模型名称

3. **请求超时**:
   - 增加超时设置
   - 减小文本块大小

### 中文编码问题

如果遇到中文乱码问题，系统会尝试自动修复。如果仍有问题，可以尝试：

1. 确保源文件使用UTF-8编码
2. 检查输出目录的文件系统是否支持UTF-8

## 高级用法

### 自定义提示模板

您可以修改 `llm_interface.py` 中的 `QA_PAIRS_HUMAN_PROMPT` 变量来自定义提示模板，以获取更好的问答对提取效果。

### 添加新的大模型支持

如果您需要添加其他大模型的支持，可以参考现有的实现，在 `llm_interface.py` 中添加新的处理函数。

## 贡献

欢迎提交问题报告、功能请求和代码贡献。请遵循以下步骤：

1. Fork 仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 致谢

- 感谢所有开源库的贡献者
- 特别感谢 jieba 分词库提供中文分词支持
- 感谢 Pandoc 和 pypandoc 提供强大的文档转换功能
- 感谢各大模型提供商提供的API服务
