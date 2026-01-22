# RAG Pipeline 使用指南

## 概述

`RAGPipeline` 是一个完整的RAG（Retrieval-Augmented Generation）流程管道，串联了从文档处理到答案生成的所有步骤：

```
OCR → 文本清洗 → Chunk分块 → Embedding索引构建 → 检索 → Rerank → LLM生成答案
```

## 快速开始

### 0. 快速测试脚本

项目提供了完整的测试脚本 `scripts/Full_RAG_TEST.py`，包含：
- 文档处理流程测试
- PDF保险条款+表格测试
- CSV表格数据测试
- 图片OCR测试

查看测试脚本：
```bash
cat scripts/Full_RAG_TEST.py
```

### 1. 基本使用

```python
from app.main import RAGPipeline, RAGPipelineConfig
from pathlib import Path

# 使用默认配置创建pipeline
pipeline = RAGPipeline()

# 处理文档（OCR -> 清洗 -> Chunk -> 构建索引）
result = pipeline.process_documents(
    input_path=Path("data/raw_data"),  # 可选，默认使用config中的raw_data_dir
    overwrite=False  # 是否覆盖已存在的处理结果
)

# 查询答案
answer = pipeline.query("如何申请意外险理赔？")
print(answer["answer"])
```

### 2. 自定义配置

```python
from app.config import RAGPipelineConfig
from app.main import RAGPipeline

# 创建自定义配置
config = RAGPipelineConfig(
    # 路径配置
    raw_data_dir=Path("data/raw_data"),
    chunks_dir=Path("data/chunks"),
    
    # Chunker配置
    chunker_target_size=1000,
    enable_semantic_splitting=True,
    
    # Reranker配置
    reranker_rank_top_k=30,
    reranker_rerank_top_k=5,
    
    # LLM配置
    llm_model_name="Qwen/Qwen2.5-3B-Instruct",
    llm_temperature=0.5,
    # 显存优化选项
    llm_load_in_4bit=True  # 启用4bit量化，显存从6-8GB降到1.5-2GB
)

# 使用自定义配置创建pipeline
pipeline = RAGPipeline(config)
```

### 3. 从已有chunks加载

如果chunks已经生成，可以直接加载：

```python
pipeline = RAGPipeline()

# 从chunks目录加载所有chunk文件并构建索引
chunks = pipeline.load_chunks_from_files()

# 或者指定特定的chunk文件
chunks = pipeline.load_chunks_from_files([
    Path("data/chunks/保险基础知多少_chunks.json"),
    Path("data/chunks/友邦保险-寿险说明书_chunks.json")
])

# 查询
answer = pipeline.query("重疾险包含哪些疾病？")
```

## 完整流程示例

### 示例1: 处理单个PDF文件

```python
from app.main import RAGPipeline
from pathlib import Path

pipeline = RAGPipeline()

# 处理单个PDF文件
result = pipeline.process_documents(
    input_path=Path("data/raw_data/保险基础知多少.pdf"),
    overwrite=False
)

print(f"生成了 {len(result['chunks'])} 个chunks")
print(f"Chunk文件保存在: {result['chunk_files']}")
```

### 示例2: 批量处理目录

```python
pipeline = RAGPipeline()

# 批量处理raw_data目录下的所有文件
result = pipeline.process_documents(
    input_path=Path("data/raw_data"),
    overwrite=False
)

print(f"共处理 {len(result['ocr_results'])} 个文件")
print(f"共生成 {len(result['chunks'])} 个chunks")
```

### 示例3: 查询并获取详细结果

```python
pipeline = RAGPipeline()

# 加载chunks（如果索引未构建）
if not pipeline.index_built:
    pipeline.load_chunks_from_files()

# 查询并返回检索到的chunks
result = pipeline.query(
    "意外险理赔需要准备哪些材料？",
    use_rerank=True,
    return_chunks=True  # 返回检索到的chunks
)

print(f"答案: {result['answer']}")
print(f"\n使用的chunks数量: {result['num_chunks_used']}")

# 查看检索到的chunks
if 'chunks' in result:
    for i, chunk_info in enumerate(result['chunks'], 1):
        print(f"\nChunk {i}:")
        print(f"  分数: {chunk_info['score']:.4f}")
        print(f"  章节: {' > '.join(chunk_info['section_path'])}")
        print(f"  文本预览: {chunk_info['text'][:100]}...")
```

## 命令行使用

### 处理文档

```bash
# 处理raw_data目录下的所有文件
python -m app.main process --input data/raw_data

# 处理单个文件
python -m app.main process --input data/raw_data/保险基础知多少.pdf

# 覆盖已存在的处理结果
python -m app.main process --input data/raw_data --overwrite
```

### 加载chunks

```bash
# 从默认chunks目录加载
python -m app.main load

# 从指定目录加载
python -m app.main load --chunks-dir data/chunks
```

### 查询

```bash
# 查询（会自动加载chunks如果索引未构建）
python -m app.main query --query "如何申请意外险理赔？"
```

### 使用配置文件

创建 `config.json`:

```json
{
  "chunker_target_size": 1000,
  "reranker_rank_top_k": 30,
  "reranker_rerank_top_k": 5,
  "llm_model_name": "Qwen/Qwen2.5-7B-Instruct"
}
```

使用配置文件：

```bash
python -m app.main process --input data/raw_data --config config.json
```

## 配置说明

### RAGPipelineConfig 参数

#### 路径配置
- `raw_data_dir`: 原始数据目录（默认: `data/raw_data`）
- `processed_dir`: OCR处理后文件目录（默认: `data/processed`）
- `cleaned_dir`: 清洗后文件目录（默认: `data/cleaned`）
- `chunks_dir`: Chunks文件目录（默认: `data/chunks`）

#### OCR配置
- `ocr_source`: MineRU模型源（`'modelscope'` 或 `'huggingface'`）
- `ocr_use_gpu`: 是否使用GPU
- `ocr_extract_images`: 是否提取图片
- `ocr_extract_tables`: 是否提取表格

#### 文本清洗配置
- `enable_text_cleaning`: 是否启用文本清洗
- `text_cleaner_repeat_threshold`: 重复话术阈值（默认: 3）

#### Chunker配置
- `chunker_target_size`: 目标chunk大小（默认: 800）
- `chunker_max_size`: 最大chunk大小（默认: 1500）
- `enable_semantic_splitting`: 是否启用语义切割
- `enable_terminology`: 是否启用术语提取

#### Embedder配置
- `embedder_model_name`: Embedding模型名称（默认: `"BAAI/bge-large-zh-v1.5"`）
- `embedder_model_path`: 本地模型路径（可选）

#### Reranker配置
- `reranker_model_name`: Reranker模型名称（默认: `"BAAI/bge-reranker-large"`）
- `reranker_rank_top_k`: Ranking阶段返回的top-k（默认: 20）
- `reranker_rerank_top_k`: Reranking阶段返回的top-k（默认: 5）

#### LLM配置
- `llm_model_name`: LLM模型名称（默认: `"Qwen/Qwen2.5-3B-Instruct"`）
  - 可选：`"Qwen/Qwen2.5-1.5B-Instruct"`（约3-4GB显存）
  - 可选：`"Qwen/Qwen2.5-7B-Instruct"`（约14GB显存）
- `llm_device`: 设备类型（默认: `None`自动检测，`'cpu'`强制使用CPU）
- `llm_max_new_tokens`: 最大生成token数（默认: 512）
- `llm_temperature`: 生成温度（默认: 0.7）
- `llm_max_context_length`: RAG prompt的最大上下文长度（默认: 3000）
- `llm_load_in_8bit`: 使用8bit量化（默认: `False`，显存减半，需要bitsandbytes）
- `llm_load_in_4bit`: 使用4bit量化（默认: `False`，显存减少75%，需要bitsandbytes）

## 工作流程

### 1. 文档处理流程

```
raw_data/*.pdf
  ↓ [OCR处理]
processed/**/*.md
  ↓ [文本清洗，可选]
cleaned/**/*.md
  ↓ [Chunker分块]
chunks/*_chunks.json
  ↓ [构建Embedding索引]
索引就绪，可以查询
```

### 2. 查询流程

```
用户查询
  ↓ [Embedding编码]
查询向量
  ↓ [Faiss向量检索]
Ranking结果（top-k）
  ↓ [Reranker重排序]
Reranking结果（top-k）
  ↓ [构建RAG Prompt]
Prompt
  ↓ [LLM生成]
答案
```

## 状态检查

```python
# 检查pipeline状态
status = pipeline.get_status()
print(status)
# {
#   "index_built": True,
#   "num_chunks": 150,
#   "components_loaded": {
#     "ocr_processor": False,
#     "chunker": True,
#     "embedder": True,
#     "reranker": True,
#     "llm": False
#   }
# }
```

## 显存优化

### 方案1: 使用更小的模型（最简单）

```python
config = RAGPipelineConfig(
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct"  # 显存从6-8GB降到3-4GB
)
```

### 方案2: 使用量化（推荐）

```bash
# 安装量化库
pip install bitsandbytes
```

```python
# 使用8bit量化（显存减半）
config = RAGPipelineConfig(
    llm_load_in_8bit=True  # 显存从6-8GB降到3-4GB
)

# 或使用4bit量化（显存减少75%）
config = RAGPipelineConfig(
    llm_load_in_4bit=True  # 显存从6-8GB降到1.5-2GB
)
```

### 方案3: 组合优化（最佳效果）

```python
config = RAGPipelineConfig(
    llm_model_name="Qwen/Qwen2.5-1.5B-Instruct",  # 更小的模型
    llm_load_in_4bit=True  # 4bit量化
    # 结果：显存需求约1-1.5GB
)
```

### 方案4: 使用CPU（无需GPU显存）

```python
config = RAGPipelineConfig(
    llm_device="cpu"  # 速度较慢但不需要显存
)
```

### 显存需求对比

| 配置 | 显存需求 | 速度 | 质量 |
|------|---------|------|------|
| Qwen2.5-3B (FP16) | 6-8GB | 快 | 高 |
| Qwen2.5-3B (8bit) | 3-4GB | 快 | 高 |
| Qwen2.5-3B (4bit) | 1.5-2GB | 快 | 中高 |
| Qwen2.5-1.5B (FP16) | 3-4GB | 快 | 中 |
| Qwen2.5-1.5B (4bit) | 1-1.5GB | 快 | 中 |
| CPU模式 | 0GB | 慢 | 高 |

## 注意事项

1. **首次运行**：首次运行需要下载模型，可能需要较长时间
   - Embedder模型：约1GB
   - Reranker模型：约1GB
   - LLM模型：约6GB（Qwen2.5-3B）

2. **显存需求**：
   - Qwen2.5-3B-Instruct：约6-8GB显存（默认）
   - Qwen2.5-1.5B-Instruct：约3-4GB显存（显存不足时推荐）
   - Qwen2.5-7B-Instruct：约14GB显存（需要大显存）
   - **使用量化可以大幅降低显存需求**（见上方显存优化章节）

3. **量化要求**：
   - 需要安装 `bitsandbytes`：`pip install bitsandbytes`
   - 仅支持CUDA设备（不支持CPU）
   - 需要NVIDIA GPU（支持CUDA）

4. **模型缓存**：所有模型都支持缓存机制，多个实例共享同一个模型，节省显存

5. **索引构建**：`process_documents()` 会自动构建索引，也可以使用 `load_chunks_from_files()` 从已有chunks构建

6. **文件组织**：
   - OCR输出保存在 `processed/` 目录
   - 清洗后的文本保存在 `cleaned/` 目录（如果启用）
   - Chunks保存在 `chunks/` 目录

## 测试用例

项目提供了完整的测试脚本 `scripts/Full_RAG_TEST.py`，包含以下测试用例：

1. **文档处理流程测试**（embedding之前的流程）
   ```bash
   # 前提：data/raw_data 目录下有原始数据
   python -m app.main process --input data/raw_data
   ```

2. **PDF保险条款+表格测试**
   ```bash
   python -m app.main query --query "如果我今年45岁，友邦终身寿险可以选择哪些付费年限？"
   ```

3. **CSV表格数据测试**
   ```bash
   python -m app.main query --query "吸烟者 (smoker=yes) 的平均保险费用(Average charges)是多少？"
   ```

4. **图片OCR测试**
   ```bash
   python -m app.main query --query "在知识库里的一份图片合同说明，这款附加合同的保险期间是多久"
   ```

## 常见问题

### Q: 如何只处理文档而不构建索引？

A: 使用 `chunker.chunk_directory()` 直接处理，不通过 `RAGPipeline.process_documents()`。

### Q: 如何更新索引？

A: 重新调用 `process_documents()` 或 `load_chunks_from_files()`。

### Q: 如何释放显存？

A: 各个模块都提供了 `clear_model_cache()` 函数：

```python
from app.embedder import clear_model_cache as clear_embedder_cache
from app.reranker import clear_model_cache as clear_reranker_cache
from app.llm import clear_model_cache as clear_llm_cache

clear_embedder_cache()
clear_reranker_cache()
clear_llm_cache()
```

### Q: 如何只使用部分功能？

A: 可以单独使用各个模块，不需要通过 `RAGPipeline`：

```python
# 只使用OCR
from app.ocr import DocumentProcessor
processor = DocumentProcessor()
result = processor.process_file("file.pdf")

# 只使用Chunker
from app.chunker import SemanticChunker
chunker = SemanticChunker()
chunks = chunker.chunk_markdown_file("file.md")

# 只使用检索
from app.reranker import create_reranker
from app.embedder import create_embedder
embedder = create_embedder()
reranker = create_reranker(embedder)
reranker.build_index(chunks)
results = reranker.search("查询")
```

## 更多信息

- [OCR使用指南](OCR_USAGE.md)
- [文本清洗使用指南](TEXT_CLEANER_USAGE.md)
- [Chunker使用指南](CHUNKER_USAGE.md)
- [Embedder使用指南](EMBEDDER_USAGE.md)
- [Reranker使用指南](RERANKER_USAGE.md)
- [LLM使用指南](LLM_USAGE.md)
