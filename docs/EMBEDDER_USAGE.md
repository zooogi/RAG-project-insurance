# Embedder 使用说明

## 概述

本项目使用 **bge-large-zh-v1.5** 模型进行文本向量化，这是一个专为中文优化的高性能embedding模型，特别适合保险文档的语义检索。

## 模型信息

- **模型名称**: BAAI/bge-large-zh-v1.5
- **模型大小**: 约 1.3GB
- **向量维度**: 1024
- **最大序列长度**: 512 tokens
- **开发机构**: 北京智源人工智能研究院
- **缓存位置**: `C:\Users\47927\.cache\huggingface\hub\`

## 快速开始

### 1. 安装依赖

```bash
pip install sentence-transformers
```

### 2. 下载模型

首次使用时运行下载脚本：

```bash
python scripts/download_model.py
```

模型会自动下载到本地缓存目录，后续使用无需重复下载。

### 3. 基本使用

```python
from app.embedder import create_embedder

# 创建embedder实例
embedder = create_embedder()

# 编码单个文本
text = "保险理赔申请流程"
embedding = embedder.encode(text)
print(embedding.shape)  # (1, 1024)

# 编码多个文本
texts = [
    "意外伤害保险理赔流程说明",
    "重大疾病保险条款详解",
    "车险理赔所需材料清单"
]
embeddings = embedder.encode(texts)
print(embeddings.shape)  # (3, 1024)
```

## 高级用法

### 1. 查询-文档检索

```python
from app.embedder import create_embedder

embedder = create_embedder()

# 文档库
documents = [
    "意外伤害保险理赔流程说明",
    "重大疾病保险条款详解",
    "车险理赔所需材料清单",
    "人寿保险投保须知"
]

# 用户查询
query = "如何申请意外险理赔？"

# 编码（查询使用特殊前缀以提升检索效果）
doc_embeddings = embedder.encode_documents(documents)
query_embedding = embedder.encode_queries(query)

# 计算相似度
similarities = embedder.similarity(query_embedding, doc_embeddings)

# 找出最相关的文档
import numpy as np
most_similar_idx = np.argmax(similarities[0])
print(f"最相关文档: {documents[most_similar_idx]}")
print(f"相似度分数: {similarities[0][most_similar_idx]:.4f}")
```

### 2. 从本地路径加载模型

如果你已经下载了模型到本地目录：

```python
embedder = create_embedder(
    model_path="./models/bge-large-zh-v1.5",
    use_mirror=False
)
```

### 3. 使用GPU加速

如果有CUDA可用的GPU：

```python
embedder = create_embedder(device='cuda')
```

### 4. 批量处理优化

```python
# 大批量文本处理
large_text_list = [...]  # 假设有1000条文本

embeddings = embedder.encode(
    large_text_list,
    batch_size=64,  # 增大批次大小
    show_progress_bar=True  # 显示进度条
)
```

## API 参考

### Embedder 类

#### 初始化参数

- `model_name` (str): 模型名称，默认 "BAAI/bge-large-zh-v1.5"
- `model_path` (str): 本地模型路径，如果指定则从本地加载
- `device` (str): 设备类型 ('cuda', 'cpu' 或 None自动检测)
- `use_mirror` (bool): 是否使用国内镜像加速下载，默认 True

#### 主要方法

**encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)**
- 将文本编码为向量
- 参数:
  - `texts`: 单个文本或文本列表
  - `batch_size`: 批处理大小
  - `show_progress_bar`: 是否显示进度条
  - `normalize_embeddings`: 是否归一化向量（推荐开启）
- 返回: numpy数组 (n_texts, 1024)

**encode_queries(queries, **kwargs)**
- 编码查询文本（自动添加检索指令前缀）
- 参数: 查询文本或查询文本列表
- 返回: numpy数组

**encode_documents(documents, **kwargs)**
- 编码文档文本
- 参数: 文档文本或文档文本列表
- 返回: numpy数组

**similarity(embeddings1, embeddings2)**
- 计算两组向量之间的余弦相似度
- 参数:
  - `embeddings1`: 第一组向量 (n, 1024)
  - `embeddings2`: 第二组向量 (m, 1024)
- 返回: 相似度矩阵 (n, m)

**get_model_info()**
- 获取模型信息
- 返回: 包含模型详细信息的字典

## 性能优化建议

### 1. 内存优化

- 对于大规模文本，分批处理避免内存溢出
- 使用 `batch_size` 参数控制每批处理的文本数量

```python
def encode_large_dataset(texts, embedder, batch_size=100):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        embeddings = embedder.encode(batch)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)
```

### 2. 速度优化

- 使用GPU加速（如果可用）
- 增大 `batch_size` 提升吞吐量
- 预先计算文档向量并缓存

```python
import pickle

# 计算并保存文档向量
doc_embeddings = embedder.encode_documents(documents)
with open('doc_embeddings.pkl', 'wb') as f:
    pickle.dump(doc_embeddings, f)

# 后续直接加载
with open('doc_embeddings.pkl', 'rb') as f:
    doc_embeddings = pickle.load(f)
```

### 3. 离线部署

生产环境建议完全离线运行：

```python
from sentence_transformers import SentenceTransformer

# 加载时指定仅使用本地文件
model = SentenceTransformer(
    'BAAI/bge-large-zh-v1.5',
    local_files_only=True  # 仅使用本地缓存
)
```

## 常见问题

### Q1: 下载速度慢怎么办？

A: 代码已默认使用国内镜像 (https://hf-mirror.com)，如果仍然很慢，可以手动下载：

```bash
# 使用 huggingface-cli
pip install huggingface_hub
huggingface-cli download BAAI/bge-large-zh-v1.5 --local-dir ./models/bge-large-zh
```

### Q2: 如何查看模型缓存位置？

A: 运行以下代码：

```python
import os
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
print(f"模型缓存位置: {cache_dir}")
```

### Q3: 向量维度可以修改吗？

A: bge-large-zh-v1.5 的向量维度固定为 1024，无法修改。如需更小的维度，可以：
- 使用 bge-small-zh-v1.5 (512维)
- 使用 PCA 等降维方法

### Q4: 如何提升检索准确率？

A: 
1. 使用 `encode_queries()` 而非 `encode()` 编码查询文本
2. 确保 `normalize_embeddings=True`（默认已开启）
3. 对文档进行合理的分块（chunk）
4. 考虑使用重排序（reranker）模型进一步优化

## 示例项目

完整的使用示例请参考：
- `app/embedder.py` - 主模块实现
- `scripts/download_model.py` - 下载和测试脚本
- `scripts/test_embed.py` - 更多测试示例

## 相关资源

- [BGE模型官方仓库](https://github.com/FlagOpen/FlagEmbedding)
- [Sentence Transformers文档](https://www.sbert.net/)
- [HuggingFace模型页面](https://huggingface.co/BAAI/bge-large-zh-v1.5)
