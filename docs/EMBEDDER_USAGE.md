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

### 4. 模型缓存机制

Embedder 使用类级别的模型缓存，多个实例共享同一个模型，节省显存：

```python
# 第一次创建 - 加载模型并缓存
embedder1 = create_embedder()
# 输出: "✓ 成功加载模型..."
# 输出: "✓ 模型已缓存，后续实例将复用此模型（节省显存）"

# 第二次创建 - 复用缓存模型
embedder2 = create_embedder()
# 输出: "✓ 复用已缓存的embedder模型: BAAI/bge-large-zh-v1.5_cuda"
```

**缓存管理**:

```python
from app.embedder import get_model_cache_info, clear_model_cache

# 查看缓存状态
cache_info = get_model_cache_info()
print(f"缓存数量: {cache_info['cache_count']}")
print(f"缓存的模型: {cache_info['cached_models']}")

# 清空缓存（释放显存）
clear_model_cache()
```

**优势**:
- 多个实例共享同一个模型，节省显存
- 复用已加载模型，无需重复加载
- 自动管理，无需手动操作

### 5. 国内镜像源

默认使用国内镜像源（hf-mirror.com）加速模型下载：

```python
# 默认使用国内镜像
embedder = create_embedder()

# 不使用镜像
embedder = create_embedder(use_mirror=False)
```


## 相关资源

- [BGE模型官方仓库](https://github.com/FlagOpen/FlagEmbedding)
- [Sentence Transformers文档](https://www.sbert.net/)
- [HuggingFace模型页面](https://huggingface.co/BAAI/bge-large-zh-v1.5)
