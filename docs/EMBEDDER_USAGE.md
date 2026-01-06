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




## 相关资源

- [BGE模型官方仓库](https://github.com/FlagOpen/FlagEmbedding)
- [Sentence Transformers文档](https://www.sbert.net/)
- [HuggingFace模型页面](https://huggingface.co/BAAI/bge-large-zh-v1.5)
