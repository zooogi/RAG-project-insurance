# Reranker 使用文档

## 概述

`Reranker` 模块提供两阶段检索功能：使用 Faiss 进行高效的向量检索（Ranking），然后使用 bge-reranker-large 进行精细重排序（Reranking）。该模块专门为保险文档的语义检索优化，支持 metadata 加权，能够显著提升检索准确性。

## 核心特性

### 1. 两阶段检索流程
- **Ranking 阶段**：使用 Faiss 向量索引快速检索 Top-K 候选（默认 K=20）
- **Reranking 阶段**：使用 Cross-Encoder 模型对候选结果进行精细排序，返回 Top-N（默认 N=5）

### 2. 模型缓存机制
- 多个 Reranker 实例共享同一个模型，节省显存
- 自动检测并复用已加载的模型
- 支持手动清空缓存释放显存

### 3. Metadata 加权
- 利用章节路径（section_path）匹配查询关键词
- 表格和列表类型的 chunk 加权
- 标题级别加权（更高级别的标题可能更相关）

### 4. 国内镜像源支持
- 默认使用 hf-mirror.com 加速模型下载
- 支持本地模型路径加载

## 模型信息

- **Reranker 模型**: BAAI/bge-reranker-large
- **模型大小**: 约 1-2GB
- **设备支持**: 自动检测 GPU/CPU
- **镜像源**: 默认使用国内镜像（hf-mirror.com）

## 安装依赖

```bash
pip install faiss-cpu  # 或 faiss-gpu（如果有CUDA）
pip install transformers
```

## 快速开始

### 1. 基本使用

```python
from app.embedder import create_embedder
from app.reranker import create_reranker
from app.chunker import chunk_file

# 1. 创建 embedder
embedder = create_embedder()

# 2. 创建 reranker
reranker = create_reranker(embedder)

# 3. 加载 chunks（从 JSON 文件或直接创建）
chunks = reranker.load_chunks_from_json("data/chunks/保险文档_chunks.json")

# 4. 构建 Faiss 索引
reranker.build_index(chunks, index_type="flat")

# 5. 执行检索
query = "如何申请意外险理赔？"
results = reranker.search(
    query,
    rank_top_k=20,    # ranking 阶段返回 top-20
    rerank_top_k=5,   # reranking 阶段返回 top-5
    use_rerank=True
)

# 6. 查看结果
for chunk, score, info in results:
    print(f"分数: {score:.4f}")
    print(f"文本: {chunk.text[:100]}...")
    print(f"章节: {' > '.join(chunk.metadata.section_path)}")
```

### 2. 从 JSON 文件加载 Chunks

```python
# 从 chunks JSON 文件加载
chunks = reranker.load_chunks_from_json("data/chunks/文档_chunks.json")
```

### 3. 使用本地模型

```python
# 如果模型已下载到本地
reranker = create_reranker(
    embedder,
    reranker_model_path="/path/to/bge-reranker-large",
    use_mirror=False
)
```

### 4. 仅使用 Ranking（不使用 Reranking）

```python
results = reranker.search(
    query,
    rank_top_k=10,
    rerank_top_k=10,
    use_rerank=False  # 跳过 reranking
)
```

## API 参考

### Reranker 类

#### `__init__()`

```python
Reranker(
    embedder: Embedder,
    reranker_model_name: str = "BAAI/bge-reranker-large",
    reranker_model_path: Optional[str] = None,
    device: Optional[str] = None,
    use_metadata: bool = True,
    use_mirror: bool = True
)
```

**参数**:
- `embedder`: Embedder 实例，用于向量化
- `reranker_model_name`: reranker 模型名称
- `reranker_model_path`: 本地 reranker 模型路径
- `device`: 设备类型 ('cuda', 'cpu' 或 None 自动检测)
- `use_metadata`: 是否使用 metadata 进行加权
- `use_mirror`: 是否使用国内镜像源（默认 True）

#### `load_chunks_from_json()`

从 JSON 文件加载 chunks。

```python
chunks = reranker.load_chunks_from_json(json_path: Union[str, Path]) -> List[Chunk]
```

#### `build_index()`

构建 Faiss 索引。

```python
reranker.build_index(
    chunks: List[Chunk],
    chunk_embeddings: Optional[np.ndarray] = None,
    index_type: str = "flat"
)
```

**参数**:
- `chunks`: Chunk 列表
- `chunk_embeddings`: 预计算的 chunk 向量（可选）
- `index_type`: faiss 索引类型 ("flat" 或 "ivf")
  - `flat`: 精确搜索，适合小到中等规模数据
  - `ivf`: 近似搜索，适合大规模数据（>10万条）

#### `search()`

完整的检索流程：ranking + reranking。

```python
results = reranker.search(
    query: str,
    rank_top_k: int = 20,
    rerank_top_k: int = 5,
    use_rerank: bool = True
) -> List[Tuple[Chunk, float, Dict[str, Any]]]
```

**参数**:
- `query`: 查询文本
- `rank_top_k`: ranking 阶段返回的 top-k 数量
- `rerank_top_k`: reranking 阶段返回的 top-k 数量
- `use_rerank`: 是否使用 rerank

**返回**:
- List of (chunk, final_score, metadata) tuples
  - `chunk`: Chunk 对象
  - `final_score`: 最终分数（rerank 分数）
  - `metadata`: 包含 rank_score, rerank_score 和 chunk metadata

#### `rank()`

仅使用 Faiss 进行向量检索。

```python
results = reranker.rank(query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]
```

#### `rerank()`

仅使用 Cross-Encoder 进行重排序。

```python
results = reranker.rerank(
    query: str,
    chunks: List[Chunk],
    top_k: int = 5,
    batch_size: int = 16
) -> List[Tuple[Chunk, float]]
```

### 便捷函数

#### `create_reranker()`

创建 Reranker 实例的便捷函数。

```python
reranker = create_reranker(
    embedder: Embedder,
    reranker_model_name: str = "BAAI/bge-reranker-large",
    reranker_model_path: Optional[str] = None,
    use_mirror: bool = True,
    **kwargs
) -> Reranker
```

#### `clear_model_cache()`

清空模型缓存，释放显存。

```python
from app.reranker import clear_model_cache
clear_model_cache()
```

#### `get_model_cache_info()`

获取模型缓存信息。

```python
from app.reranker import get_model_cache_info
cache_info = get_model_cache_info()
print(cache_info)
# {'cached_models': ['BAAI/bge-reranker-large_cuda'], 'cache_count': 1}
```

## 完整示例

### 示例 1: 基本检索流程

```python
from app.embedder import create_embedder
from app.reranker import create_reranker

# 初始化
embedder = create_embedder()
reranker = create_reranker(embedder)

# 加载数据
chunks = reranker.load_chunks_from_json("data/chunks/保险文档_chunks.json")

# 构建索引
reranker.build_index(chunks, index_type="flat")

# 检索
query = "车险理赔需要准备什么材料？"
results = reranker.search(query, rank_top_k=20, rerank_top_k=5)

# 处理结果
for i, (chunk, score, info) in enumerate(results, 1):
    print(f"\n【排名 {i}】分数: {score:.4f}")
    print(f"Ranking分数: {info['rank_score']:.4f}")
    print(f"Rerank分数: {info['rerank_score']:.4f}")
    print(f"文本: {chunk.text[:150]}...")
    print(f"章节: {' > '.join(chunk.metadata.section_path)}")
```

### 示例 2: 对比 Ranking 和 Reranking

```python
query = "互联网保险的发展趋势"

# 仅 Ranking
rank_results = reranker.search(
    query,
    rank_top_k=10,
    rerank_top_k=10,
    use_rerank=False
)

# Ranking + Reranking
rerank_results = reranker.search(
    query,
    rank_top_k=20,
    rerank_top_k=5,
    use_rerank=True
)

# 对比结果
print("仅 Ranking 结果:")
for chunk, score, _ in rank_results:
    print(f"  {score:.4f}: {chunk.text[:80]}...")

print("\nRanking + Reranking 结果:")
for chunk, score, info in rerank_results:
    print(f"  {score:.4f}: {chunk.text[:80]}...")
```

### 示例 3: 使用预计算的 Embeddings

```python
# 如果已经计算好 embeddings
import numpy as np
embeddings = np.load("embeddings.npy")  # 形状: (n_chunks, embedding_dim)

# 直接使用预计算的 embeddings
reranker.build_index(chunks, chunk_embeddings=embeddings)
```

## 模型缓存机制

### 工作原理

Reranker 使用类级别的模型缓存，多个实例共享同一个模型：

```python
# 第一次创建 - 加载模型并缓存
reranker1 = create_reranker(embedder)
# 输出: "✓ 成功加载reranker模型..."
# 输出: "✓ 模型已缓存，后续实例将复用此模型（节省显存）"

# 第二次创建 - 复用缓存模型
reranker2 = create_reranker(embedder)
# 输出: "✓ 复用已缓存的reranker模型: BAAI/bge-reranker-large_cuda"
```

### 缓存管理

```python
from app.reranker import get_model_cache_info, clear_model_cache

# 查看缓存状态
cache_info = get_model_cache_info()
print(f"缓存数量: {cache_info['cache_count']}")
print(f"缓存的模型: {cache_info['cached_models']}")

# 清空缓存（释放显存）
clear_model_cache()
```

## 性能优化建议

### 1. 索引类型选择

- **小规模数据（<1万条）**: 使用 `index_type="flat"`（精确搜索）
- **中等规模数据（1-10万条）**: 使用 `index_type="flat"`（精确搜索）
- **大规模数据（>10万条）**: 使用 `index_type="ivf"`（近似搜索，更快）

### 2. Top-K 参数调优

- `rank_top_k`: 建议设置为 20-50，太小可能漏掉相关结果，太大影响性能
- `rerank_top_k`: 建议设置为 3-10，根据最终需要的结果数量调整

### 3. 批量处理

如果需要对多个查询进行检索，可以批量处理：

```python
queries = ["查询1", "查询2", "查询3"]
all_results = []

for query in queries:
    results = reranker.search(query, rank_top_k=20, rerank_top_k=5)
    all_results.append(results)
```

### 4. GPU 加速

如果有 GPU，reranker 会自动使用 GPU 加速：

```python
# 自动检测，如果有 GPU 会使用 GPU
reranker = create_reranker(embedder)

# 或手动指定
reranker = create_reranker(embedder, device='cuda')
```

## Metadata 加权策略

Reranker 支持基于 metadata 的加权，提升检索准确性：

1. **章节路径匹配**: 查询关键词出现在 section_path 中时加分（+0.1 分/关键词）
2. **表格类型**: 包含表格的 chunk 加分（+0.05 分）
3. **列表类型**: 包含列表的 chunk 加分（+0.03 分）
4. **标题级别**: 更高级别的标题（heading_level <= 2）加分（+0.02 分）

可以通过 `use_metadata=False` 禁用加权：

```python
reranker = create_reranker(embedder, use_metadata=False)
```

## 常见问题

### Q1: 首次运行很慢

**原因**: 首次运行需要下载 reranker 模型（约 1-2GB）

**解决**: 
- 耐心等待模型下载完成
- 使用国内镜像源（默认已启用）加速下载
- 模型会自动缓存，后续运行会很快

### Q2: 显存不足

**原因**: 多个实例重复加载模型

**解决**:
- 使用模型缓存机制（默认已启用）
- 多个实例会自动共享同一个模型
- 如需释放显存，调用 `clear_model_cache()`

### Q3: 检索结果不准确

**解决**:
- 调整 `rank_top_k` 和 `rerank_top_k` 参数
- 启用 metadata 加权（`use_metadata=True`）
- 检查 chunks 的质量和 metadata 信息

### Q4: 大规模数据检索慢

**解决**:
- 使用 `index_type="ivf"` 进行近似搜索
- 减少 `rank_top_k` 的值
- 使用 GPU 加速

### Q5: 如何保存和加载索引

**当前版本**: 索引存储在内存中，程序退出后丢失

**未来计划**: 支持保存 Faiss 索引到文件，下次直接加载

## 测试

运行测试脚本：

```bash
python scripts/test_reranker.py
```

测试脚本会：
1. 加载模型
2. 创建测试数据
3. 构建索引
4. 测试检索功能
5. 测试模型缓存机制

## 相关资源

- [Faiss 官方文档](https://github.com/facebookresearch/faiss)
- [BGE Reranker 模型](https://huggingface.co/BAAI/bge-reranker-large)
- [Transformers 文档](https://huggingface.co/docs/transformers)

## 更新日志

### v1.0.0 (2026-01-08)
- 初始版本
- 支持 Faiss 向量检索
- 支持 bge-reranker-large 重排序
- 支持模型缓存机制
- 支持 metadata 加权
- 支持国内镜像源
