# Chunker 使用文档

## 概述

`SemanticChunker` 是一个基于语义的智能文档分块器，专门用于处理 OCR 产出的 Markdown 文件。它能够保留文档结构、识别表格和图片，并生成结构化的 chunk 输出。

> 📋 **数据目录说明**：处理后的chunks会保存到 `data/chunks/` 目录，详细说明请查看 [数据目录结构说明文档](../docs/DATA_STRUCTURE.md)

## 核心特性

### 1. 仅以 OCR 产出的 Markdown 作为输入源
- 支持标准 Markdown 格式
- 支持 HTML 表格（`<table>` 标签）
- 支持图片引用（`![](path)` 格式）

### 2. 最小必要的文本规范化
- 压缩连续空行（超过2个空行压缩为2个）
- 移除行尾空格
- 保留其他所有格式和结构

### 3. 保留文档结构上下文
- 跟踪标题层级（H1-H6）
- 记录每个 chunk 所属的章节路径
- 维护文档的语义层次结构

### 4. 优先保证语义完整性
- 不会在句子中间拆分
- 相同章节下的小块可以合并
- 智能处理超长段落

### 5. 表格作为不可拆分的原子单元
- 表格始终作为独立 chunk
- 保留完整的表格结构
- 添加 `has_table` 标记

### 6. 识别并记录图片引用
- 提取图片路径
- 记录在 chunk metadata 中
- 不将图片内容参与 embedding

### 7. 统一的结构化输出格式
- 每个 chunk 包含唯一 ID
- 丰富的 metadata 信息
- 便于后续检索和 reranking

## 安装和导入

```python
from app.chunker import SemanticChunker, chunk_file, chunk_directory
from pathlib import Path
```

## 基本用法

### 1. 处理单个文件

```python
from app.chunker import SemanticChunker
from pathlib import Path

# 创建 chunker 实例
chunker = SemanticChunker(
    target_chunk_size=800,    # 目标 chunk 大小（字符数）
    max_chunk_size=1500,      # 最大 chunk 大小
    min_chunk_size=200,       # 最小 chunk 大小
    overlap_size=100          # chunk 之间的重叠大小（暂未使用）
)

# 处理单个文件
md_file = Path("data/processed/保险基础知多少/保险基础知多少/hybrid_auto/保险基础知多少.md")
chunks = chunker.chunk_markdown_file(md_file)

# 查看结果
print(f"生成了 {len(chunks)} 个 chunks")
for chunk in chunks[:3]:
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"类型: {chunk.metadata.chunk_type}")
    print(f"章节: {' > '.join(chunk.metadata.section_path)}")
    print(f"文本: {chunk.text[:100]}...")
    print()
```

### 2. 批量处理目录

```python
from app.chunker import SemanticChunker
from pathlib import Path

chunker = SemanticChunker()

# 批量处理目录下的所有 Markdown 文件
input_dir = Path("data/processed")
output_dir = Path("data/chunks")

results = chunker.chunk_directory(
    input_dir=input_dir,
    output_dir=output_dir,  # 如果提供，会自动保存 JSON
    pattern="**/*.md"       # 文件匹配模式
)

# 查看结果
for file_path, chunks in results.items():
    print(f"{file_path}: {len(chunks)} chunks")
```

### 3. 使用便捷函数

```python
from app.chunker import chunk_file, chunk_directory

# 处理单个文件
chunks = chunk_file(
    "data/processed/保险基础知多少/保险基础知多少/hybrid_auto/保险基础知多少.md",
    target_chunk_size=800,
    max_chunk_size=1500
)

# 批量处理
results = chunk_directory(
    input_dir="data/processed",
    output_dir="data/chunks",
    target_chunk_size=800
)
```

### 4. 命令行使用

```bash
# 处理单个文件
python app/chunker.py data/processed/保险基础知多少/保险基础知多少/hybrid_auto/保险基础知多少.md output.json

# 批量处理目录
python app/chunker.py data/processed data/chunks
```

## 输出文件说明

### 输出位置

Chunker处理后的结果会保存到 `data/chunks/` 目录：

```
data/chunks/
├── 保险基础知多少_chunks.json          # 生产数据（用于embedding）
├── 友邦保险-寿险说明书_chunks.json      # 生产数据（用于embedding）
├── 保险图片_chunks.json                # 生产数据（用于embedding）
└── test_semantic_chunks.json          # 测试数据（可删除）
```

### 文件命名规则

- **生产数据**：`{原文件名}_chunks.json`
- **测试数据**：`test_*.json`（可以删除）

### 文件内容

每个JSON文件包含一个数组，每个元素是一个chunk对象，包含：
- `chunk_id`：唯一标识符
- `text`：原始文本（包含所有句子）
- `embedding_text`：用于embedding的文本（已过滤跳过embedding的句子）
- `metadata`：丰富的元数据信息
- `sentence_infos`：句子级别的信息（是否跳过embedding）

> 📋 **详细说明**：关于data目录下所有文件的详细说明，请查看 [数据目录结构说明文档](DATA_STRUCTURE.md)

## Chunk 数据结构

### Chunk 对象

```python
@dataclass
class Chunk:
    chunk_id: str           # 唯一标识符（UUID）
    text: str              # chunk 文本内容
    metadata: ChunkMetadata # 元数据
```

### ChunkMetadata 对象（完整字段）

```python
@dataclass
class ChunkMetadata:
    # 基础字段
    chunk_id: str
    chunk_type: str              # paragraph, table, list, mixed
    section_path: List[str]      # 标题层级路径
    heading_level: int           # 当前所属标题级别
    char_count: int
    image_refs: List[str]        # 图片引用路径
    source_file: str
    start_line: Optional[int]
    end_line: Optional[int]
    has_table: bool
    has_list: bool
    skip_embedding: bool         # 整个chunk是否跳过embedding
    
    # 语义切割相关字段（新增）
    semantic_type: Optional[str]  # '给付', '免责', '条件', '定义', '其他'
    clause_number: Optional[str] # 条款编号
    is_core_section: bool        # 是否属于核心条款区
    trigger_words: Optional[List[str]]  # 语义触发词列表
    
    # 术语相关字段（新增）
    key_terms: Optional[List[str]]  # 规范术语列表
```

### 字段说明

**基础字段**：
- `chunk_type`：chunk类型（段落、表格、列表等）
- `section_path`：章节路径，如 `["保险责任", "给付条件"]`
- `image_refs`：图片引用路径列表

**语义切割字段**：
- `semantic_type`：语义类型，用于识别chunk的语义含义
- `trigger_words`：触发语义类型的词汇
- `is_core_section`：是否在核心条款区（影响重复话术的判断）

**术语字段**：
- `key_terms`：从chunk文本中提取的规范术语列表

### ChunkMetadata 对象（旧版字段，向后兼容）
class ChunkMetadata:
    chunk_id: str              # chunk ID
    chunk_type: str            # 类型: paragraph, table, list, mixed
    section_path: List[str]    # 章节路径，如 ["科普词01", "健康告知"]
    heading_level: int         # 所属标题级别 (1-6)
    char_count: int            # 字符数
    image_refs: List[str]      # 图片引用路径列表
    source_file: str           # 源文件路径
    start_line: int            # 起始行号（可选）
    end_line: int              # 结束行号（可选）
    has_table: bool            # 是否包含表格
    has_list: bool             # 是否包含列表
```

### JSON 输出格式（完整示例）

```json
{
  "chunk_id": "a254d929-30c7-4b7a-a483-a0cfab2dad76",
  "text": "被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。",
  "embedding_text": "被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。",
  "metadata": {
    "chunk_id": "a254d929-30c7-4b7a-a483-a0cfab2dad76",
    "chunk_type": "paragraph",
    "section_path": ["保险责任"],
    "heading_level": 1,
    "char_count": 30,
    "image_refs": [],
    "source_file": "data/processed/保险基础知多少.md",
    "start_line": 10,
    "end_line": 15,
    "has_table": false,
    "has_list": false,
    "skip_embedding": false,
    "skip_sentences": null,
    "semantic_type": "给付",
    "clause_number": null,
    "is_core_section": true,
    "trigger_words": ["按照合同约定给付"],
    "key_terms": ["保险金", "保险人", "给付", "被保险人", "意外伤害", "身故"]
  },
  "sentence_infos": [
    {
      "text": "被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。",
      "skip_embedding": false,
      "reason": ""
    }
  ]
}
```

**字段说明**：
- `text`：原始文本（包含所有句子，包括跳过embedding的）
- `embedding_text`：用于embedding的文本（已过滤跳过embedding的句子）
- `semantic_type`：语义类型（给付/免责/条件/定义/其他）
- `trigger_words`：触发语义类型的词汇列表
- `key_terms`：提取的规范术语列表
- `is_core_section`：是否在核心条款区
- `sentence_infos`：每个句子的详细信息（是否跳过embedding）

## 统计信息

```python
# 获取 chunks 的统计信息
stats = chunker.get_statistics(chunks)

print(stats)
# 输出:
# {
#     'total_chunks': 10,
#     'avg_chunk_size': 295.8,
#     'min_chunk_size': 243,
#     'max_chunk_size': 368,
#     'chunk_type_distribution': {'paragraph': 6, 'list': 4},
#     'chunks_with_images': 0,
#     'chunks_with_tables': 0,
#     'chunks_with_lists': 4
# }
```

## 高级用法

### 1. 自定义分块策略

```python
# 创建更大的 chunks（适合长文档）
chunker_large = SemanticChunker(
    target_chunk_size=1200,
    max_chunk_size=2000,
    min_chunk_size=300
)

# 创建更小的 chunks（适合精细检索）
chunker_small = SemanticChunker(
    target_chunk_size=500,
    max_chunk_size=800,
    min_chunk_size=100
)
```

### 2. 过滤特定类型的 chunks

```python
# 只获取包含表格的 chunks
table_chunks = [c for c in chunks if c.metadata.has_table]

# 只获取特定章节的 chunks
section_chunks = [
    c for c in chunks 
    if "保险责任" in c.metadata.section_path
]

# 只获取大于特定大小的 chunks
large_chunks = [
    c for c in chunks 
    if c.metadata.char_count > 500
]
```

### 3. 导出为不同格式

```python
import json

# 导出为 JSON
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(
        [chunk.to_dict() for chunk in chunks],
        f,
        ensure_ascii=False,
        indent=2
    )

# 导出为纯文本（用于调试）
with open("chunks.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks, 1):
        f.write(f"=== Chunk {i} ===\n")
        f.write(f"ID: {chunk.chunk_id}\n")
        f.write(f"Section: {' > '.join(chunk.metadata.section_path)}\n")
        f.write(f"Type: {chunk.metadata.chunk_type}\n")
        f.write(f"Size: {chunk.metadata.char_count}\n")
        f.write(f"\n{chunk.text}\n\n")
```

## 分块策略说明

### 1. 标题处理
- 标题不会被包含在 chunk 文本中
- 标题用于更新章节上下文
- 每个 chunk 记录其所属的完整章节路径

### 2. 表格处理
- 表格始终作为独立的 chunk
- 不会与其他内容合并
- 保留完整的 HTML 表格结构

### 3. 列表处理
- 连续的列表项会被合并
- 包括缩进的子内容
- 尽量保持列表的完整性

### 4. 段落处理
- 连续的非空行组成段落
- 段落可以与其他段落合并（在同一章节下）
- 超长段落会按句子拆分

### 5. 合并策略
- 在同一章节下的小块可以合并
- 合并后不超过 `target_chunk_size`
- 如果合并后超过 `max_chunk_size`，则不合并

## 测试

运行测试脚本：

```bash
python scripts/test_chunker.py
```

测试包括：
1. 单个文件分块测试
2. 表格处理测试
3. 批量处理测试
4. 章节层级保留测试

## 性能考虑

- **内存使用**: 每次处理一个文件，内存占用较小
- **处理速度**: 取决于文件大小，通常每秒可处理数千行
- **输出大小**: JSON 输出约为原始文本的 2-3 倍（包含 metadata）

## 常见问题

### Q: 如何调整 chunk 大小？
A: 通过 `target_chunk_size`、`max_chunk_size` 和 `min_chunk_size` 参数调整。

### Q: 表格太大怎么办？
A: 表格作为原子单元不会被拆分。如果表格超过 `max_chunk_size`，它仍会作为一个完整的 chunk。

### Q: 如何处理图片？
A: 图片路径会被提取并记录在 `metadata.image_refs` 中，但图片内容不会包含在 chunk 文本中。

### Q: chunk 之间有重叠吗？
A: 当前版本没有实现重叠功能，但保留了 `overlap_size` 参数供未来扩展。

### Q: 如何保证语义完整性？
A: 通过识别文档结构（标题、段落、列表、表格）并在语义边界处分块，避免在句子中间拆分。

## 后续集成

Chunker 的输出可以直接用于：

1. **Embedding**: 将 chunk 文本转换为向量
2. **向量数据库**: 存储 chunk 及其 metadata
3. **检索**: 基于 section_path 和 chunk_type 进行过滤
4. **Reranking**: 使用 metadata 信息优化排序

示例：

```python
from app.chunker import chunk_file
from app.embedder import get_embeddings  # 假设有这个函数

# 1. 分块
chunks = chunk_file("document.md")

# 2. 生成 embeddings
for chunk in chunks:
    embedding = get_embeddings(chunk.text)
    # 存储到向量数据库
    # vector_db.insert(
    #     id=chunk.chunk_id,
    #     vector=embedding,
    #     metadata=chunk.metadata
    # )
```

## 更新日志

### v1.0.0 (2026-01-08)
- 初始版本
- 支持基本的语义分块
- 表格、列表、段落识别
- 章节层级跟踪
- 图片引用提取
- 结构化输出

## 贡献

如有问题或建议，请提交 issue 或 pull request。
