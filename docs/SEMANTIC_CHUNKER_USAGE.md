# 语义切割和术语建模功能使用说明

## 概述

本模块实现了基于规则的语义切割和保险业务术语建模功能，用于改进保险条款文档的chunking过程。

## 功能特性

### Step1: 语义切割（规则拆分法）

#### 功能说明
- 识别语义触发词，将文本拆分成语义原子
- 每个语义原子标注明确的语义类型和metadata
- 表格和图片不参与语义切割，但添加metadata

#### 语义类型
- **给付**：识别给付词汇（如"保险人给付"、"赔付"、"承担责任"）
- **免责**：识别免责词汇（如"但"、"除外"、"不承担"、"不予赔付"）
- **条件**：识别条件结构（如"因……导致"、"在……情况下"）
- **定义**：识别定义句式（如"指"、"是指"、"包括"）
- **其他**：不属于以上类型的文本

#### Metadata信息
- `semantic_type`: 语义类型
- `clause_number`: 条款编号（如果存在）
- `is_core_section`: 是否属于核心条款区
- `trigger_words`: 语义触发词列表
- `section_path`: 章节层级路径
- `heading_level`: 标题级别

### Step2: 长度控制

#### 功能说明
- 先进行语义拆分，保证"不混逻辑"
- 如果单个语义原子仍然过长，在同一语义原子内部按结构（编号、标点）/长度做切分
- **不跨语义切分**：确保每个chunk只包含一种语义类型

#### 切分策略
1. 优先按编号切分（如：1. 2. 3. 或 一、二、三、）
2. 其次按标点符号切分（，。；：）
3. 最后按固定长度切分（保留语义类型）

### Step3: 术语建模

#### 功能说明
- 维护保险业务专业术语表（规范术语和变体）
- 遍历所有chunk文本，提取匹配的规范术语
- 将规范术语写入chunk的metadata

#### 术语表结构
```python
{
    "保险金": ["保险金", "保险金额", "保险给付", "保险赔付", ...],
    "保险责任": ["保险责任", "保障责任", "保障范围", ...],
    ...
}
```

#### Metadata信息
- `key_terms`: 规范术语列表

#### 术语增强（未来功能）
- 用户查询时，使用术语表匹配query
- Reranking时给匹配术语的chunk加分
- 回答时显示引用的术语标签和条款位置

## 使用方法

### 基本用法

```python
from app.chunker import SemanticChunker
from pathlib import Path

# 创建chunker（默认启用所有功能）
chunker = SemanticChunker(
    target_chunk_size=800,
    max_chunk_size=1500,
    min_chunk_size=200,
    enable_text_cleaning=True,
    enable_semantic_splitting=True,  # 启用语义切割
    enable_terminology=True  # 启用术语提取
)

# 处理文件
chunks = chunker.chunk_markdown_file(Path("data/processed/example.md"))

# 查看结果
for chunk in chunks:
    print(f"语义类型: {chunk.metadata.semantic_type}")
    print(f"触发词: {chunk.metadata.trigger_words}")
    print(f"术语: {chunk.metadata.key_terms}")
    print(f"文本: {chunk.text[:100]}...")
```

### 自定义语义切割器

```python
from app.semantic_splitter import SemanticSplitter

splitter = SemanticSplitter()
atoms = splitter.split_into_semantic_atoms(text)

for atom in atoms:
    print(f"类型: {atom.semantic_type}")
    print(f"触发词: {atom.trigger_words}")
    print(f"文本: {atom.text}")
```

### 自定义术语表

```python
from app.insurance_terminology import InsuranceTerminology

# 使用默认术语表
terminology = InsuranceTerminology()

# 从文件加载
terminology = InsuranceTerminology(terminology_file=Path("data/terminology.json"))

# 添加新术语
terminology.add_term("新术语", ["变体1", "变体2"])

# 提取术语
terms = terminology.extract_terms("文本内容")
```

## 输出格式

### ChunkMetadata新增字段

```python
@dataclass
class ChunkMetadata:
    # ... 原有字段 ...
    
    # 语义切割相关
    semantic_type: Optional[str] = None  # '给付', '免责', '条件', '定义', '其他'
    clause_number: Optional[str] = None  # 条款编号
    is_core_section: bool = False  # 是否属于核心条款区
    trigger_words: Optional[List[str]] = None  # 语义触发词列表
    
    # 术语相关
    key_terms: Optional[List[str]] = None  # 规范术语列表
```

### JSON输出示例

```json
{
  "chunk_id": "uuid",
  "text": "原始文本",
  "embedding_text": "用于embedding的文本",
  "metadata": {
    "semantic_type": "给付",
    "trigger_words": ["按照合同约定给付"],
    "clause_number": "第一条",
    "is_core_section": true,
    "key_terms": ["保险金", "保险人", "给付"],
    ...
  },
  "sentence_infos": [...]
}
```

## 注意事项

1. **表格处理**：表格不参与语义切割，但会添加术语提取和metadata
2. **图片处理**：图片保留引用，但不放入embedding，也不参与语义切割
3. **语义完整性**：确保每个chunk只包含一种语义类型，不跨语义切分
4. **术语匹配**：使用简单的字符串匹配，对于中文文本效果良好

## 测试

运行测试脚本：

```bash
python scripts/test_semantic_chunker.py
```

## 未来改进

1. **术语增强**：
   - 实现query术语匹配
   - Reranking时给匹配术语的chunk加分
   - 回答时显示术语标签和条款位置

2. **语义类型扩展**：
   - 添加更多语义类型（如"时间"、"金额"等）
   - 支持自定义语义规则

3. **性能优化**：
   - 优化长文本的语义切割性能
   - 缓存术语匹配结果
