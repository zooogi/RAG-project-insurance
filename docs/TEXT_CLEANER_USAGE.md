# 文本清洗功能使用文档

## 概述

文本清洗模块 (`app/text_cleaner.py`) 提供了针对OCR产出的Markdown文本的清洗和降噪功能，主要包括：

1. **基础清洗**：去除页眉页脚、页码、合并OCR断句
2. **句级拆分**：按标点拆分成最小单元
3. **语义降噪**：识别兜底话术和重复话术，标记跳过embedding

## 功能说明

### 1. 基础清洗

#### 1.1 去除页眉页脚
- **逻辑**：识别在文档开头和结尾重复出现的大段文本
- **参数**：`min_repeat_length`（默认20字符）- 页眉页脚最小重复长度
- **策略**：统计每行出现频率，如果某行在开头/结尾出现多次且长度足够，标记为页眉页脚

#### 1.2 去除页码和目录重复
- **识别模式**：
  - `第xx页` / `第 x 页`
  - `第xx页`（中文数字）
  - `Page xx`
  - `页码：xx`
  - 单独的数字行（可能是页码）
- **策略**：匹配页码模式，过滤掉页码行（但保留长度较长的行，可能是正常内容）

#### 1.3 合并OCR断句
- **逻辑**：
  - 前一句没有标点（。！？；）
  - 后一句不是特殊行（标题、列表、表格等）
- **策略**：检测断句情况，将应该合并的行用空格连接

#### 1.4 保留表格和图片
- 表格和图片内容不会被清洗，完整保留

### 2. 句级拆分

按标点符号拆分成最小单元：
- 支持的标点：`。！？；`
- 换行也作为句子分隔符
- 保留空行

### 3. 语义降噪

#### 3.1 规则触发：兜底话术识别

使用关键词和正则表达式识别常见的法律兜底话术：

**内置模式**：
- `本合同未尽事宜`
- `保险人保留最终解释权`
- `本合同的解释权归.*?所有`
- `其他未尽事宜.*?约定`
- `本条款.*?解释权`
- `最终解释权.*?保险人`
- `保险人.*?有权.*?解释`
- `本附加合同.*?未尽事宜`
- `其他.*?以.*?为准`

**自定义模式**：
可以通过修改 `TextCleaner` 的 `boilerplate_patterns` 添加自定义模式。

#### 3.2 统计方法：重复话术识别

- **逻辑**：统计句子在整个文档中的出现频率
- **参数**：`repeat_threshold`（默认3次）- 重复次数阈值
- **策略**：
  - 如果一句话出现次数 > `repeat_threshold`
  - 且不在核心条款区（通过关键词判断）
  - 则标记为低信息句子，跳过embedding

**核心条款区关键词**（默认）：
- `保险责任`、`保险金`、`理赔`、`给付`、`保障范围`
- `保险金额`、`保险费`、`保险期间`、`等待期`

可以通过 `core_section_keywords` 参数自定义。

#### 3.3 跳过embedding策略

- **保留原文**：所有句子（包括跳过embedding的）都保留在原始文本中
- **标记跳过**：通过 `SentenceInfo.skip_embedding` 标记
- **原因记录**：通过 `SentenceInfo.reason` 记录跳过原因（`boilerplate` 或 `repetitive`）
- **embedding文本**：通过 `Chunk.get_embedding_text()` 获取过滤后的文本

## 使用方法

### 基本用法

```python
from app.text_cleaner import TextCleaner
from app.chunker import SemanticChunker
from pathlib import Path

# 创建文本清洗器
cleaner = TextCleaner(
    min_repeat_length=20,      # 页眉页脚最小重复长度
    repeat_threshold=3,         # 重复话术阈值
    core_section_keywords=[...] # 核心条款区关键词（可选）
)

# 基础清洗
text = "..."
cleaned_text = cleaner.basic_clean(text)

# 句级拆分
sentences = cleaner.split_into_sentences(cleaned_text)

# 语义降噪
sentence_infos = cleaner.semantic_denoise(sentences)
```

### 与Chunker集成

```python
from app.chunker import SemanticChunker
from pathlib import Path

# 创建chunker（默认启用文本清洗）
chunker = SemanticChunker(
    target_chunk_size=800,
    max_chunk_size=1500,
    min_chunk_size=200,
    enable_text_cleaning=True  # 启用文本清洗
)

# 处理文件
md_file = Path("data/processed/example.md")
chunks = chunker.chunk_markdown_file(md_file)

# 获取embedding文本（已过滤跳过embedding的句子）
for chunk in chunks:
    embedding_text = chunk.get_embedding_text()
    print(f"原始文本: {chunk.text}")
    print(f"Embedding文本: {embedding_text}")
    
    # 查看句子信息
    if chunk.sentence_infos:
        for i, info in enumerate(chunk.sentence_infos):
            if info.skip_embedding:
                print(f"跳过句子 {i}: {info.text} (原因: {info.reason})")
```

### 自定义文本清洗器

```python
from app.text_cleaner import TextCleaner
from app.chunker import SemanticChunker

# 创建自定义清洗器
custom_cleaner = TextCleaner(
    min_repeat_length=30,
    repeat_threshold=5,
    core_section_keywords=["保险责任", "理赔", "给付"]
)

# 使用自定义清洗器
chunker = SemanticChunker(
    enable_text_cleaning=True,
    text_cleaner=custom_cleaner
)
```

## 输出格式

### Chunk结构

```python
@dataclass
class Chunk:
    chunk_id: str
    text: str  # 原始文本（包含所有句子）
    metadata: ChunkMetadata
    sentence_infos: Optional[List[SentenceInfo]]  # 句子信息列表
```

### SentenceInfo结构

```python
@dataclass
class SentenceInfo:
    text: str              # 句子文本
    skip_embedding: bool   # 是否跳过embedding
    reason: str            # 跳过原因：'boilerplate' 或 'repetitive'
```

### JSON输出格式

```json
{
  "chunk_id": "...",
  "text": "原始文本...",
  "embedding_text": "用于embedding的文本...",
  "metadata": {
    "skip_embedding": false,
    ...
  },
  "sentence_infos": [
    {
      "text": "句子文本",
      "skip_embedding": true,
      "reason": "boilerplate"
    },
    ...
  ]
}
```

## 测试

运行测试脚本：

```bash
python scripts/test_text_cleaner.py
```

测试包括：
1. 基础清洗功能测试
2. 句级拆分测试
3. 兜底话术识别测试
4. 语义降噪测试
5. Chunker集成测试
6. 真实文件测试

## 清洗后文本的保存

### 自动保存功能

当启用文本清洗时，清洗后的文本会自动保存到 `data/cleaned/` 目录，**不会覆盖原始文件**。

- **原始文件位置**：`data/processed/xxx/xxx.md`（保持不变）
- **清洗后文件位置**：`data/cleaned/xxx/xxx.md`（自动创建）
- **目录结构**：保持与 `processed` 目录相同的相对路径结构

### 控制保存行为

```python
# 启用清洗并保存清洗后的文本（默认）
chunker = SemanticChunker(
    enable_text_cleaning=True,
    save_cleaned_text=True  # 默认True
)

# 启用清洗但不保存清洗后的文本
chunker = SemanticChunker(
    enable_text_cleaning=True,
    save_cleaned_text=False
)

# 自定义清洗后文本的输出目录
chunker = SemanticChunker(
    enable_text_cleaning=True,
    save_cleaned_text=True,
    cleaned_output_dir=Path("custom/cleaned/path")
)
```

## 注意事项

1. **表格和图片**：表格和图片内容不会被清洗，完整保留
2. **原文保留**：所有句子都保留在原始文本中，只是标记是否跳过embedding
3. **原始文件不覆盖**：清洗后的文本单独保存，原始OCR文件保持不变
4. **全局统计**：重复话术的识别是在整个文档范围内进行的
5. **核心条款区**：在核心条款区的重复句子不会被标记为低信息
6. **性能考虑**：对于大文档，全局语义降噪可能需要一些时间

## 扩展

### 添加自定义兜底话术模式

```python
from app.text_cleaner import TextCleaner
import re

cleaner = TextCleaner()
cleaner.boilerplate_patterns.append(r"自定义模式")
cleaner.boilerplate_regexes.append(re.compile(r"自定义模式"))
```

### 调整核心条款区关键词

```python
cleaner = TextCleaner(
    core_section_keywords=[
        "保险责任",
        "理赔",
        "给付",
        "自定义关键词"
    ]
)
```

## 示例

参见 `scripts/test_text_cleaner.py` 中的详细示例。
