# 数据目录结构说明

本文档详细说明 `data/` 目录下所有文件和子目录的含义、用途和生成流程。

## 📁 目录结构概览

```
data/
├── raw_data/          # 原始数据（用户提供的PDF/图片等）
├── processed/         # OCR处理后的Markdown文件（保留原始结构）
├── cleaned/           # 文本清洗后的Markdown文件（基础清洗结果）
├── chunks/            # 分块后的JSON文件（最终用于embedding的数据）
└── mineru_test/       # MineRU测试输出（临时文件，可删除）
```

---

## 📂 详细说明

### 1. `raw_data/` - 原始数据目录

**用途**：存放用户提供的原始文件

**内容**：
- PDF文件（如：`保险基础知多少.pdf`）
- 图片文件（如：`保险图片.jpg`）
- CSV文件（如：`insurance - 副本.csv`）

**说明**：
- 这是**输入数据**，不会被程序修改
- 所有处理流程都从这里开始
- 建议保留原始文件作为备份

**示例文件**：
```
raw_data/
├── 保险基础知多少.pdf              # 原始PDF文档
├── 友邦保险-寿险说明书.pdf          # 原始PDF文档
├── 保险图片.jpg                    # 原始图片
└── insurance - 副本.csv            # 原始CSV文件
```

---

### 2. `processed/` - OCR处理后的文件目录

**用途**：存放OCR处理后的Markdown文件和中间文件

**生成流程**：
```
raw_data/*.pdf → OCR处理 → processed/**/*.md
```

**内容**：
- `*.md`：OCR处理后的Markdown文件（**这是主要输出**）
- `images/`：提取的图片文件
- `*_content_list.json`：OCR中间数据
- `*_layout.pdf`：布局分析结果
- `*_model.json`：模型输出
- `*_origin.pdf`：原始PDF副本

**说明**：
- **`*.md`文件**：这是OCR的主要输出，包含文本、表格、图片引用
- **目录结构**：保持与原始文件相同的目录结构
- **其他文件**：OCR的中间文件，可以删除以节省空间

**示例结构**：
```
processed/
└── 保险基础知多少/
    └── 保险基础知多少/
        └── hybrid_auto/
            ├── 保险基础知多少.md              # ⭐ 主要输出：OCR后的Markdown
            ├── 保险基础知多少_content_list.json  # OCR中间文件（可删除）
            ├── 保险基础知多少_layout.pdf        # OCR中间文件（可删除）
            └── ...
```

**重要**：
- `*.md` 文件是**后续处理的基础**
- 这些文件会被 `chunker` 读取进行分块处理
- 其他中间文件可以删除，不影响后续流程

---

### 3. `cleaned/` - 文本清洗后的文件目录

**用途**：存放文本清洗后的Markdown文件

**生成流程**：
```
processed/**/*.md → 文本清洗 → cleaned/**/*.md
```

**清洗内容**：
- ✅ 去除页码（如"第1页"）
- ✅ 去除页眉页脚（重复出现的行）
- ✅ 合并OCR断句（修复OCR导致的错误换行）
- ❌ **不修改**：表格和图片信息

**说明**：
- 这是**中间产物**，用于查看清洗效果
- 清洗后的文本**不会覆盖**原始 `processed/` 文件
- 目录结构保持与 `processed/` 相同

**示例结构**：
```
cleaned/
└── 保险基础知多少/
    └── 保险基础知多少/
        └── hybrid_auto/
            └── 保险基础知多少.md    # 清洗后的Markdown（对比用）
```

**重要**：
- 这些文件是**可选的**，主要用于调试和验证
- 如果不需要查看清洗效果，可以删除此目录
- 不影响后续的chunking和embedding流程

**测试文件**：
- `test_cleaner.md`：测试基础清洗功能的临时文件（测试后会被删除）
- `test_semantic.md`：测试语义切割的临时文件（测试后会被删除）

---

### 4. `chunks/` - 分块后的JSON文件目录 ⭐

**用途**：存放最终用于embedding的chunk数据（JSON格式）

**生成流程**：
```
processed/**/*.md → Chunker处理 → chunks/*_chunks.json
```

**处理内容**：
1. **基础分块**：按段落、表格、列表等结构分块
2. **语义切割**：识别语义类型（给付、免责、条件等）
3. **术语提取**：提取保险业务专业术语
4. **语义降噪**：标记重复话术和兜底话术

**文件格式**：
- 文件名：`{原文件名}_chunks.json`
- 格式：JSON数组，每个元素是一个chunk对象

**Chunk结构**：
```json
{
  "chunk_id": "uuid",
  "text": "原始文本（包含所有句子）",
  "embedding_text": "用于embedding的文本（已过滤跳过embedding的句子）",
  "metadata": {
    "chunk_type": "paragraph/table/list",
    "section_path": ["章节1", "章节2"],
    "semantic_type": "给付/免责/条件/定义/其他",
    "trigger_words": ["触发词1", "触发词2"],
    "key_terms": ["术语1", "术语2"],
    "is_core_section": true/false,
    "clause_number": "第一条",
    ...
  },
  "sentence_infos": [
    {
      "text": "句子文本",
      "skip_embedding": true/false,
      "reason": "boilerplate/repetitive"
    }
  ]
}
```

**示例文件**：
```
chunks/
├── 保险基础知多少_chunks.json              # ⭐ 生产数据
├── 友邦保险-寿险说明书_chunks.json          # ⭐ 生产数据
├── 保险图片_chunks.json                    # ⭐ 生产数据
├── insurance - 副本_chunks.json            # ⭐ 生产数据
└── test_semantic_chunks.json              # 测试数据（可删除）
```

**重要**：
- 这些JSON文件是**最终用于embedding的数据**
- 包含完整的metadata信息（语义类型、术语、章节路径等）
- `embedding_text` 字段是实际用于向量化的文本
- `test_semantic_chunks.json` 是测试产物，可以删除

**使用场景**：
- Embedder模块读取这些文件进行向量化
- Reranker模块使用metadata进行重排序
- LLM模块使用这些chunks生成答案

---

### 5. `mineru_test/` - MineRU测试输出目录

**用途**：MineRU OCR工具的测试输出

**说明**：
- 这是**临时测试目录**，可以删除
- 用于测试MineRU OCR功能
- 正式处理时应该使用 `processed/` 目录

**建议**：
- 测试完成后可以删除此目录
- 不影响项目的正常运行

---

## 🔄 完整数据流程

```
1. 原始数据
   raw_data/*.pdf
   ↓
2. OCR处理
   processed/**/*.md  (主要输出)
   processed/**/images/  (提取的图片)
   ↓
3. 文本清洗（可选）
   cleaned/**/*.md  (清洗后的文本，用于查看效果)
   ↓
4. 分块处理
   chunks/*_chunks.json  (最终用于embedding的数据)
   ↓
5. 向量化
   Embedder读取chunks/*_chunks.json
   ↓
6. 检索和重排序
   Reranker使用chunks和metadata
   ↓
7. 生成答案
   LLM使用检索到的chunks
```

---

## 📋 文件类型总结

| 目录 | 文件类型 | 用途 | 是否必需 | 可删除 |
|------|---------|------|---------|--------|
| `raw_data/` | PDF/图片/CSV | 原始输入数据 | ✅ 必需 | ❌ 不可删除 |
| `processed/` | `*.md` | OCR输出（主要） | ✅ 必需 | ❌ 不可删除 |
| `processed/` | `*_*.json`, `*_layout.pdf` | OCR中间文件 | ⚠️ 可选 | ✅ 可删除 |
| `cleaned/` | `*.md` | 清洗后的文本 | ⚠️ 可选 | ✅ 可删除 |
| `chunks/` | `*_chunks.json` | 最终chunk数据 | ✅ 必需 | ❌ 不可删除 |
| `chunks/` | `test_*.json` | 测试数据 | ⚠️ 可选 | ✅ 可删除 |
| `mineru_test/` | 所有文件 | 测试输出 | ⚠️ 可选 | ✅ 可删除 |

---

## 🧹 清理建议

### 可以安全删除的文件/目录：

1. **OCR中间文件**（不影响功能）：
   ```
   processed/**/*_content_list.json
   processed/**/*_layout.pdf
   processed/**/*_model.json
   processed/**/*_middle.json
   ```

2. **测试文件**：
   ```
   cleaned/test_*.md
   chunks/test_*.json
   mineru_test/  (整个目录)
   ```

3. **清洗后的文件**（如果不需要查看清洗效果）：
   ```
   cleaned/  (整个目录)
   ```

### 必须保留的文件/目录：

1. **原始数据**：
   ```
   raw_data/  (整个目录)
   ```

2. **OCR输出**：
   ```
   processed/**/*.md  (Markdown文件)
   processed/**/images/  (图片文件)
   ```

3. **Chunk数据**：
   ```
   chunks/*_chunks.json  (生产数据，排除test_*.json)
   ```

---

## 💡 使用建议

### 对于开发者：
- 保留所有目录以便调试
- 定期清理测试文件和中间文件
- 使用 `cleaned/` 目录验证清洗效果

### 对于使用者：
- 只需要关注 `chunks/*_chunks.json` 文件
- 可以删除 `cleaned/` 和 `mineru_test/` 目录
- 保留 `raw_data/` 和 `processed/*.md` 作为备份

### 对于生产环境：
- 只保留必需文件：
  - `raw_data/`（备份）
  - `processed/**/*.md`（OCR输出）
  - `chunks/*_chunks.json`（chunk数据，排除test_*.json）
- 删除所有测试文件和中间文件

---

## 🔍 快速查找

### 我想查看OCR结果：
→ 查看 `processed/**/*.md` 文件

### 我想查看清洗效果：
→ 对比 `processed/**/*.md` 和 `cleaned/**/*.md`

### 我想查看最终chunk数据：
→ 查看 `chunks/*_chunks.json` 文件（排除test_*.json）

### 我想重新处理某个文件：
→ 从 `raw_data/` 开始，重新运行OCR → Chunker流程

### 我想清理空间：
→ 删除 `mineru_test/`、`cleaned/`、OCR中间文件、测试文件

---

## ❓ 常见问题

**Q: 为什么有这么多目录？**
A: 每个目录代表处理流程的一个阶段，便于调试和追踪数据变化。

**Q: `cleaned/` 和 `processed/` 有什么区别？**
A: `processed/` 是OCR的原始输出，`cleaned/` 是经过基础清洗后的文本（去除页码、页眉页脚等）。

**Q: 哪些文件是最终用于embedding的？**
A: `chunks/*_chunks.json` 文件中的 `embedding_text` 字段。

**Q: 测试文件可以删除吗？**
A: 可以，`test_*.json` 和 `test_*.md` 都是测试产物，不影响生产流程。

**Q: 如何重新生成chunks？**
A: 删除 `chunks/*_chunks.json` 文件，重新运行 `chunker.chunk_markdown_file()` 或 `chunker.chunk_directory()`。
