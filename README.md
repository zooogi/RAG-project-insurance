# RAG保险项目

基于检索增强生成(RAG)的保险文档智能问答系统。

 
## 📁 项目结构

```
RAG-保险项目/
├── app/                    # 应用代码
│   ├── ocr.py             # PDF处理模块
│   ├── embedder.py        # 文本向量化模块
│   ├── chunker.py         # 文本分块模块
│   ├── reranker.py        # 检索和重排序模块
│   ├── config.py          # 配置文件
│   └── main.py            # 主程序
|
├── data/                   # 数据目录（详见 docs/DATA_STRUCTURE.md）
│   ├── raw_data/          # 原始数据（PDF/图片等）
│   ├── processed/         # OCR处理后的Markdown文件
│   ├── cleaned/           # 文本清洗后的Markdown文件（可选）
│   └── chunks/            # 分块后的JSON文件（最终用于embedding）
|
├── docs/                   # 文档
│   ├── DATA_STRUCTURE.md  # 📋 数据目录结构说明（重要！）
│   ├── OCR_USAGE.md       # OCR模块使用指南
│   ├── CHUNKER_USAGE.md   # Chunker模块使用指南
│   ├── TEXT_CLEANER_USAGE.md  # 文本清洗模块使用指南
│   ├── SEMANTIC_CHUNKER_USAGE.md  # 语义切割和术语提取使用指南
│   ├── EMBEDDER_USAGE.md  # Embedder使用指南
│   └── RERANKER_USAGE.md  # Reranker使用指南
├── scripts/                # 工具脚本
│   ├── test_ocr.py        # OCR测试脚本
│   ├── test_embed.py      # Embedder测试脚本
│   ├── test_chunker.py    # Chunker测试脚本
│   ├── test_reranker.py   # Reranker测试脚本
│   ├── test_llm.py        # LLM测试脚本
│   └── download_embed.py  # 下载向量模型
├── requirements.txt        # Python依赖
└── README.md              # 本文件
```

## ✨ 主要功能

### 1. PDF文档处理 (OCR模块)
- 支持文字版和扫描版PDF
- 自动提取文本、表格和图片
- 双引擎支持：MineRU（高级）+ PyMuPDF（备选）
- 输出Markdown和纯文本格式

### 2. 文本分块 (Chunker模块)
- **基础分块**：基于语义的智能分块，保留文档结构上下文（标题层级）
- **语义切割**：识别语义类型（给付、免责、条件、定义等），按语义原子拆分
- **术语提取**：提取保险业务专业术语，写入chunk metadata
- **语义降噪**：识别并跳过重复话术和兜底话术（不影响原始文本）
- **表格处理**：表格作为不可拆分的原子单元
- **图片处理**：识别并记录图片引用，不参与embedding
- **统一输出**：结构化的JSON格式，包含丰富的metadata信息

### 3. 文本向量化 (Embedder模块)
- 使用bge-large-zh-v1.5中文向量模型
- 支持文档和查询的向量化
- 高效的语义相似度计算

### 4. 检索和重排序 (Reranker模块)
- **Ranking阶段**：使用Faiss进行高效的向量检索
- **Reranking阶段**：使用bge-reranker-large进行精细重排序
- 支持metadata加权（利用章节路径、chunk类型等信息）
- 两阶段检索流程，兼顾效率和准确性

### 5. LLM生成 (LLM模块)
- 使用Qwen2.5-3B-Instruct模型进行答案生成（默认，约6-8GB显存）
- 支持更小模型：Qwen2.5-1.5B-Instruct（约3-4GB显存）
- 自动构建RAG prompt（查询 + 检索到的chunks）
- 支持模型缓存机制，节省显存
- 可调节生成参数（temperature、top_p等）

### 6. API接口


## 🚀 快速开始

### 安装依赖

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd RAG-保险项目

# 2. 创建虚拟环境（推荐）
conda create -n rag python=3.10
conda activate rag

# 在终端中运行（设置国内镜像环境）（做好优先设置好环境！！！）
export HF_ENDPOINT=https://hf-mirror.com

# 3. 安装依赖包
pip install -r requirements.txt

# 使用国内镜像加速（可选）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 首次使用配置

#### 使用MineRU（推荐，功能更强大）

```bash
# 安装MineRU和依赖
# 下载依赖包已经包含了安装MineRU
pip install -r requirements.txt
#pip install mineru>=2.7.0 accelerate>=0.20.0

#check 有内容就是下载成功
mineru --help

# test 首次运行会自动下载模型（约1-2GB，需要时间）
mineru -p data/pdf/保险基础知多少.pdf -o data/mineru_test --source modelscope
```

#### 使用Embedding和Reranker

```bash
# 测试Embedder模块
python scripts/test_embed.py

# 测试Reranker模块（首次运行会自动下载模型）
python scripts/test_reranker.py

# 测试LLM模块（首次运行会自动下载模型，约6GB）
python scripts/test_llm.py
```

**注意**：
- Reranker使用`BAAI/bge-reranker-large`模型，首次运行会自动下载（约1-2GB）
- LLM使用`Qwen/Qwen2.5-3B-Instruct`模型，首次运行会自动下载（约6GB）
- **默认使用国内镜像源（hf-mirror.com）加速下载**
- 如果使用GPU，建议安装`faiss-gpu`以提升检索速度
- **显存需求**：
  - Qwen2.5-3B：约6-8GB显存（默认）
  - Qwen2.5-1.5B：约3-4GB显存（显存不足时使用）
  - 如果显存不足，可以使用CPU模式或更小的模型
- 模型会自动缓存到`~/.cache/huggingface/hub/`目录

## 📋 数据目录说明

**重要**：请先阅读 [数据目录结构说明文档](docs/DATA_STRUCTURE.md)，了解 `data/` 目录下所有文件的含义和用途。

### 📂 目录快速参考

| 目录 | 内容 | 用途 | 是否必需 |
|------|------|------|---------|
| `raw_data/` | PDF/图片/CSV | 原始输入数据（不会被修改） | ✅ 必需 |
| `processed/` | `*.md` + 中间文件 | OCR处理后的Markdown（主要输出） | ✅ 必需 |
| `cleaned/` | `*.md` | 文本清洗后的Markdown（查看效果用） | ⚠️ 可选 |
| `chunks/` | `*_chunks.json` | 最终用于embedding的chunk数据 | ✅ 必需 |
| `mineru_test/` | 测试文件 | MineRU测试输出 | ❌ 可删除 |

### 🔄 数据流程

```
raw_data/*.pdf 
  ↓ [OCR处理]
processed/**/*.md (主要输出)
  ↓ [文本清洗，可选]
cleaned/**/*.md (查看清洗效果)
  ↓ [Chunker分块]
chunks/*_chunks.json ⭐ (最终用于embedding)
```

### 📝 重要文件说明

- **`processed/**/*.md`**：OCR的主要输出，包含文本、表格、图片引用
- **`chunks/*_chunks.json`**：最终用于embedding的数据，包含：
  - `text`：原始文本
  - `embedding_text`：用于embedding的文本（已过滤跳过embedding的句子）
  - `metadata`：丰富的元数据（语义类型、术语、章节路径等）
  - `sentence_infos`：句子级别的信息

### 🧹 清理建议

**可以删除**：
- `cleaned/` 目录（如果不需要查看清洗效果）
- `mineru_test/` 目录（测试文件）
- `chunks/test_*.json`（测试数据）
- `processed/**/*_*.json`、`*_layout.pdf` 等OCR中间文件

**必须保留**：
- `raw_data/`（原始数据备份）
- `processed/**/*.md`（OCR输出）
- `chunks/*_chunks.json`（生产数据，排除test_*.json）

详细说明请查看：[docs/DATA_STRUCTURE.md](docs/DATA_STRUCTURE.md)
