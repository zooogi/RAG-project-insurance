# RAG保险项目

基于检索增强生成(RAG)的保险文档智能问答系统。

 
## 📁 项目结构

```
RAG-保险项目/
├── app/                    # 应用代码
│   ├── ocr.py             # PDF处理模块
│   ├── embedder.py        # 文本向量化模块
│   ├── chunker.py         # 文本分块模块
│   ├── config.py          # 配置文件
│   └── main.py            # 主程序
|
├── data/                   # 数据目录
│   ├── pdf/               # 原始PDF文件
│   └── processed/         # 处理后的文件
|
├── docs/                   # 文档
│   ├── OCR_USAGE.md       # OCR模块使用指南
│   ├── CHUNKER_USAGE.md   # Chunker模块使用指南
│   └── EMBEDDER_USAGE.md  # Embedder使用指南
├── scripts/                # 工具脚本
│   ├── test_ocr.py        # OCR测试脚本
│   ├── test_embed.py      # Embedder测试脚本
│   ├── test_chunker.py     #chunker测试脚本
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
- 基于语义的智能分块
- 保留文档结构上下文（标题层级）
- 表格作为不可拆分的原子单元
- 识别并记录图片引用
- 统一的结构化输出格式

### 3. 文本向量化 (Embedder模块)
- 使用bge-large-zh-v1.5中文向量模型
- 支持文档和查询的向量化
- 高效的语义相似度计算

### 4. reranking

### 5. llm

### 6. API


## 🚀 快速开始

### 安装依赖

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd RAG-保险项目

# 2. 创建虚拟环境（推荐）
conda create -n rag python=3.10
conda activate rag

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

#### 使用embedding
