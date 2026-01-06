# OCR模块使用指南

## 📖 简介

OCR模块（`app/ocr.py`）提供了基于MineRU的PDF文档处理功能，能够自动提取PDF中的文本、表格和图片信息，并将结果保存为多种格式。

OCR模块（`app/ocr.py`）结构设计（封装成）：
```
class PDFProcessor:
    - process_single_pdf()    # 处理单个PDF
    - process_all_pdfs()      # 批量处理所有PDF
    - extract_from_mineru()   # 从MineRU输出提取内容
    - save_to_processed()     # 保存到data/processed目录
```

## ✨ 主要功能

- ✅ 批量处理PDF文件
- ✅ 自动提取文本内容
- ✅ 识别并提取表格信息
- ✅ 提取图片元数据
- ✅ 输出多种格式（Markdown、纯文本、JSON）
- ✅ 智能跳过已处理文件
- ✅ 自动清理临时文件
- ✅ 详细的日志记录

## 📁 输出目录结构

处理后的文件保存在 `data/processed/` 目录下：

```
data/processed/
├── markdown/           # Markdown格式文本
│   ├── 保险基础知多少.md
│   ├── 中国互联网保险发展报告2024.md
│   └── ...
├── text/              # 纯文本格式
│   ├── 保险基础知多少.txt
│   └── ...
├── json/              # JSON元数据
│   ├── 保险基础知多少_metadata.json
│   └── ...
├── images/            # 图片信息
│   ├── 保险基础知多少_images.json
│   └── ...
└── tables/            # 表格信息
    ├── 保险基础知多少_tables.json
    └── ...
```


## ⚠️ 重要提示

MineRU处理PDF需要较长时间，特别是大文件：
- 小文件（10-20页）：约 2-5 分钟
- 中等文件（50-100页）：约 10-20 分钟  
- 大文件（100+页）：约 20-60 分钟

## 🚀 推荐使用流程

### 第一步：测试功能（只处理1个小文件）

```bash
# 按 Ctrl+C 中断当前运行的进程（如果有）
# 然后运行测试脚本
python scripts/test_ocr.py
```

**这个脚本会：**
- ✅ 只处理"友邦保险-寿险说明书.pdf"（较小的文件）
- ✅ 验证OCR功能是否正常
- ✅ 大约需要 3-5 分钟

### 第二步：批量处理所有PDF

测试通过后，再批量处理所有文件：

```bash
python scripts/run_ocr.py
```

**这个脚本会：**
- 显示所有4个PDF文件
- 询问是否开始处理
- 逐个处理所有PDF（总共约30-60分钟）

## 📊 你的PDF文件列表

1. **友邦保险-寿险说明书.pdf** ⭐ 推荐先测试这个
2. 保险基础知多少.pdf
3. 平安-寿险说明书.pdf
4. 中国互联网保险发展报告2024.pdf（最大，处理最慢）

## 🔍 检查处理进度

### 方法1：查看日志
```bash
# 查看最新日志
type logs\ocr_*.log
```

### 方法2：查看输出目录
```bash
# 查看已处理的文件
dir data\processed\markdown
```

### 方法3：查看进程
```bash
# 查看Python进程（如果内存占用很高说明正在处理）
tasklist | findstr python
```

## ⚡ 当前情况说明

你之前运行的是 `run_ocr.py`，它正在处理"中国互联网保险发展报告2024.pdf"这个大文件。

**建议操作：**

1. 按 `Ctrl+C` 中断当前进程
2. 运行 `python scripts/test_ocr.py` 先测试小文件
3. 测试成功后，再运行 `python scripts/run_ocr.py` 批量处理

## 📝 处理完成后

处理完成的文件会保存在：
```
data/processed/
├── markdown/          # Markdown格式
├── text/             # 纯文本格式
├── json/             # JSON元数据
├── images/           # 图片信息
└── tables/           # 表格信息
```

## 💡 提示

- 处理过程中可以查看日志了解进度
- 已处理的文件会自动跳过，不会重复处理
- 如果中断了处理，下次运行会从未处理的文件继续
