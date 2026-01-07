# OCR模块使用指南

## 📖 简介

OCR模块使用MineRU进行高质量的PDF文档解析和文本提取。MineRU是一个强大的文档解析工具，能够准确识别文本、表格、图片等多种元素，并保持文档的结构信息。

## ✨ 主要功能

- ✅ 高质量PDF文档解析（基于MineRU）
- ✅ 自动提取文本、表格、图片
- ✅ 保留文档结构信息（标题层级、列表等）
- ✅ 输出Markdown和JSON格式
- ✅ 支持单个和批量处理
- ✅ 按页码提取文本
- ✅ 文档统计信息

## 🚀 快速开始

### 1. 基本使用

```python
from app.ocr import process_single_pdf

# 处理单个PDF文件
result = process_single_pdf("data/pdf/保险基础知多少.pdf")

# 查看提取的Markdown文本
print(result["markdown"])

# 查看统计信息
print(result["statistics"])
```

### 2. 使用PDFProcessor类

```python
from app.ocr import PDFProcessor

# 创建处理器
processor = PDFProcessor(
    output_base_dir="data/processed",
    source="modelscope"  # 使用国内镜像
)

# 处理PDF
result = processor.process_pdf("data/pdf/保险基础知多少.pdf")

# 提取纯文本
text = processor.extract_text(result)

# 按页提取文本
pages_text = processor.extract_by_page(result)
for page_idx, page_text in pages_text.items():
    print(f"第{page_idx}页: {len(page_text)}字符")

# 保存文本到文件
processor.save_text(result)
```

### 3. 批量处理

```python
from app.ocr import create_processor

# 创建处理器
processor = create_processor()

# 批量处理目录下的所有PDF
results = processor.batch_process("data/pdf")

# 查看处理结果
for result in results:
    print(f"PDF: {result['pdf_name']}")
    print(f"页数: {result['statistics']['total_pages']}")
    print(f"文本长度: {result['statistics']['total_text_length']}")
```

## 📊 返回结果结构

`process_pdf()` 方法返回一个包含以下信息的字典：

```python
{
    "pdf_path": "原始PDF文件的绝对路径",
    "pdf_name": "PDF文件名（不含扩展名）",
    "output_dir": "输出目录路径",
    "mineru_output_dir": "MineRU输出目录路径",
    
    # 提取的内容
    "markdown": "Markdown格式的文本",
    "content_list": [  # 结构化内容列表
        {
            "type": "text",  # 类型: text, list, image, table
            "text": "文本内容",
            "text_level": 1,  # 标题层级（可选）
            "bbox": [x1, y1, x2, y2],  # 边界框坐标
            "page_idx": 0  # 页码索引
        },
        # ... 更多项目
    ],
    "content_list_v2": [...],  # 内容列表v2版本
    "middle_data": {...},  # 中间处理数据
    "model_data": {...},  # 模型数据
    
    # 文件路径
    "files": {
        "markdown": "Markdown文件路径",
        "content_list": "内容列表JSON文件路径",
        "content_list_v2": "内容列表v2 JSON文件路径",
        "middle": "中间数据JSON文件路径",
        "model": "模型数据JSON文件路径",
        "layout_pdf": "布局PDF文件路径",
        "origin_pdf": "原始PDF副本路径"
    },
    
    # 统计信息
    "statistics": {
        "total_items": 60,  # 总项目数
        "text_items": 45,  # 文本项数
        "list_items": 10,  # 列表项数
        "image_items": 3,  # 图片项数
        "table_items": 2,  # 表格项数
        "total_pages": 5,  # 总页数
        "pages": [0, 1, 2, 3, 4],  # 页码列表
        "total_text_length": 5000  # 总文本长度
    }
}
```

## 🔧 API参考

### PDFProcessor类

#### 初始化参数

```python
PDFProcessor(
    output_base_dir="data/processed",  # 输出基础目录
    source="modelscope",  # 模型源: "modelscope" 或 "huggingface"
    use_gpu=True  # 是否使用GPU加速
)
```

#### 主要方法

##### process_pdf()

处理单个PDF文件。

```python
result = processor.process_pdf(
    pdf_path="path/to/file.pdf",  # PDF文件路径
    output_dir=None,  # 输出目录（None则自动生成）
    extract_images=True,  # 是否提取图片
    extract_tables=True  # 是否提取表格
)
```

##### extract_text()

从处理结果中提取纯文本。

```python
text = processor.extract_text(result)
```

##### extract_by_page()

按页码提取文本。

```python
pages_text = processor.extract_by_page(result)
# 返回: {0: "第0页文本", 1: "第1页文本", ...}
```

##### save_text()

保存提取的文本到文件。

```python
text_file = processor.save_text(
    result,
    output_path=None  # 输出路径（None则自动生成）
)
```

##### batch_process()

批量处理PDF文件。

```python
results = processor.batch_process(
    pdf_dir="data/pdf",  # PDF目录
    pattern="*.pdf"  # 文件匹配模式
)
```

### 便捷函数

#### create_processor()

创建PDFProcessor实例。

```python
from app.ocr import create_processor

processor = create_processor(
    output_base_dir="data/processed",
    source="modelscope"
)
```

#### process_single_pdf()

快速处理单个PDF文件。

```python
from app.ocr import process_single_pdf

result = process_single_pdf(
    pdf_path="data/pdf/example.pdf",
    output_dir=None
)
```

## 📝 使用示例

### 示例1: 提取保险文档文本

```python
from app.ocr import create_processor

# 创建处理器
processor = create_processor()

# 处理保险文档
result = processor.process_pdf("data/pdf/保险基础知多少.pdf")

# 提取文本
text = processor.extract_text(result)

# 保存为txt文件
processor.save_text(result, "output/保险基础知识.txt")

print(f"提取完成！文本长度: {len(text)} 字符")
```

### 示例2: 分析文档结构

```python
from app.ocr import process_single_pdf

# 处理PDF
result = process_single_pdf("data/pdf/保险基础知多少.pdf")

# 分析content_list
content_list = result["content_list"]

# 提取所有标题
titles = [
    item["text"] 
    for item in content_list 
    if item.get("type") == "text" and item.get("text_level") == 1
]

print("文档标题:")
for i, title in enumerate(titles, 1):
    print(f"{i}. {title}")
```

### 示例3: 按页处理

```python
from app.ocr import create_processor

processor = create_processor()
result = processor.process_pdf("data/pdf/保险基础知多少.pdf")

# 按页提取文本
pages_text = processor.extract_by_page(result)

# 处理每一页
for page_idx, page_text in pages_text.items():
    print(f"\n{'='*60}")
    print(f"第 {page_idx + 1} 页")
    print(f"{'='*60}")
    print(page_text[:200])  # 显示前200字符
```

### 示例4: 批量处理并统计

```python
from app.ocr import create_processor

processor = create_processor()

# 批量处理
results = processor.batch_process("data/pdf")

# 统计分析
total_pages = 0
total_text_length = 0

for result in results:
    stats = result["statistics"]
    total_pages += stats["total_pages"]
    total_text_length += stats["total_text_length"]
    
    print(f"\n{result['pdf_name']}:")
    print(f"  页数: {stats['total_pages']}")
    print(f"  文本长度: {stats['total_text_length']}")

print(f"\n总计:")
print(f"  文档数: {len(results)}")
print(f"  总页数: {total_pages}")
print(f"  总文本长度: {total_text_length}")
```

## 🎯 MineRU输出结构

MineRU处理后会生成以下文件结构：

```
data/processed/
└── 保险基础知多少/
    └── 保险基础知多少/
        └── hybrid_auto/
            ├── 保险基础知多少.md                    # Markdown格式文本
            ├── 保险基础知多少_content_list.json     # 结构化内容列表
            ├── 保险基础知多少_content_list_v2.json  # 内容列表v2
            ├── 保险基础知多少_middle.json           # 中间处理数据
            ├── 保险基础知多少_model.json            # 模型数据
            ├── 保险基础知多少_layout.pdf            # 布局标注PDF
            └── 保险基础知多少_origin.pdf            # 原始PDF副本
```

## ⚙️ 配置说明

### 模型源选择

- `modelscope`: 使用国内镜像（推荐，速度快）
- `huggingface`: 使用HuggingFace官方源

```python
# 使用国内镜像
processor = PDFProcessor(source="modelscope")

# 使用官方源
processor = PDFProcessor(source="huggingface")
```

### GPU加速

```python
# 启用GPU（默认）
processor = PDFProcessor(use_gpu=True)

# 仅使用CPU
processor = PDFProcessor(use_gpu=False)
```

## 🐛 常见问题

### Q1: MineRU未安装

**错误**: `MineRU未安装，请运行: pip install mineru>=2.7.0`

**解决**: 
```bash
pip install mineru>=2.7.0
```

### Q2: 首次运行很慢

**原因**: MineRU首次运行会自动下载模型（约1-2GB）

**解决**: 耐心等待模型下载完成，后续运行会很快

### Q3: 处理超时

**错误**: `MineRU处理超时（超过10分钟）`

**解决**: 
- 检查PDF文件大小，超大文件可能需要更长时间
- 可以修改`process_pdf()`中的timeout参数

### Q4: 内存不足

**解决**:
- 减少批量处理的文件数量
- 使用GPU加速
- 增加系统内存

## 📚 相关资源

- [MineRU官方文档](https://github.com/opendatalab/MinerU)
- [项目README](../README.md)
- [Embedder使用指南](./EMBEDDER_USAGE.md)

## 💡 最佳实践

1. **首次使用**: 先用小文件测试，确保MineRU正常工作
2. **批量处理**: 建议分批处理，避免一次处理过多文件
3. **结果保存**: 及时保存处理结果，避免重复处理
4. **错误处理**: 使用try-except捕获异常，确保批量处理的稳定性
5. **资源管理**: 处理大量文件时注意磁盘空间

## 🔄 更新日志

### v1.0.0 (2026-01-07)
- ✨ 初始版本
- ✅ 支持MineRU PDF处理
- ✅ 支持文本提取和保存
- ✅ 支持批量处理
- ✅ 支持按页提取
- ✅ 提供详细统计信息
