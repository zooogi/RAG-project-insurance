"""
OCR模块运行脚本
简单易用的PDF批量处理工具
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ocr import PDFProcessor
from loguru import logger


def main():
    """主函数"""
    print("=" * 70)
    print("PDF文档处理工具 - 基于MineRU")
    print("=" * 70)
    print()
    
    # 配置日志
    logger.add(
        "logs/ocr_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    
    # 创建处理器
    print("📂 初始化PDF处理器...")
    processor = PDFProcessor(
        pdf_dir="data/pdf",
        output_dir="data/processed",
        temp_dir="data/mineru_temp",
        source="modelscope"
    )
    
    # 获取PDF文件列表
    pdf_files = processor.get_pdf_files()
    
    if not pdf_files:
        print("❌ 未找到PDF文件，请将PDF文件放入 data/pdf/ 目录")
        return
    
    print(f"📄 找到 {len(pdf_files)} 个PDF文件:")
    for i, pdf_file in enumerate(pdf_files, 1):
        status = "✓ 已处理" if processor.is_processed(pdf_file.stem) else "○ 待处理"
        print(f"   {i}. {pdf_file.name} [{status}]")
    
    print()
    
    # 询问用户
    choice = input("是否开始批量处理？(y/n): ").strip().lower()
    
    if choice != 'y':
        print("已取消处理")
        return
    
    print()
    print("🚀 开始批量处理PDF文件...")
    print("-" * 70)
    
    # 批量处理
    success_count, fail_count = processor.process_all_pdfs(skip_if_exists=True)
    
    print()
    print("-" * 70)
    print("✅ 处理完成！")
    print(f"   成功: {success_count} 个")
    print(f"   失败: {fail_count} 个")
    print()
    
    # 显示处理摘要
    summary = processor.get_processing_summary()
    print("📊 处理摘要:")
    print(f"   总PDF数量: {summary['total_pdfs']}")
    print(f"   已处理数量: {summary['processed_pdfs']}")
    print(f"   Markdown文件: {summary['markdown_files']}")
    print(f"   文本文件: {summary['text_files']}")
    print(f"   JSON元数据: {summary['json_files']}")
    print(f"   图片信息: {summary['image_files']}")
    print(f"   表格信息: {summary['table_files']}")
    print()
    
    print("📁 输出目录: data/processed/")
    print("   ├── markdown/  - Markdown格式文本")
    print("   ├── text/      - 纯文本格式")
    print("   ├── json/      - JSON元数据")
    print("   ├── images/    - 图片信息")
    print("   └── tables/    - 表格信息")
    print()
    print("=" * 70)
    print("💡 提示: 查看 docs/OCR_USAGE.md 了解更多使用方法")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {str(e)}")
        logger.exception("运行出错")
        sys.exit(1)
