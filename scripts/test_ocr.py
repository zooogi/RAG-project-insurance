"""
OCR模块测试脚本
测试PDF处理功能
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ocr import PDFProcessor
from loguru import logger


def test_single_pdf():
    """测试处理单个PDF"""
    logger.info("=" * 60)
    logger.info("测试1: 处理单个PDF文件")
    logger.info("=" * 60)
    
    processor = PDFProcessor()
    
    # 指定测试文件：友邦保险-寿险说明书.pdf
    test_file_name = "友邦保险-寿险说明书.pdf"
    test_pdf = Path("data/pdf") / test_file_name
    
    if not test_pdf.exists():
        logger.error(f"测试文件不存在: {test_file_name}")
        logger.info("将使用第一个可用的PDF文件")
        
        pdf_files = processor.get_pdf_files()
        if not pdf_files:
            logger.error("没有找到PDF文件")
            return False
        test_pdf = pdf_files[0]
    
    logger.info(f"测试文件: {test_pdf.name}")
    
    success = processor.process_single_pdf(test_pdf, skip_if_exists=False)
    
    if success:
        logger.success(f"✓ 单个PDF处理成功: {test_pdf.name}")
        return True
    else:
        logger.error(f"✗ 单个PDF处理失败: {test_pdf.name}")
        return False


def test_batch_processing():
    """测试批量处理PDF（仅用于验证功能，不实际处理）"""
    logger.info("=" * 60)
    logger.info("测试2: 验证批量处理功能（不实际执行）")
    logger.info("=" * 60)
    
    processor = PDFProcessor()
    
    # 只检查功能是否可用，不实际批量处理
    pdf_files = processor.get_pdf_files()
    logger.info(f"发现 {len(pdf_files)} 个PDF文件可供批量处理")
    logger.info("提示: 使用 python scripts/run_ocr.py 进行批量处理")
    
    logger.success("✓ 批量处理功能可用")
    return True


def test_processing_summary():
    """测试获取处理摘要"""
    logger.info("=" * 60)
    logger.info("测试3: 获取处理摘要")
    logger.info("=" * 60)
    
    processor = PDFProcessor()
    summary = processor.get_processing_summary()
    
    logger.info("处理摘要:")
    logger.info(f"  总PDF数量: {summary['total_pdfs']}")
    logger.info(f"  已处理数量: {summary['processed_pdfs']}")
    logger.info(f"  Markdown文件: {summary['markdown_files']}")
    logger.info(f"  文本文件: {summary['text_files']}")
    logger.info(f"  JSON元数据: {summary['json_files']}")
    logger.info(f"  图片信息: {summary['image_files']}")
    logger.info(f"  表格信息: {summary['table_files']}")
    
    if summary['processed_pdfs'] > 0:
        logger.success("✓ 已有处理完成的PDF")
        return True
    else:
        logger.warning("✗ 还没有处理完成的PDF")
        return False


def test_check_processed():
    """测试检查PDF是否已处理"""
    logger.info("=" * 60)
    logger.info("测试4: 检查PDF处理状态")
    logger.info("=" * 60)
    
    processor = PDFProcessor()
    pdf_files = processor.get_pdf_files()
    
    for pdf_file in pdf_files:
        is_processed = processor.is_processed(pdf_file.stem)
        status = "✓ 已处理" if is_processed else "✗ 未处理"
        logger.info(f"  {pdf_file.name}: {status}")
    
    return True


def test_output_structure():
    """测试输出目录结构"""
    logger.info("=" * 60)
    logger.info("测试5: 检查输出目录结构")
    logger.info("=" * 60)
    
    output_dir = Path("data/processed")
    
    subdirs = ["markdown", "text", "json", "images", "tables"]
    all_exist = True
    
    for subdir in subdirs:
        subdir_path = output_dir / subdir
        exists = subdir_path.exists()
        status = "✓ 存在" if exists else "✗ 不存在"
        logger.info(f"  {subdir}/: {status}")
        
        if exists:
            file_count = len(list(subdir_path.glob("*")))
            logger.info(f"    文件数量: {file_count}")
        
        all_exist = all_exist and exists
    
    if all_exist:
        logger.success("✓ 输出目录结构完整")
        return True
    else:
        logger.warning("✗ 输出目录结构不完整")
        return False


def main():
    """运行所有测试"""
    logger.info("\n" + "=" * 60)
    logger.info("OCR模块测试 - 仅测试单个PDF文件")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("输出目录结构", test_output_structure),
        ("检查处理状态", test_check_processed),
        ("处理摘要", test_processing_summary),
        ("单个PDF处理", test_single_pdf),
        ("批量处理功能验证", test_batch_processing),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"测试 '{test_name}' 出错: {str(e)}")
            results.append((test_name, False))
        
        print()  # 空行分隔
    
    # 显示测试结果汇总
    logger.info("=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info("=" * 60)
    logger.info(f"总计: {passed} 通过, {failed} 失败")
    logger.info("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
