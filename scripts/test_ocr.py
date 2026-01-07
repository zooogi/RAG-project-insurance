"""
OCR模块测试脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ocr import PDFProcessor, create_processor, process_single_pdf


def test_single_pdf():
    """测试单个PDF处理"""
    print("=" * 60)
    print("测试1: 单个PDF处理")
    print("=" * 60)
    
    # 测试文件
    test_pdf = project_root / "data/pdf/保险基础知多少.pdf"
    
    if not test_pdf.exists():
        print(f"✗ 测试文件不存在: {test_pdf}")
        return False
    
    try:
        # 使用便捷函数处理
        result = process_single_pdf(test_pdf)
        
        print("\n处理结果:")
        print(f"  PDF名称: {result['pdf_name']}")
        print(f"  输出目录: {result['output_dir']}")
        
        # 显示统计信息
        if "statistics" in result:
            print("\n文档统计:")
            stats = result["statistics"]
            print(f"  总页数: {stats['total_pages']}")
            print(f"  总项目数: {stats['total_items']}")
            print(f"  文本项: {stats['text_items']}")
            print(f"  列表项: {stats['list_items']}")
            print(f"  图片项: {stats['image_items']}")
            print(f"  表格项: {stats['table_items']}")
            print(f"  总文本长度: {stats['total_text_length']} 字符")
        
        # 显示文件列表
        if "files" in result:
            print("\n生成的文件:")
            for file_type, file_path in result["files"].items():
                print(f"  {file_type}: {Path(file_path).name}")
        
        # 提取文本预览
        if "markdown" in result:
            text = result["markdown"]
            print(f"\n提取的文本长度: {len(text)} 字符")
            print("\n文本预览（前300字符）:")
            print("-" * 60)
            print(text[:300])
            print("-" * 60)
        
        print("\n✓ 测试1通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_processor_class():
    """测试PDFProcessor类的各种方法"""
    print("\n" + "=" * 60)
    print("测试2: PDFProcessor类方法")
    print("=" * 60)
    
    test_pdf = project_root / "data/pdf/保险基础知多少.pdf"
    
    if not test_pdf.exists():
        print(f"✗ 测试文件不存在: {test_pdf}")
        return False
    
    try:
        # 创建处理器
        processor = create_processor()
        
        # 处理PDF
        result = processor.process_pdf(test_pdf)
        
        # 测试extract_text方法
        print("\n测试 extract_text():")
        text = processor.extract_text(result)
        print(f"  提取文本长度: {len(text)} 字符")
        
        # 测试extract_by_page方法
        print("\n测试 extract_by_page():")
        pages_text = processor.extract_by_page(result)
        print(f"  提取了 {len(pages_text)} 页")
        for page_idx, page_text in list(pages_text.items())[:3]:
            print(f"  第 {page_idx} 页: {len(page_text)} 字符")
        
        # 测试save_text方法
        print("\n测试 save_text():")
        text_file = processor.save_text(result)
        print(f"  文本已保存")
        
        # 验证文件存在
        if text_file.exists():
            print(f"  ✓ 文件存在: {text_file}")
            with open(text_file, 'r', encoding='utf-8') as f:
                saved_text = f.read()
            print(f"  ✓ 文件大小: {len(saved_text)} 字符")
        else:
            print(f"  ✗ 文件不存在: {text_file}")
            return False
        
        print("\n✓ 测试2通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_process():
    """测试批量处理"""
    print("\n" + "=" * 60)
    print("测试3: 批量处理PDF")
    print("=" * 60)
    
    pdf_dir = project_root / "data/pdf"
    
    if not pdf_dir.exists():
        print(f"✗ PDF目录不存在: {pdf_dir}")
        return False
    
    try:
        processor = create_processor()
        
        # 批量处理
        results = processor.batch_process(pdf_dir)
        
        print(f"\n批量处理结果:")
        print(f"  成功处理: {len(results)} 个PDF")
        
        for i, result in enumerate(results, 1):
            print(f"\n  PDF {i}: {result['pdf_name']}")
            if "statistics" in result:
                stats = result["statistics"]
                print(f"    页数: {stats['total_pages']}")
                print(f"    文本长度: {stats['total_text_length']} 字符")
        
        print("\n✓ 测试3通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_content_list_parsing():
    """测试content_list解析"""
    print("\n" + "=" * 60)
    print("测试4: content_list数据解析")
    print("=" * 60)
    
    test_pdf = project_root / "data/pdf/保险基础知多少.pdf"
    
    if not test_pdf.exists():
        print(f"✗ 测试文件不存在: {test_pdf}")
        return False
    
    try:
        processor = create_processor()
        result = processor.process_pdf(test_pdf)
        
        if "content_list" not in result:
            print("✗ 结果中没有content_list")
            return False
        
        content_list = result["content_list"]
        
        print(f"\ncontent_list分析:")
        print(f"  总项目数: {len(content_list)}")
        
        # 分析不同类型的项目
        type_counts = {}
        for item in content_list:
            item_type = item.get("type", "unknown")
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        print("\n  项目类型分布:")
        for item_type, count in type_counts.items():
            print(f"    {item_type}: {count}")
        
        # 显示前几个项目的示例
        print("\n  前3个项目示例:")
        for i, item in enumerate(content_list[:3], 1):
            print(f"\n  项目 {i}:")
            print(f"    类型: {item.get('type')}")
            print(f"    页码: {item.get('page_idx')}")
            if "text" in item:
                text_preview = item["text"][:50] + "..." if len(item["text"]) > 50 else item["text"]
                print(f"    文本: {text_preview}")
            if "text_level" in item:
                print(f"    文本级别: {item['text_level']}")
        
        print("\n✓ 测试4通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("OCR模块测试套件")
    print("=" * 60)
    
    tests = [
        ("单个PDF处理", test_single_pdf),
        ("PDFProcessor类方法", test_processor_class),
        ("批量处理", test_batch_process),
        ("content_list解析", test_content_list_parsing),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n测试 '{test_name}' 发生异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
