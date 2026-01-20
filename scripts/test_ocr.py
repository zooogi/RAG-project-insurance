"""
OCR模块测试脚本
"""
import sys
import json
from pathlib import Path

# #region agent log
log_path = Path("/home/chanson/Zhang/RAG-保险项目/.cursor/debug.log")
def _debug_log(location, message, data, hypothesis_id="E"):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
_debug_log("test_ocr.py:11", "测试脚本开始执行", {"python_version": sys.version}, "E")
# #endregion agent log

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# #region agent log
_debug_log("test_ocr.py:18", "开始导入OCR模块", {"project_root": str(project_root)}, "E")
# #endregion agent log

try:
    from app.ocr import (
        DocumentProcessor, 
        PDFProcessor, 
        create_processor, 
        process_single_pdf,
        process_single_file,
        clear_paddleocr_cache
    )
    # #region agent log
    _debug_log("test_ocr.py:28", "OCR模块导入成功", {}, "E")
    # #endregion agent log
except Exception as e:
    # #region agent log
    _debug_log("test_ocr.py:30", "OCR模块导入失败", {"error": str(e), "error_type": type(e).__name__}, "E")
    # #endregion agent log
    raise


def test_single_pdf():
    """测试单个PDF处理"""
    print("=" * 60)
    print("测试1: 单个PDF处理")
    print("=" * 60)
    
    # 测试文件
    test_pdf = project_root / "data/raw_data/保险基础知多少.pdf"
    
    if not test_pdf.exists():
        print(f"✗ 测试文件不存在: {test_pdf}")
        return False
    
    try:
        # 使用便捷函数处理
        result = process_single_pdf(test_pdf)
        
        print("\n处理结果:")
        print(f"  文件名称: {result.get('file_name', result.get('pdf_name', 'unknown'))}")
        print(f"  文件类型: {result.get('file_type', 'pdf')}")
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
    """测试DocumentProcessor类的各种方法（支持PDF、图片、CSV）"""
    print("\n" + "=" * 60)
    print("测试2: DocumentProcessor类方法")
    print("=" * 60)
    
    processor = create_processor(use_paddleocr_slim=True)
    success_count = 0
    
    # 测试PDF文件的方法
    test_pdf = project_root / "data/raw_data/保险基础知多少.pdf"
    if test_pdf.exists():
        try:
            print("\n--- 测试PDF文件处理方法 ---")
            result = processor.process_pdf(test_pdf)
            
            # 测试extract_text方法
            print("\n测试 extract_text():")
            text = processor.extract_text(result)
            print(f"  ✓ 提取文本长度: {len(text)} 字符")
            
            # 测试extract_by_page方法（仅PDF）
            print("\n测试 extract_by_page() (仅PDF):")
            pages_text = processor.extract_by_page(result)
            print(f"  ✓ 提取了 {len(pages_text)} 页")
            for page_idx, page_text in list(pages_text.items())[:3]:
                print(f"    第 {page_idx + 1} 页: {len(page_text)} 字符")
            
            # 测试save_text方法
            print("\n测试 save_text():")
            text_file = processor.save_text(result)
            if text_file.exists():
                print(f"  ✓ 文件已保存: {text_file.name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ PDF处理测试失败: {e}")
    
    # 测试CSV文件的方法
    test_csv = project_root / "data/raw_data/insurance - 副本.csv"
    if test_csv.exists():
        try:
            print("\n--- 测试CSV文件处理方法 ---")
            result = processor.process_csv(test_csv)
            
            # 测试extract_text方法
            print("\n测试 extract_text():")
            text = processor.extract_text(result)
            print(f"  ✓ 提取文本长度: {len(text)} 字符")
            
            # 测试save_text方法
            print("\n测试 save_text():")
            text_file = processor.save_text(result)
            if text_file.exists():
                print(f"  ✓ 文件已保存: {text_file.name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ CSV处理测试失败: {e}")
    
    # 测试图片文件的方法
    test_image = project_root / "data/raw_data/保险图片.jpg"
    if test_image.exists():
        try:
            print("\n--- 测试图片文件处理方法 ---")
            result = processor.process_image(test_image)
            
            # 测试extract_text方法
            print("\n测试 extract_text():")
            text = processor.extract_text(result)
            print(f"  ✓ 提取文本长度: {len(text)} 字符")
            
            # 测试save_text方法
            print("\n测试 save_text():")
            text_file = processor.save_text(result)
            if text_file.exists():
                print(f"  ✓ 文件已保存: {text_file.name}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 图片处理测试失败: {e}")
    
    if success_count > 0:
        print(f"\n✓ 测试2通过（成功测试 {success_count} 种文件类型）")
        return True
    else:
        print("\n✗ 测试2失败：没有可用的测试文件")
        return False


def test_batch_process():
    """测试批量处理所有文件类型（PDF、图片、CSV）"""
    print("\n" + "=" * 60)
    print("测试3: 批量处理所有文件类型")
    print("=" * 60)
    
    input_dir = project_root / "data/raw_data"
    
    if not input_dir.exists():
        print(f"✗ 输入目录不存在: {input_dir}")
        return False
    
    try:
        processor = create_processor(use_paddleocr_slim=True)
        
        # 批量处理所有支持的文件类型
        results = processor.batch_process(input_dir)
        
        print(f"\n批量处理结果:")
        print(f"  成功处理: {len(results)} 个文件")
        
        # 按文件类型统计
        type_counts = {}
        for result in results:
            file_type = result.get('file_type', 'unknown')
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
        
        print(f"\n文件类型分布:")
        for file_type, count in type_counts.items():
            print(f"  {file_type}: {count} 个")
        
        # 显示每个文件的详细信息
        for i, result in enumerate(results, 1):
            file_name = result.get('file_name', result.get('pdf_name', 'unknown'))
            file_type = result.get('file_type', 'unknown')
            print(f"\n  文件 {i}: {file_name} ({file_type})")
            if "statistics" in result:
                stats = result["statistics"]
                if file_type == "pdf":
                    print(f"    页数: {stats.get('total_pages', 'N/A')}")
                elif file_type == "image":
                    print(f"    识别行数: {stats.get('total_lines', 'N/A')}")
                elif file_type == "csv":
                    print(f"    行数: {stats.get('total_rows', 'N/A')}, 列数: {stats.get('total_columns', 'N/A')}")
                print(f"    文本长度: {stats.get('total_text_length', 'N/A')} 字符")
        
        print("\n✓ 测试3通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_processing():
    """测试图片处理（JPG/PNG等）"""
    print("\n" + "=" * 60)
    print("测试4: 图片OCR处理")
    print("=" * 60)
    
    # 测试图片文件
    test_image = project_root / "data/raw_data/保险图片.jpg"
    
    if not test_image.exists():
        print(f"⚠ 测试图片不存在: {test_image}")
        print("  跳过图片OCR测试")
        return True  # 不视为失败
    
    try:
        processor = create_processor(use_paddleocr_slim=True)
        
        # 处理图片
        result = processor.process_image(test_image)
        
        print("\n处理结果:")
        print(f"  文件名称: {result['file_name']}")
        print(f"  文件类型: {result['file_type']}")
        
        # 显示统计信息
        if "statistics" in result:
            stats = result["statistics"]
            print(f"\nOCR统计:")
            print(f"  识别行数: {stats['total_lines']}")
            print(f"  文本长度: {stats['total_text_length']} 字符")
        
        # 显示识别的文本预览
        if "text_lines" in result:
            print(f"\n识别到的文本（前5行）:")
            for i, line in enumerate(result["text_lines"][:5], 1):
                print(f"  {i}. {line[:50]}...")
        
        # 显示Markdown文件
        if "files" in result and "markdown" in result["files"]:
            md_file = Path(result["files"]["markdown"])
            if md_file.exists():
                print(f"\n✓ Markdown文件已生成: {md_file}")
        
        print("\n✓ 测试4通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_csv_processing():
    """测试CSV文件处理"""
    print("\n" + "=" * 60)
    print("测试5: CSV文件处理")
    print("=" * 60)
    
    # 测试CSV文件
    test_csv = project_root / "data/raw_data/insurance - 副本.csv"
    
    if not test_csv.exists():
        print(f"⚠ 测试CSV不存在: {test_csv}")
        print("  跳过CSV处理测试")
        return True  # 不视为失败
    
    try:
        processor = create_processor()
        
        # 处理CSV
        result = processor.process_csv(test_csv)
        
        print("\n处理结果:")
        print(f"  文件名称: {result['file_name']}")
        print(f"  文件类型: {result['file_type']}")
        
        # 显示统计信息
        if "statistics" in result:
            stats = result["statistics"]
            print(f"\nCSV统计:")
            print(f"  总行数: {stats['total_rows']}")
            print(f"  总列数: {stats['total_columns']}")
            print(f"  列名: {', '.join(stats['column_names'][:5])}...")
            print(f"  文本长度: {stats['total_text_length']} 字符")
        
        # 显示Markdown文件
        if "files" in result and "markdown" in result["files"]:
            md_file = Path(result["files"]["markdown"])
            if md_file.exists():
                print(f"\n✓ Markdown文件已生成: {md_file}")
                # 显示Markdown预览
                with open(md_file, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                print(f"\nMarkdown预览（前300字符）:")
                print("-" * 60)
                print(md_content[:300])
                print("-" * 60)
        
        print("\n✓ 测试5通过")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试5失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_file_type_detection():
    """测试自动文件类型识别"""
    print("\n" + "=" * 60)
    print("测试6: 自动文件类型识别")
    print("=" * 60)
    
    processor = create_processor(use_paddleocr_slim=True)
    
    test_files = [
        ("PDF", project_root / "data/raw_data/保险基础知多少.pdf"),
        ("图片", project_root / "data/raw_data/保险图片.jpg"),
        ("CSV", project_root / "data/raw_data/insurance - 副本.csv"),
    ]
    
    success_count = 0
    for file_type, file_path in test_files:
        if not file_path.exists():
            print(f"⚠ {file_type}文件不存在: {file_path.name}")
            continue
        
        try:
            print(f"\n处理 {file_type} 文件: {file_path.name}")
            result = processor.process_file(file_path)
            detected_type = result.get('file_type', 'unknown')
            print(f"  ✓ 自动识别类型: {detected_type}")
            success_count += 1
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
    
    if success_count > 0:
        print(f"\n✓ 测试6通过（成功处理 {success_count} 个文件）")
        return True
    else:
        print("\n⚠ 测试6: 没有可用的测试文件")
        return True  # 不视为失败


def test_content_list_parsing():
    """测试content_list解析"""
    print("\n" + "=" * 60)
    print("测试7: content_list数据解析")
    print("=" * 60)
    
    test_pdf = project_root / "data/raw_data/保险基础知多少.pdf"
    
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
        ("图片OCR处理", test_image_processing),
        ("CSV文件处理", test_csv_processing),
        ("自动文件类型识别", test_auto_file_type_detection),
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
