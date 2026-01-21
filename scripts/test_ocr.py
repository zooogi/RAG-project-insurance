"""
OCR模块测试脚本 - 增强版，包含图片信息显示和直接OCR测试
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from app.ocr import (
        DocumentProcessor, 
        PDFProcessor, 
        create_processor, 
        process_single_pdf,
        process_single_file,
        clear_paddleocr_cache
    )
except Exception as e:
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


def display_image_info(image_path):
    """显示图片信息和预览"""
    print(f"\n{'='*60}")
    print(f"图片信息: {image_path.name}")
    print(f"{'='*60}")
    
    try:
        # 使用PIL打开图片
        img_pil = Image.open(image_path)
        print(f"PIL格式: {img_pil.format}")
        print(f"PIL尺寸: {img_pil.size}")
        print(f"PIL模式: {img_pil.mode}")
        
        # 使用OpenCV打开图片
        img_cv = cv2.imread(str(image_path))
        if img_cv is not None:
            height, width, channels = img_cv.shape
            print(f"OpenCV尺寸: {width}x{height}, 通道数: {channels}")
            
            # 显示部分像素值（用于调试）
            print(f"\n左上角10x10区域的BGR值:")
            for y in range(min(10, height)):
                row = img_cv[y, :min(10, width)]
                print(f"  行{y}: {row}")
            
            # 检查图片不同区域的像素值
            print(f"\n图片像素值统计:")
            print(f"  左上角(100x100)平均值: {img_cv[:100, :100].mean():.2f}")
            print(f"  中心区域(100x100)平均值: {img_cv[height//2-50:height//2+50, width//2-50:width//2+50].mean():.2f}")
            print(f"  右下角(100x100)平均值: {img_cv[-100:, -100:].mean():.2f}")
            print(f"  全图平均值: {img_cv.mean():.2f}")
            print(f"  全图最小值: {img_cv.min()}")
            print(f"  全图最大值: {img_cv.max()}")
            
            # 检查是否有非白色区域
            non_white_pixels = (img_cv < 250).sum()
            print(f"  非白色像素数量: {non_white_pixels} / {img_cv.size}")
            print(f"  非白色像素比例: {non_white_pixels / img_cv.size * 100:.2f}%")
                
        else:
            print("⚠ 无法用OpenCV读取图片")
            
    except Exception as e:
        print(f"读取图片时出错: {e}")


def test_ocr_directly(image_path):
    """直接测试PaddleOCR"""
    print(f"\n{'='*60}")
    print("直接测试PaddleOCR")
    print(f"{'='*60}")
    
    try:
        # 导入DocumentProcessor以使用解析方法
        from app.ocr import DocumentProcessor
        processor = DocumentProcessor(
            output_base_dir=str(project_root / "data/processed"),
            use_paddleocr_slim=True
        )
        
        # 尝试不同的OCR配置（移除不支持的参数）
        test_configs = [
            {
                "name": "默认配置",
                "params": {
                    "lang": 'ch'
                }
            }
        ]
        
        for config in test_configs:
            print(f"\n测试配置: {config['name']}")
            print(f"参数: {config['params']}")
            
            try:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(**config['params'])
                
                # 执行OCR（不使用cls参数，因为新版本不支持）
                try:
                    result_raw = ocr.ocr(str(image_path))
                except TypeError as e:
                    # 如果ocr方法不支持，尝试使用predict方法
                    if "unexpected keyword argument" in str(e) or "predict" in str(e).lower():
                        result_raw = ocr.predict(str(image_path))
                    else:
                        raise
                
                # 打印原始结果结构用于调试
                print(f"\n原始OCR结果类型: {type(result_raw)}")
                if result_raw:
                    print(f"原始OCR结果长度: {len(result_raw)}")
                    if len(result_raw) > 0:
                        print(f"第一个元素类型: {type(result_raw[0])}")
                        if isinstance(result_raw[0], list):
                            print(f"第一个元素长度: {len(result_raw[0])}")
                            if len(result_raw[0]) > 0:
                                print(f"第一个文本块示例: {result_raw[0][0]}")
                        elif hasattr(result_raw[0], '__dict__'):
                            print(f"对象属性: {list(result_raw[0].__dict__.keys())}")
                
                print(f"OCR结果: {len(result_raw[0]) if result_raw and len(result_raw) > 0 and isinstance(result_raw[0], list) else 0} 个文本块")
                
                # 使用我们的解析方法提取文本
                text_lines = processor._parse_paddleocr_result(result_raw)
                
                print(f"识别到的文本行数: {len(text_lines)}")
                if text_lines:
                    print(f"\n识别到的文本（前10行）:")
                    for i, line in enumerate(text_lines[:10], 1):
                        print(f"  [{i}] {line[:80]}...")
                else:
                    print("⚠ 没有识别到任何文本")
                    # 打印原始结果的前几个元素用于调试
                    if result_raw and len(result_raw) > 0:
                        print(f"\n原始结果前3个元素:")
                        for i, item in enumerate(result_raw[0][:3] if isinstance(result_raw[0], list) else []):
                            print(f"  [{i+1}] {item}")
                    
            except Exception as e:
                print(f"配置 {config['name']} 失败: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"直接测试失败: {e}")
        import traceback
        traceback.print_exc()


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
        # 显示图片信息
        display_image_info(test_image)
        
        processor = create_processor(use_paddleocr_slim=True)
        
        # 明确指定输出目录为 data/processed/保险图片（确保不会输出到preprocessed目录）
        output_dir = project_root / "data" / "processed" / "保险图片"
        
        # 处理图片，明确指定输出目录
        result = processor.process_image(test_image, output_dir=str(output_dir), overwrite=True)
        
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
            print(f"\n识别到的文本（前10行）:")
            for i, line in enumerate(result["text_lines"][:10], 1):
                print(f"  [{i}] {line[:80]}...")
        
        # 显示Markdown文件
        if "files" in result and "markdown" in result["files"]:
            md_file = Path(result["files"]["markdown"])
            if md_file.exists():
                print(f"\n✓ Markdown文件已生成: {md_file}")
                print(f"  输出目录: {result.get('output_dir', 'N/A')}")
                # 验证文件内容
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"  文件大小: {len(content)} 字符")
                    print(f"  文件行数: {len(content.splitlines())} 行")
                    if content.strip():
                        print(f"  前3行预览:")
                        for i, line in enumerate(content.splitlines()[:3], 1):
                            print(f"    {i}. {line[:60]}...")
                    else:
                        print("  ⚠ 警告: 文件内容为空！")
        
        # 直接测试PaddleOCR
        test_ocr_directly(test_image)
        
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
    print("OCR模块测试套件（增强版）")
    print("=" * 60)
    
    tests = [
        ("单个PDF处理", test_single_pdf),
        ("PDFProcessor类方法", test_processor_class),
        ("批量处理", test_batch_process),
        ("图片OCR处理（含图片信息显示和直接OCR测试）", test_image_processing),
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
