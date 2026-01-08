#!/usr/bin/env python3
# scripts/test_chunker.py

"""
测试chunker功能的脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.chunker import SemanticChunker
import json


def test_single_file():
    """测试单个文件的分块"""
    print("=" * 80)
    print("测试1: 单个文件分块")
    print("=" * 80)
    
    # 测试文件路径
    test_file = project_root / "data/processed/保险基础知多少/保险基础知多少/hybrid_auto/保险基础知多少.md"
    
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    print(f"📄 处理文件: {test_file.name}\n")
    
    # 创建chunker
    chunker = SemanticChunker(
        target_chunk_size=800,
        max_chunk_size=1500,
        min_chunk_size=200
    )
    
    # 执行分块
    chunks = chunker.chunk_markdown_file(test_file)
    
    print(f"✅ 成功生成 {len(chunks)} 个chunks\n")
    
    # 获取统计信息
    stats = chunker.get_statistics(chunks)
    print("📊 统计信息:")
    print(f"  总chunk数: {stats['total_chunks']}")
    print(f"  平均大小: {stats['avg_chunk_size']:.1f} 字符")
    print(f"  最小大小: {stats['min_chunk_size']} 字符")
    print(f"  最大大小: {stats['max_chunk_size']} 字符")
    print(f"  类型分布: {stats['chunk_type_distribution']}")
    print(f"  包含图片的chunk数: {stats['chunks_with_images']}")
    print(f"  包含表格的chunk数: {stats['chunks_with_tables']}")
    print(f"  包含列表的chunk数: {stats['chunks_with_lists']}")
    
    # 显示前3个chunk的示例
    print("\n📝 前3个chunk示例:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"类型: {chunk.metadata.chunk_type}")
        print(f"章节路径: {' > '.join(chunk.metadata.section_path)}")
        print(f"字符数: {chunk.metadata.char_count}")
        print(f"文本预览: {chunk.text[:100]}...")
        if chunk.metadata.image_refs:
            print(f"图片引用: {chunk.metadata.image_refs}")
    
    # 保存结果
    output_file = project_root / "data/chunks/保险基础知多少_chunks.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            [chunk.to_dict() for chunk in chunks],
            f,
            ensure_ascii=False,
            indent=2
        )
    print(f"\n💾 结果已保存到: {output_file}")
    
    return chunks


def test_table_handling():
    """测试表格处理"""
    print("\n" + "=" * 80)
    print("测试2: 表格处理")
    print("=" * 80)
    
    # 测试包含表格的文件
    test_file = project_root / "data/processed/平安-寿险说明书/平安-寿险说明书/hybrid_auto/平安-寿险说明书.md"
    
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    print(f"📄 处理文件: {test_file.name}\n")
    
    chunker = SemanticChunker(
        target_chunk_size=800,
        max_chunk_size=1500,
        min_chunk_size=200
    )
    
    chunks = chunker.chunk_markdown_file(test_file)
    
    # 找出所有包含表格的chunk
    table_chunks = [c for c in chunks if c.metadata.has_table]
    
    print(f"✅ 总共 {len(chunks)} 个chunks")
    print(f"📊 其中包含表格的chunk: {len(table_chunks)} 个\n")
    
    # 显示第一个表格chunk
    if table_chunks:
        print("📋 第一个表格chunk示例:")
        chunk = table_chunks[0]
        print(f"ID: {chunk.chunk_id}")
        print(f"章节路径: {' > '.join(chunk.metadata.section_path)}")
        print(f"字符数: {chunk.metadata.char_count}")
        print(f"文本预览:\n{chunk.text[:300]}...")
    
    return chunks


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 80)
    print("测试3: 批量处理目录")
    print("=" * 80)
    
    input_dir = project_root / "data/processed"
    output_dir = project_root / "data/chunks"
    
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}\n")
    
    chunker = SemanticChunker(
        target_chunk_size=800,
        max_chunk_size=1500,
        min_chunk_size=200
    )
    
    results = chunker.chunk_directory(input_dir, output_dir)
    
    print(f"\n✅ 批量处理完成!")
    print(f"📊 总共处理了 {len(results)} 个文件")
    
    # 统计总chunk数
    total_chunks = sum(len(chunks) for chunks in results.values())
    print(f"📊 总共生成了 {total_chunks} 个chunks")
    
    return results


def test_section_hierarchy():
    """测试章节层级保留"""
    print("\n" + "=" * 80)
    print("测试4: 章节层级保留")
    print("=" * 80)
    
    test_file = project_root / "data/processed/保险基础知多少/保险基础知多少/hybrid_auto/保险基础知多少.md"
    
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    chunker = SemanticChunker()
    chunks = chunker.chunk_markdown_file(test_file)
    
    print("📚 章节层级示例 (前10个chunk):\n")
    for i, chunk in enumerate(chunks[:10], 1):
        section = ' > '.join(chunk.metadata.section_path) if chunk.metadata.section_path else '(无章节)'
        print(f"{i:2d}. [{chunk.metadata.chunk_type:10s}] {section}")
        print(f"    字符数: {chunk.metadata.char_count}, 级别: {chunk.metadata.heading_level}")
    
    return chunks


def main():
    """主测试函数"""
    print("\n🚀 开始测试 SemanticChunker\n")
    
    try:
        # 测试1: 单个文件
        chunks1 = test_single_file()
        
        # 测试2: 表格处理
        chunks2 = test_table_handling()
        
        # 测试3: 批量处理
        results = test_batch_processing()
        
        # 测试4: 章节层级
        chunks4 = test_section_hierarchy()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
