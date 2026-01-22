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

from app.chunker import SemanticChunker, SemanticSplitter, InsuranceTerminology
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


def test_semantic_splitting():
    """测试语义切割"""
    print("\n" + "=" * 80)
    print("测试5: 语义切割")
    print("=" * 80)
    
    splitter = SemanticSplitter()
    
    test_text = """被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。
但被保险人因自杀导致身故的，保险人不承担保险责任。
在保险期间内，如果被保险人发生重大疾病，保险人将按照合同约定给付保险金。"""
    
    print("原始文本:")
    print(test_text)
    print("\n语义原子:")
    
    atoms = splitter.split_into_semantic_atoms(test_text)
    for i, atom in enumerate(atoms, 1):
        print(f"\n原子 {i}:")
        print(f"  类型: {atom.semantic_type}")
        print(f"  触发词: {atom.trigger_words}")
        print(f"  文本: {atom.text[:100]}...")


def test_terminology():
    """测试术语提取"""
    print("\n" + "=" * 80)
    print("测试6: 术语提取")
    print("=" * 80)
    
    terminology = InsuranceTerminology()
    
    test_text = """本保险合同约定的保险责任包括以下内容。被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。
保险费应在保险期间内按时缴纳。保险金额为人民币100万元。"""
    
    print("原始文本:")
    print(test_text)
    print("\n提取的术语:")
    
    terms = terminology.extract_terms(test_text)
    for term in sorted(terms):
        print(f"  - {term}")


def test_semantic_chunker_integration():
    """测试语义切割和术语提取的chunker集成"""
    print("\n" + "=" * 80)
    print("测试7: 语义切割和术语提取集成")
    print("=" * 80)
    
    # 创建测试文件
    test_file = project_root / "data/test_semantic.md"
    test_content = """# 保险责任

被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。

但被保险人因自杀导致身故的，保险人不承担保险责任。

在保险期间内，如果被保险人发生重大疾病，保险人将按照合同约定给付保险金。

保险费应在保险期间内按时缴纳。保险金额为人民币100万元。"""
    
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"创建测试文件: {test_file}\n")
    
    # 使用chunker处理（启用语义切割和术语提取）
    chunker = SemanticChunker(
        target_chunk_size=500,
        max_chunk_size=1000,
        min_chunk_size=100,
        enable_text_cleaning=True,
        enable_semantic_splitting=True,
        enable_terminology=True
    )
    
    chunks = chunker.chunk_markdown_file(test_file)
    
    print(f"生成了 {len(chunks)} 个chunks\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(f"语义类型: {chunk.metadata.semantic_type}")
        print(f"触发词: {chunk.metadata.trigger_words}")
        print(f"核心条款区: {chunk.metadata.is_core_section}")
        print(f"条款编号: {chunk.metadata.clause_number}")
        print(f"术语: {chunk.metadata.key_terms}")
        print(f"文本长度: {len(chunk.text)} 字符")
        print(f"文本预览: {chunk.text[:100]}...")
        print()
    
    # 保存结果到data/chunks
    output_file = project_root / "data/chunks/test_semantic_chunks.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            [chunk.to_dict() for chunk in chunks],
            f,
            ensure_ascii=False,
            indent=2
        )
    print(f"💾 结果已保存到: {output_file}")
    
    # 清理测试文件
    test_file.unlink()
    
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
        
        # 测试5: 语义切割
        test_semantic_splitting()
        
        # 测试6: 术语提取
        test_terminology()
        
        # 测试7: 语义切割和术语提取集成
        chunks7 = test_semantic_chunker_integration()
        
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
