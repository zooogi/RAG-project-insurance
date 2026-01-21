
"""
测试文本清洗功能的脚本
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.text_cleaner import TextCleaner
from app.chunker import SemanticChunker
import json


def test_basic_cleaning():
    """测试基础清洗功能"""
    print("=" * 80)
    print("测试1: 基础清洗功能")
    print("=" * 80)
    
    cleaner = TextCleaner()
    
    # 测试文本（包含页码、页眉页脚、OCR断句）
    test_text = """第1页
这是页眉内容 这是页眉内容
# 第一章 保险责任

本保险合同约定的保险责任包括以下内容。被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。被保险人因疾病导致身故的，保险人按照合同约定给付保险金。

被保险人因意外伤害导致伤残的，保险人按照合同约定给付保险金。被保险人因疾病导致伤残的，保险人按照合同约定给付保险金。

第2页
这是页眉内容 这是页眉内容
# 第二章 保险金申请

保险金申请人应向保险人提交以下材料。保险金申请人因特殊原因不能提供以下材料的，应提供其他合法有效的材料。

第3页
这是页眉内容 这是页眉内容
"""
    
    print("原始文本:")
    print(test_text)
    print("\n" + "-" * 80 + "\n")
    
    cleaned = cleaner.basic_clean(test_text)
    
    print("清洗后文本:")
    print(cleaned)
    print("\n" + "-" * 80 + "\n")
    
    return cleaned


def test_sentence_splitting():
    """测试句级拆分"""
    print("=" * 80)
    print("测试2: 句级拆分")
    print("=" * 80)
    
    cleaner = TextCleaner()
    
    test_text = """本保险合同约定的保险责任包括以下内容。被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金！
被保险人因疾病导致身故的，保险人按照合同约定给付保险金？
保险金申请人应向保险人提交以下材料；保险金申请人因特殊原因不能提供以下材料的，应提供其他合法有效的材料。"""
    
    print("原始文本:")
    print(test_text)
    print("\n" + "-" * 80 + "\n")
    
    sentences = cleaner.split_into_sentences(test_text)
    
    print(f"拆分成 {len(sentences)} 个句子:")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    return sentences


def test_boilerplate_detection():
    """测试兜底话术识别"""
    print("\n" + "=" * 80)
    print("测试3: 兜底话术识别")
    print("=" * 80)
    
    cleaner = TextCleaner()
    
    test_sentences = [
        "本合同未尽事宜，按照相关法律法规执行。",
        "保险人保留最终解释权。",
        "本合同的解释权归保险人所有。",
        "被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。",
        "其他未尽事宜，按照双方约定执行。",
    ]
    
    print("测试句子:")
    for i, sentence in enumerate(test_sentences, 1):
        is_boilerplate = cleaner.is_boilerplate_sentence(sentence)
        print(f"{i}. [{is_boilerplate}] {sentence}")
    
    return test_sentences


def test_semantic_denoise():
    """测试语义降噪"""
    print("\n" + "=" * 80)
    print("测试4: 语义降噪")
    print("=" * 80)
    
    cleaner = TextCleaner(repeat_threshold=2)
    
    test_sentences = [
        "本合同未尽事宜，按照相关法律法规执行。",  # 兜底话术
        "被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。",  # 正常内容
        "被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。",  # 重复
        "被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。",  # 重复
        "保险金申请人应向保险人提交以下材料。",  # 正常内容
        "保险人保留最终解释权。",  # 兜底话术
    ]
    
    print("测试句子:")
    for i, sentence in enumerate(test_sentences, 1):
        print(f"{i}. {sentence}")
    
    print("\n语义降噪结果:")
    sentence_infos = cleaner.semantic_denoise(test_sentences)
    
    for i, info in enumerate(sentence_infos, 1):
        status = "❌ 跳过embedding" if info.skip_embedding else "✅ 正常"
        reason = f" ({info.reason})" if info.skip_embedding else ""
        print(f"{i}. [{status}{reason}] {info.text}")
    
    return sentence_infos


def test_chunker_integration():
    """测试chunker集成"""
    print("\n" + "=" * 80)
    print("测试5: Chunker集成测试")
    print("=" * 80)
    
    # 创建一个测试markdown文件
    test_file = project_root / "data/test_cleaner.md"
    
    test_content = """# 保险责任

本保险合同约定的保险责任包括以下内容。被保险人因意外伤害导致身故的，保险人按照合同约定给付保险金。

被保险人因疾病导致身故的，保险人按照合同约定给付保险金。

本合同未尽事宜，按照相关法律法规执行。

保险人保留最终解释权。

# 保险金申请

保险金申请人应向保险人提交以下材料。保险金申请人因特殊原因不能提供以下材料的，应提供其他合法有效的材料。

保险金申请人应向保险人提交以下材料。保险金申请人因特殊原因不能提供以下材料的，应提供其他合法有效的材料。

保险金申请人应向保险人提交以下材料。保险金申请人因特殊原因不能提供以下材料的，应提供其他合法有效的材料。
"""
    
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"创建测试文件: {test_file}\n")
    
    # 使用chunker处理
    chunker = SemanticChunker(
        target_chunk_size=500,
        max_chunk_size=1000,
        min_chunk_size=100,
        enable_text_cleaning=True
    )
    
    chunks = chunker.chunk_markdown_file(test_file)
    
    print(f"生成了 {len(chunks)} 个chunks\n")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"章节: {' > '.join(chunk.metadata.section_path)}")
        print(f"原始文本长度: {len(chunk.text)} 字符")
        print(f"跳过embedding: {chunk.metadata.skip_embedding}")
        
        embedding_text = chunk.get_embedding_text()
        print(f"Embedding文本长度: {len(embedding_text)} 字符")
        
        if chunk.sentence_infos:
            skipped_count = sum(1 for info in chunk.sentence_infos if info.skip_embedding)
            print(f"跳过embedding的句子数: {skipped_count}/{len(chunk.sentence_infos)}")
            
            print("\n句子详情:")
            for j, info in enumerate(chunk.sentence_infos, 1):
                status = "❌" if info.skip_embedding else "✅"
                reason = f" ({info.reason})" if info.skip_embedding else ""
                print(f"  {status} {j}. {info.text[:50]}...{reason}")
        
        print(f"\n原始文本预览:\n{chunk.text[:200]}...")
        print(f"\nEmbedding文本预览:\n{embedding_text[:200]}...")
        print()
    
    # 清理测试文件
    test_file.unlink()
    
    return chunks


def test_real_file():
    """测试真实文件"""
    print("\n" + "=" * 80)
    print("测试6: 真实文件测试")
    print("=" * 80)
    
    test_file = project_root / "data/processed/保险图片/保险图片.md"
    
    if not test_file.exists():
        print(f"❌ 测试文件不存在: {test_file}")
        return None
    
    print(f"处理文件: {test_file.name}\n")
    
    chunker = SemanticChunker(
        target_chunk_size=800,
        max_chunk_size=1500,
        min_chunk_size=200,
        enable_text_cleaning=True
    )
    
    chunks = chunker.chunk_markdown_file(test_file)
    
    print(f"✅ 成功生成 {len(chunks)} 个chunks\n")
    
    # 统计跳过embedding的情况
    total_chunks = len(chunks)
    skipped_chunks = sum(1 for c in chunks if c.metadata.skip_embedding)
    chunks_with_skipped_sentences = sum(
        1 for c in chunks 
        if c.sentence_infos and any(info.skip_embedding for info in c.sentence_infos)
    )
    
    total_sentences = sum(
        len(c.sentence_infos) if c.sentence_infos else 0 
        for c in chunks
    )
    skipped_sentences = sum(
        sum(1 for info in c.sentence_infos if info.skip_embedding)
        for c in chunks if c.sentence_infos
    )
    
    print("📊 统计信息:")
    print(f"  总chunk数: {total_chunks}")
    print(f"  完全跳过embedding的chunk数: {skipped_chunks}")
    print(f"  包含跳过embedding句子的chunk数: {chunks_with_skipped_sentences}")
    print(f"  总句子数: {total_sentences}")
    print(f"  跳过embedding的句子数: {skipped_sentences}")
    
    # 显示前3个chunk的示例
    print("\n📝 前3个chunk示例:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"章节: {' > '.join(chunk.metadata.section_path)}")
        print(f"原始文本长度: {len(chunk.text)} 字符")
        embedding_text = chunk.get_embedding_text()
        print(f"Embedding文本长度: {len(embedding_text)} 字符")
        
        if chunk.sentence_infos:
            skipped = [j for j, info in enumerate(chunk.sentence_infos) if info.skip_embedding]
            if skipped:
                print(f"跳过embedding的句子索引: {skipped}")
    
    return chunks


def main():
    """主测试函数"""
    print("\n🚀 开始测试文本清洗功能\n")
    
    try:
        # 测试1: 基础清洗
        cleaned = test_basic_cleaning()
        
        # 测试2: 句级拆分
        sentences = test_sentence_splitting()
        
        # 测试3: 兜底话术识别
        test_boilerplate_detection()
        
        # 测试4: 语义降噪
        sentence_infos = test_semantic_denoise()
        
        # 测试5: Chunker集成
        chunks = test_chunker_integration()
        
        # 测试6: 真实文件
        real_chunks = test_real_file()
        
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
