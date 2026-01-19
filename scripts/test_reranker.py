"""
测试 Reranker 模块
包含faiss向量检索和bge-reranker-large重排序功能
"""
import os
import sys
import json
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.embedder import create_embedder
from app.reranker import create_reranker
import numpy as np


def find_local_model(model_name: str, model_type: str = "embedder"):
    """
    查找本地缓存的模型路径
    
    Args:
        model_name: 模型名称（如 "BAAI/bge-large-zh-v1.5"）
        model_type: 模型类型 ("embedder" 或 "reranker")
    
    Returns:
        本地模型路径或None
    """
    cache_dir = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub"
    )
    
    # 将模型名称转换为目录名
    model_dir_name = model_name.replace("/", "--")
    model_path = os.path.join(cache_dir, f"models--{model_dir_name}", "snapshots")
    
    if os.path.exists(model_path):
        snapshots = [d for d in os.listdir(model_path) 
                    if os.path.isdir(os.path.join(model_path, d))]
        if snapshots:
            latest_snapshot = sorted(snapshots)[-1]
            return os.path.join(model_path, latest_snapshot)
    
    return None


def load_chunks_from_data_dir(data_dir: str = "data/chunks"):
    """
    从data/chunks目录加载所有chunks文件
    
    Args:
        data_dir: chunks目录路径
        
    Returns:
        List of (file_path, chunks) tuples
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"警告: {data_dir} 目录不存在")
        return []
    
    chunks_files = list(data_path.glob("*_chunks.json"))
    print(f"找到 {len(chunks_files)} 个chunks文件")
    
    all_chunks = []
    for chunks_file in chunks_files:
        print(f"\n加载: {chunks_file.name}")
        try:
            from app.reranker import Reranker
            from app.embedder import create_embedder
            
            # 临时创建reranker来加载chunks
            embedder = create_embedder()
            reranker = create_reranker(embedder)
            chunks = reranker.load_chunks_from_json(chunks_file)
            all_chunks.extend(chunks)
            print(f"  ✓ 加载了 {len(chunks)} 个chunks")
        except Exception as e:
            print(f"  ✗ 加载失败: {e}")
            continue
    
    return all_chunks


def main():
    print("=" * 70)
    print("测试 Reranker 模块")
    print("=" * 70)
    print("\n功能测试：")
    print("  1. Faiss向量检索 (Ranking)")
    print("  2. BGE-Reranker重排序 (Reranking)")
    print("  3. 完整检索流程")
    print("=" * 70)
    
    try:
        # 1. 加载embedder模型
        print("\n📦 步骤1: 加载Embedder模型...")
        embedder_model_path = find_local_model("BAAI/bge-large-zh-v1.5", "embedder")
        if embedder_model_path:
            print(f"✓ 使用本地模型: {embedder_model_path}")
            embedder = create_embedder(model_path=embedder_model_path)
        else:
            print("⚠ 未找到本地模型，将从网络下载...")
            embedder = create_embedder()
        print("✓ Embedder加载成功")
        
        # 2. 加载reranker模型
        print("\n📦 步骤2: 加载Reranker模型...")
        reranker_model_path = find_local_model("BAAI/bge-reranker-large", "reranker")
        if reranker_model_path:
            print(f"✓ 使用本地模型: {reranker_model_path}")
            reranker = create_reranker(
                embedder,
                reranker_model_path=reranker_model_path
            )
        else:
            print("⚠ 未找到本地模型，将从网络下载...")
            print("  提示: 首次下载可能需要较长时间")
            reranker = create_reranker(embedder)
        print("✓ Reranker加载成功")
        
        # 3. 加载chunks数据
        print("\n📦 步骤3: 加载chunks数据...")
        chunks = load_chunks_from_data_dir("data/chunks")
        
        if not chunks:
            print("\n⚠ 未找到chunks数据，使用测试数据...")
            # 使用测试数据
            from app.chunker import Chunk, ChunkMetadata
            test_texts = [
                "意外伤害保险理赔流程说明，需要准备身份证、医疗证明等材料",
                "重大疾病保险条款详解，包含30种重大疾病的保障范围",
                "车险理赔所需材料清单：驾驶证、行驶证、事故证明等",
                "人寿保险投保须知，年龄限制和健康告知要求",
                "医疗保险报销范围介绍，包括门诊和住院费用"
            ]
            chunks = []
            for i, text in enumerate(test_texts):
                metadata = ChunkMetadata(
                    chunk_id=f"test_{i}",
                    chunk_type="paragraph",
                    section_path=["测试章节"],
                    heading_level=1,
                    char_count=len(text),
                    image_refs=[],
                    source_file="test.md"
                )
                chunk = Chunk(chunk_id=f"test_{i}", text=text, metadata=metadata)
                chunks.append(chunk)
        
        print(f"✓ 总共 {len(chunks)} 个chunks")
        
        # 4. 构建faiss索引
        print("\n📦 步骤4: 构建Faiss索引...")
        reranker.build_index(chunks, index_type="flat")
        print("✓ 索引构建完成")
        
        # 显示索引信息
        index_info = reranker.get_index_info()
        print("\n索引信息:")
        for key, value in index_info.items():
            print(f"  • {key}: {value}")
        
        # 5. 测试查询
        print("\n" + "=" * 70)
        print("🔍 测试检索功能")
        print("=" * 70)
        
        test_queries = [
            "如何申请意外险理赔？",
            "重疾险包含哪些疾病？",
            "车险需要准备什么材料？",
            "互联网保险的发展趋势"
        ]
        
        for query_idx, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"查询 {query_idx}: {query}")
            print("=" * 70)
            
            # 执行完整检索流程
            results = reranker.search(
                query,
                rank_top_k=20,
                rerank_top_k=5,
                use_rerank=True
            )
            
            print(f"\n✓ 找到 {len(results)} 个相关结果:\n")
            
            for rank, (chunk, final_score, info) in enumerate(results, 1):
                print(f"【排名 {rank}】分数: {final_score:.4f}")
                print(f"  Ranking分数: {info['rank_score']:.4f}")
                print(f"  Rerank分数: {info['rerank_score']:.4f}")
                
                # 显示chunk信息
                text_preview = chunk.text[:150] + "..." if len(chunk.text) > 150 else chunk.text
                print(f"  文本预览: {text_preview}")
                
                if 'metadata' in info:
                    meta = info['metadata']
                    print(f"  章节路径: {' > '.join(meta.get('section_path', []))}")
                    print(f"  类型: {meta.get('chunk_type', 'unknown')}")
                    if meta.get('has_table'):
                        print(f"  ✓ 包含表格")
                    if meta.get('has_list'):
                        print(f"  ✓ 包含列表")
                print()
        
        # 6. 对比测试：仅ranking vs ranking+reranking
        print("\n" + "=" * 70)
        print("📊 对比测试: 仅Ranking vs Ranking+Reranking")
        print("=" * 70)
        
        query = "互联网保险的发展趋势"
        print(f"\n查询: {query}\n")
        
        # 仅ranking
        print("【仅Ranking结果】")
        rank_only_results = reranker.search(
            query,
            rank_top_k=5,
            rerank_top_k=5,
            use_rerank=False
        )
        for i, (chunk, score, info) in enumerate(rank_only_results, 1):
            print(f"  {i}. [分数: {score:.4f}] {chunk.text[:80]}...")
        
        # ranking + reranking
        print("\n【Ranking + Reranking结果】")
        rerank_results = reranker.search(
            query,
            rank_top_k=20,
            rerank_top_k=5,
            use_rerank=True
        )
        for i, (chunk, score, info) in enumerate(rerank_results, 1):
            print(f"  {i}. [分数: {score:.4f}] {chunk.text[:80]}...")
            print(f"     (Ranking: {info['rank_score']:.4f}, Rerank: {info['rerank_score']:.4f})")
        
        print("\n" + "=" * 70)
        print("✓ 所有测试完成！")
        print("=" * 70)
        print("\n提示：")
        print("  • Faiss索引已构建，可以快速进行向量检索")
        print("  • Reranker模型可以对结果进行精细排序")
        print("  • 结合metadata加权可以提升保险条款检索的准确性")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示：")
        print("  • 确保已安装faiss: pip install faiss-cpu 或 pip install faiss-gpu")
        print("  • 确保已安装transformers: pip install transformers")
        print("  • 如果模型未下载，首次运行会自动下载")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
