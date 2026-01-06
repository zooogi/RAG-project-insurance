"""
测试 bge-large-zh-v1.5 embedding 模型
注意：此脚本使用已下载的模型，不会重新下载
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.embedder import create_embedder
import numpy as np

def main():
    print("=" * 70)
    print("测试 bge-large-zh-v1.5 模型")
    print("=" * 70)
    print("\n注意：此脚本使用已缓存的模型，不会重新下载")
    print("如果模型未下载，请先运行: python scripts/download_model.py\n")
    
    try:
        # 创建 embedder（使用本地缓存的模型，避免重复下载）
        print("\n📦 加载模型...")
        
        # 构建本地模型缓存路径
        model_cache_path = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "huggingface",
            "hub",
            "models--BAAI--bge-large-zh-v1.5",
            "snapshots"
        )
        
        # 检查本地缓存是否存在
        if os.path.exists(model_cache_path):
            # 获取最新的snapshot目录
            snapshots = [d for d in os.listdir(model_cache_path) if os.path.isdir(os.path.join(model_cache_path, d))]
            if snapshots:
                # 使用最新的snapshot
                latest_snapshot = sorted(snapshots)[-1]
                local_model_path = os.path.join(model_cache_path, latest_snapshot)
                print(f"✓ 找到本地缓存模型: {local_model_path}")
                
                # 使用本地路径加载，完全避免网络请求
                embedder = create_embedder(model_path=local_model_path)
            else:
                raise FileNotFoundError("模型缓存目录存在但为空")
        else:
            # 如果本地没有缓存，提示用户先下载
            raise FileNotFoundError(
                f"未找到本地模型缓存！\n"
                f"请先运行: python scripts/download_model.py\n"
                f"预期路径: {model_cache_path}"
            )
        
        print("✓ 模型加载成功！（使用本地缓存，无需联网）")
        
        # 显示模型信息
        print("\n模型详细信息:")
        info = embedder.get_model_info()
        for key, value in info.items():
            print(f"  • {key}: {value}")
        
        # 测试文本
        print("\n" + "=" * 70)
        print("测试保险相关文本编码")
        print("=" * 70)
        
        documents = [
            "意外伤害保险理赔流程说明",
            "重大疾病保险条款详解",
            "车险理赔所需材料清单",
            "人寿保险投保须知",
            "医疗保险报销范围介绍"
        ]
        
        queries = [
            "如何申请意外险理赔？",
            "重疾险包含哪些疾病？",
            "车险需要准备什么材料？"
        ]
        
        # 编码文档
        print("\n📄 编码文档...")
        doc_embeddings = embedder.encode_documents(
            documents, 
            show_progress_bar=False
        )
        print(f"✓ 文档向量形状: {doc_embeddings.shape}")
        print(f"  - 文档数量: {len(documents)}")
        print(f"  - 向量维度: {doc_embeddings.shape[1]}")
        
        # 详细展示每个文档的原文和embedding向量
        print("\n" + "=" * 70)
        print("文档详细信息（原文 → Embedding向量）")
        print("=" * 70)
        
        for i, doc in enumerate(documents):
            print(f"\n📄 文档 {i+1}:")
            print(f"  原文: \"{doc}\"")
            print(f"  向量维度: {doc_embeddings.shape[1]}")
            print(f"  向量前20个值: {doc_embeddings[i][:20]}")
            print(f"  向量统计:")
            print(f"    - 最大值: {np.max(doc_embeddings[i]):.6f}")
            print(f"    - 最小值: {np.min(doc_embeddings[i]):.6f}")
            print(f"    - 均值: {np.mean(doc_embeddings[i]):.6f}")
            print(f"    - 标准差: {np.std(doc_embeddings[i]):.6f}")
            print(f"    - L2范数: {np.linalg.norm(doc_embeddings[i]):.6f}")
        
        # 编码查询
        print("\n" + "=" * 70)
        print("🔍 编码查询...")
        print("=" * 70)
        query_embeddings = embedder.encode_queries(
            queries,
            show_progress_bar=False
        )
        print(f"\n✓ 查询向量形状: {query_embeddings.shape}")
        print(f"  - 查询数量: {len(queries)}")
        print(f"  - 向量维度: {query_embeddings.shape[1]}")
        
        # 详细展示每个查询的原文和embedding向量
        print("\n" + "=" * 70)
        print("查询详细信息（原文 → Embedding向量）")
        print("=" * 70)
        
        for i, query in enumerate(queries):
            print(f"\n🔍 查询 {i+1}:")
            print(f"  原文: \"{query}\"")
            print(f"  向量维度: {query_embeddings.shape[1]}")
            print(f"  向量前20个值: {query_embeddings[i][:20]}")
            print(f"  向量统计:")
            print(f"    - 最大值: {np.max(query_embeddings[i]):.6f}")
            print(f"    - 最小值: {np.min(query_embeddings[i]):.6f}")
            print(f"    - 均值: {np.mean(query_embeddings[i]):.6f}")
            print(f"    - 标准差: {np.std(query_embeddings[i]):.6f}")
            print(f"    - L2范数: {np.linalg.norm(query_embeddings[i]):.6f}")
        
        # 计算相似度
        print("\n📊 计算相似度...")
        similarities = embedder.similarity(query_embeddings, doc_embeddings)
        
        print("\n" + "=" * 70)
        print("查询-文档相似度结果")
        print("=" * 70)
        
        for i, query in enumerate(queries):
            print(f"\n🔍 查询 {i+1}: {query}")
            print("-" * 70)
            
            # 获取排序后的索引（从高到低）
            sorted_indices = np.argsort(similarities[i])[::-1]
            
            for rank, j in enumerate(sorted_indices, 1):
                similarity_score = similarities[i][j]
                print(f"  {rank}. [相似度: {similarity_score:.4f}] {documents[j]}")
            
            # 标记最相关的文档
            most_similar_idx = sorted_indices[0]
            print(f"\n  ✓ 最相关文档: {documents[most_similar_idx]}")
            print(f"    相似度分数: {similarities[i][most_similar_idx]:.4f}")
        
        # 额外测试：单个文本编码
        print("\n" + "=" * 70)
        print("测试单个文本编码")
        print("=" * 70)
        
        single_text = "保险理赔需要哪些材料？"
        print(f"\n文本: {single_text}")
        
        single_embedding = embedder.encode(single_text, show_progress_bar=False)
        print(f"✓ 向量形状: {single_embedding.shape}")
        print(f"  向量前5个值: {single_embedding[0][:5]}")
        
        # 验证向量归一化
        norm = np.linalg.norm(single_embedding[0])
        print(f"  向量范数: {norm:.6f} (应接近1.0，表示已归一化)")
        
        print("\n" + "=" * 70)
        print("✓ 所有测试完成！")
        print("=" * 70)
        print("\n提示：")
        print("  • 模型已成功加载并测试")
        print("  • 相似度分数范围: -1 到 1 (越接近1越相似)")
        print("  • 向量已归一化，适合用于余弦相似度计算")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示：如果模型未下载，请先运行: python scripts/download_model.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
