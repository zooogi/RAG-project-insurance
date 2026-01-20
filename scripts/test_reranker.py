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


def create_test_chunks():
    """
    创建测试用的chunks数据
    
    Returns:
        Chunk列表
    """
    from app.chunker import Chunk, ChunkMetadata
    
    test_texts = [
        # 保险理赔相关
        "意外伤害保险理赔流程说明。当发生意外伤害事故时，被保险人需要准备以下材料：1. 身份证原件及复印件；2. 医疗诊断证明书；3. 医疗费用发票；4. 事故证明文件；5. 保险合同原件。理赔申请提交后，保险公司会在15个工作日内完成审核并支付理赔款。",
        
        "重大疾病保险条款详解。本保险涵盖30种重大疾病，包括：恶性肿瘤、急性心肌梗塞、脑中风后遗症、重大器官移植术或造血干细胞移植术、冠状动脉搭桥术、终末期肾病、多个肢体缺失、急性或亚急性重症肝炎、良性脑肿瘤、慢性肝功能衰竭失代偿期、脑炎后遗症或脑膜炎后遗症、深度昏迷、双耳失聪、双目失明、瘫痪、心脏瓣膜手术、严重阿尔茨海默病、严重脑损伤、严重帕金森病、严重Ⅲ度烧伤、严重原发性肺动脉高压、严重运动神经元病、语言能力丧失、重型再生障碍性贫血、主动脉手术等。",
        
        "车险理赔所需材料清单。发生交通事故后，申请车险理赔需要准备：驾驶证、行驶证、身份证、交通事故责任认定书、车辆维修发票、医疗费用发票（如有人员受伤）、事故现场照片、保险单原件。提交完整材料后，保险公司会在10个工作日内完成理赔审核。",
        
        "人寿保险投保须知。投保年龄限制为18-65周岁，需要如实填写健康告知。投保时需要提供：身份证、银行卡、体检报告（根据保额和年龄要求）。保险等待期为90天，等待期内因疾病导致的保险事故不予理赔。",
        
        "医疗保险报销范围介绍。本医疗保险覆盖以下费用：1. 住院医疗费用：床位费、药品费、检查费、手术费等；2. 门诊特殊疾病费用：恶性肿瘤放化疗、肾透析、器官移植后抗排异治疗等；3. 急诊医疗费用：急诊挂号费、急诊检查费、急诊药品费等。年度报销上限为50万元。",
        
        # 互联网保险相关
        "互联网保险的发展趋势。2024年，中国互联网保险保费收入预计将重回两位数增长，占全行业原保费收入的比例有望超过10%。互联网保险通过数字化技术，实现了产品创新、渠道拓展和服务升级，成为推动保险行业高质量发展的重要引擎。",
        
        "互联网保险的优势。互联网保险具有以下优势：1. 便捷性：24小时在线投保，无需线下排队；2. 透明度高：产品条款清晰，价格公开透明；3. 个性化定制：根据用户需求定制保险方案；4. 成本更低：减少中间环节，降低运营成本；5. 服务高效：理赔流程简化，处理速度快。",
        
        "保险行业数字化转型。保险行业正在拥抱数智化转型，运用人工智能、大数据等技术提升服务水平。监管部门鼓励保险公司运用新技术，提升数智化水平，为保险业高质量发展提供新动能。互联网保险作为数智化的典型代表，将引领行业创新发展。",
        
        # 保险基础知识
        "保险的基本原理。保险是一种风险转移机制，通过集合大量同质风险，运用大数法则和概率论原理，实现风险的分散和转移。投保人缴纳保费，保险公司承担保险责任，当发生保险事故时，保险公司按照合同约定进行赔偿或给付。",
        
        "保险合同的要素。保险合同包含以下要素：1. 当事人：投保人、保险人；2. 关系人：被保险人、受益人；3. 保险标的：被保险的财产或人身；4. 保险责任：保险公司承担的风险范围；5. 保险金额：保险公司承担的最高赔偿限额；6. 保险费：投保人需要缴纳的费用；7. 保险期间：保险合同的有效期限。"
    ]
    
    chunks = []
    for i, text in enumerate(test_texts):
        # 根据文本内容确定章节路径
        if "理赔" in text:
            section_path = ["保险理赔"]
        elif "互联网" in text:
            section_path = ["互联网保险"]
        elif "基础" in text or "原理" in text or "合同" in text:
            section_path = ["保险基础知识"]
        else:
            section_path = ["保险条款"]
        
        metadata = ChunkMetadata(
            chunk_id=f"test_chunk_{i}",
            chunk_type="paragraph",
            section_path=section_path,
            heading_level=2,
            char_count=len(text),
            image_refs=[],
            source_file="test_data.md",
            start_line=i * 10,
            end_line=i * 10 + 5,
            has_table=False,
            has_list="清单" in text or "包括" in text or "以下" in text
        )
        chunk = Chunk(chunk_id=f"test_chunk_{i}", text=text, metadata=metadata)
        chunks.append(chunk)
    
    return chunks


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
                reranker_model_path=reranker_model_path,
                use_mirror=False  # 本地模型不需要镜像
            )
        else:
            print("⚠ 未找到本地模型，将从网络下载...")
            print("  提示: 首次下载可能需要较长时间")
            print("  ✓ 已配置使用国内镜像源加速下载")
            reranker = create_reranker(embedder, use_mirror=True)
        print("✓ Reranker加载成功")
        
        # 3. 创建测试chunks数据
        print("\n📦 步骤3: 创建测试chunks数据...")
        chunks = create_test_chunks()
        print(f"✓ 成功创建 {len(chunks)} 个测试chunks")
        
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
        
        # 7. 测试模型缓存机制
        print("\n" + "=" * 70)
        print("💾 测试模型缓存机制（节省显存）")
        print("=" * 70)
        
        from app.reranker import get_model_cache_info, clear_model_cache
        
        # 显示缓存信息
        cache_info = get_model_cache_info()
        print(f"\n当前模型缓存状态:")
        print(f"  缓存数量: {cache_info['cache_count']}")
        print(f"  缓存的模型: {cache_info['cached_models']}")
        
        # 创建第二个实例，应该复用模型
        print("\n创建第二个Reranker实例（应该复用模型，不占用额外显存）...")
        reranker2 = create_reranker(embedder, use_mirror=True)
        print("✓ 第二个实例创建成功（如果看到'复用已缓存的reranker模型'，说明缓存生效）")
        
        # 再次查看缓存信息
        cache_info2 = get_model_cache_info()
        print(f"\n缓存状态（创建第二个实例后）:")
        print(f"  缓存数量: {cache_info2['cache_count']}")
        print(f"  说明: 两个实例共享同一个模型，节省显存！")
        
        print("\n" + "=" * 70)
        print("✓ 所有测试完成！")
        print("=" * 70)
        print("\n提示：")
        print("  • Faiss索引已构建，可以快速进行向量检索")
        print("  • Reranker模型可以对结果进行精细排序")
        print("  • 结合metadata加权可以提升保险条款检索的准确性")
        print("  • 模型缓存机制：多个实例共享模型，节省显存")
        print("  • 如需释放显存，可调用: from app.reranker import clear_model_cache; clear_model_cache()")
        
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
