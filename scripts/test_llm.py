"""
测试 LLM 模块
使用Qwen2.5模型进行RAG答案生成
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.llm import create_llm
from app.chunker import Chunk, ChunkMetadata


def find_local_model(model_name: str):
    """
    查找本地缓存的模型路径
    
    Args:
        model_name: 模型名称（如 "Qwen/Qwen2.5-7B-Instruct"）
    
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
    """创建测试用的chunks数据"""
    chunks = [
        Chunk(
            chunk_id="test_1",
            text="意外伤害保险理赔流程说明。当发生意外伤害事故时，被保险人需要准备以下材料：1. 身份证原件及复印件；2. 医疗诊断证明书；3. 医疗费用发票；4. 事故证明文件；5. 保险合同原件。理赔申请提交后，保险公司会在15个工作日内完成审核并支付理赔款。",
            metadata=ChunkMetadata(
                chunk_id="test_1",
                chunk_type="paragraph",
                section_path=["保险理赔", "意外伤害保险"],
                heading_level=2,
                char_count=150,
                image_refs=[],
                source_file="test.md"
            )
        ),
        Chunk(
            chunk_id="test_2",
            text="重大疾病保险条款详解。本保险涵盖30种重大疾病，包括：恶性肿瘤、急性心肌梗塞、脑中风后遗症、重大器官移植术或造血干细胞移植术、冠状动脉搭桥术、终末期肾病等。保险金额根据投保时约定的保额确定，等待期为90天。",
            metadata=ChunkMetadata(
                chunk_id="test_2",
                chunk_type="paragraph",
                section_path=["保险条款", "重大疾病保险"],
                heading_level=2,
                char_count=120,
                image_refs=[],
                source_file="test.md"
            )
        ),
        Chunk(
            chunk_id="test_3",
            text="车险理赔所需材料清单。发生交通事故后，申请车险理赔需要准备：驾驶证、行驶证、身份证、交通事故责任认定书、车辆维修发票、医疗费用发票（如有人员受伤）、事故现场照片、保险单原件。提交完整材料后，保险公司会在10个工作日内完成理赔审核。",
            metadata=ChunkMetadata(
                chunk_id="test_3",
                chunk_type="paragraph",
                section_path=["保险理赔", "车险"],
                heading_level=2,
                char_count=130,
                image_refs=[],
                source_file="test.md"
            )
        )
    ]
    return chunks


def main():
    print("=" * 70)
    print("测试 LLM 模块")
    print("=" * 70)
    print("\n功能测试：")
    print("  1. Qwen2.5模型加载")
    print("  2. RAG Prompt构建")
    print("  3. 答案生成")
    print("  4. 模型缓存机制")
    print("=" * 70)
    
    try:
        # 1. 加载LLM模型
        print("\n📦 步骤1: 加载LLM模型...")
        print("⚠ 注意: 首次运行需要下载Qwen2.5模型（约6GB），请耐心等待...")
        print("  ✓ 已配置使用国内镜像源加速下载")
        print("  💡 提示: 如果显存不足，可以使用Qwen2.5-1.5B-Instruct（约3GB）")
        
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        model_path = find_local_model(model_name)
        
        if model_path:
            print(f"✓ 找到本地缓存模型: {model_path}")
            llm = create_llm(model_path=model_path, use_mirror=False)
        else:
            print("⚠ 未找到本地模型，将从网络下载...")
            llm = create_llm(model_name=model_name, use_mirror=True)
        
        print("✓ LLM加载成功")
        
        # 显示模型信息
        model_info = llm.get_model_info()
        print("\n模型详细信息:")
        for key, value in model_info.items():
            print(f"  • {key}: {value}")
        
        # 2. 创建测试chunks
        print("\n📦 步骤2: 创建测试chunks数据...")
        chunks = create_test_chunks()
        print(f"✓ 成功创建 {len(chunks)} 个测试chunks")
        
        # 3. 测试查询
        print("\n" + "=" * 70)
        print("🔍 测试RAG答案生成")
        print("=" * 70)
        
        test_queries = [
            "如何申请意外险理赔？需要准备哪些材料？",
            "重疾险包含哪些疾病？",
            "车险理赔需要准备什么材料？"
        ]
        
        for query_idx, query in enumerate(test_queries, 1):
            print(f"\n{'='*70}")
            print(f"查询 {query_idx}: {query}")
            print("=" * 70)
            
            # 生成答案
            result = llm.answer(query, chunks)
            
            print(f"\n✓ 生成的答案:")
            print(f"  {result['answer']}")
            print(f"\n  元信息:")
            print(f"    - 使用的chunks数量: {result['num_chunks_used']}")
            print(f"    - Prompt长度: {result['prompt_length']} 字符")
        
        # 4. 测试模型缓存机制
        print("\n" + "=" * 70)
        print("💾 测试模型缓存机制（节省显存）")
        print("=" * 70)
        
        from app.llm import get_model_cache_info, clear_model_cache
        
        # 显示缓存信息
        cache_info = get_model_cache_info()
        print(f"\n当前模型缓存状态:")
        print(f"  缓存数量: {cache_info['cache_count']}")
        print(f"  缓存的模型: {cache_info['cached_models']}")
        
        # 创建第二个实例，应该复用模型
        print("\n创建第二个LLM实例（应该复用模型，不占用额外显存）...")
        llm2 = create_llm(model_name=model_name, use_mirror=True)
        print("✓ 第二个实例创建成功（如果看到'复用已缓存的LLM模型'，说明缓存生效）")
        
        # 再次查看缓存信息
        cache_info2 = get_model_cache_info()
        print(f"\n缓存状态（创建第二个实例后）:")
        print(f"  缓存数量: {cache_info2['cache_count']}")
        print(f"  说明: 两个实例共享同一个模型，节省显存！")
        
        # 5. 测试不同的生成参数
        print("\n" + "=" * 70)
        print("⚙️ 测试不同的生成参数")
        print("=" * 70)
        
        query = "意外险理赔需要多长时间？"
        print(f"\n查询: {query}")
        
        # 默认参数
        result1 = llm.answer(query, chunks[:1])
        print(f"\n【默认参数】答案: {result1['answer'][:200]}...")
        
        # 调整temperature（更保守）
        result2 = llm.answer(query, chunks[:1], temperature=0.3)
        print(f"\n【Temperature=0.3】答案: {result2['answer'][:200]}...")
        
        print("\n" + "=" * 70)
        print("✓ 所有测试完成！")
        print("=" * 70)
        print("\n提示：")
        print("  • LLM模型已成功加载并测试")
        print("  • 模型缓存机制：多个实例共享模型，节省显存")
        print("  • 可以通过调整temperature和top_p参数控制生成效果")
        print("  • 如需释放显存，可调用: from app.llm import clear_model_cache; clear_model_cache()")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示：")
        print("  • 确保已安装transformers: pip install transformers")
        print("  • Qwen2.5-3B需要约6-8GB显存，如果显存不足：")
        print("    - 使用Qwen2.5-1.5B-Instruct（约3-4GB显存）")
        print("    - 或使用CPU模式（速度较慢）")
        print("  • 如果模型未下载，首次运行会自动下载（约6GB）")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
