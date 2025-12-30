"""
下载并测试 bge-large-zh-v1.5 模型
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.embedder import create_embedder

def main():
    print("=" * 70)
    print("开始下载 bge-large-zh-v1.5 模型")
    print("=" * 70)
    print("\n注意：首次下载模型大小约 1.3GB，请耐心等待...")
    print("模型将缓存到: C:\\Users\\47927\\.cache\\huggingface\\hub\\\n")
    
    try:
        # 创建embedder（会自动下载模型）
        embedder = create_embedder(
            model_name="BAAI/bge-large-zh-v1.5",
            use_mirror=True  # 使用国内镜像加速
        )
        
        print("\n" + "=" * 70)
        print("✓ 模型下载并加载成功！")
        print("=" * 70)
        
        # 显示模型信息
        print("\n模型详细信息:")
        info = embedder.get_model_info()
        for key, value in info.items():
            print(f"  • {key}: {value}")
        
        # 简单测试
        print("\n" + "=" * 70)
        print("运行简单测试...")
        print("=" * 70)
        
        test_texts = [
            "保险理赔申请流程",
            "意外险投保说明"
        ]
        
        print(f"\n测试文本: {test_texts}")
        embeddings = embedder.encode(test_texts, show_progress_bar=False)
        print(f"✓ 生成向量成功！")
        print(f"  向量形状: {embeddings.shape}")
        print(f"  向量维度: {embeddings.shape[1]}")
        print(f"  第一个向量前5个值: {embeddings[0][:5]}")
        
        print("\n" + "=" * 70)
        print("✓ 所有测试通过！模型已准备就绪。")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
