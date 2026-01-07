"""
下载 bge-large-zh-v1.5 embedding 模型
"""
import os
import sys

# 必须在导入任何HuggingFace库之前设置环境变量！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.embedder import create_embedder

def main():
    print("=" * 70)
    print("开始下载 bge-large-zh-v1.5 模型")
    print("=" * 70)
    print("\n✓ 已配置使用国内镜像: https://hf-mirror.com")
    print("注意：首次下载模型大小约 1.3GB，请耐心等待...")
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
        
        print("\n" + "=" * 70)
        print("✓ 模型已准备就绪！")
        print("=" * 70)
        print("\n提示：使用 'python scripts/test_embed.py' 来测试模型功能")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
