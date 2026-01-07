"""
Embedding模块 - 使用bge-large-zh-v1.5模型进行文本向量化
"""
import os
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """文本嵌入器类，用于将文本转换为向量表示"""
    
    def __init__(
        self, 
        model_name: str = "BAAI/bge-large-zh-v1.5",
        model_path: str = None,
        device: str = None,
        use_mirror: bool = True
    ):
        """
        初始化Embedder
        
        Args:
            model_name: 模型名称，默认使用bge-large-zh-v1.5
            model_path: 本地模型路径，如果指定则从本地加载
            device: 设备类型 ('cuda', 'cpu' 或 None自动检测)
            use_mirror: 是否使用国内镜像加速下载
        """
        self.model_name = model_name
        self.model_path = model_path
        
        # 加载模型
        print(f"正在加载模型: {model_path or model_name}")
        try:
            if model_path and os.path.exists(model_path):
                # 从本地路径加载
                self.model = SentenceTransformer(model_path, device=device)
                print(f"✓ 成功从本地加载模型: {model_path}")
            else:
                # 从HuggingFace加载（首次会自动下载）
                self.model = SentenceTransformer(model_name, device=device)
                print(f"✓ 成功加载模型: {model_name}")
                
                # 显示模型缓存位置
                cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                print(f"模型缓存位置: {cache_dir}")
                
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            raise
        
        # 获取向量维度
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"向量维度: {self.embedding_dim}")
    
    def encode(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            show_progress_bar: 是否显示进度条
            normalize_embeddings: 是否归一化向量（推荐开启，用于余弦相似度计算）
        
        Returns:
            numpy数组，形状为 (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码查询文本（为查询添加特殊前缀以提升检索效果）
        
        Args:
            queries: 查询文本或查询文本列表
            **kwargs: 传递给encode的其他参数
        
        Returns:
            查询向量
        """
        if isinstance(queries, str):
            queries = [queries]
        
        # BGE模型建议为查询添加指令前缀
        queries_with_instruction = [f"为这个句子生成表示以用于检索相关文章：{q}" for q in queries]
        
        return self.encode(queries_with_instruction, **kwargs)
    
    def encode_documents(self, documents: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码文档文本
        
        Args:
            documents: 文档文本或文档文本列表
            **kwargs: 传递给encode的其他参数
        
        Returns:
            文档向量
        """
        return self.encode(documents, **kwargs)
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        计算两组向量之间的余弦相似度
        
        Args:
            embeddings1: 第一组向量 (n, dim)
            embeddings2: 第二组向量 (m, dim)
        
        Returns:
            相似度矩阵 (n, m)
        """
        # 如果向量已归一化，点积即为余弦相似度
        return np.dot(embeddings1, embeddings2.T)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "embedding_dim": self.embedding_dim,
            "device": str(self.model.device),
            "max_seq_length": self.model.max_seq_length
        }


# 便捷函数
def create_embedder(
    model_name: str = "BAAI/bge-large-zh-v1.5",
    model_path: str = None,
    **kwargs
) -> Embedder:
    """
    创建Embedder实例的便捷函数
    
    Args:
        model_name: 模型名称
        model_path: 本地模型路径
        **kwargs: 其他参数
    
    Returns:
        Embedder实例
    """
    return Embedder(model_name=model_name, model_path=model_path, **kwargs)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 Embedder 模块")
    print("=" * 60)
    
    # 创建embedder实例（首次运行会下载模型）
    embedder = create_embedder()
    
    # 显示模型信息
    print("\n模型信息:")
    for key, value in embedder.get_model_info().items():
        print(f"  {key}: {value}")
    
    # 测试文本
    print("\n" + "=" * 60)
    print("测试保险相关文本编码")
    print("=" * 60)
    
    documents = [
        "意外伤害保险理赔流程说明",
        "重大疾病保险条款详解",
        "车险理赔所需材料清单",
        "人寿保险投保须知"
    ]
    
    queries = [
        "如何申请意外险理赔？",
        "重疾险包含哪些疾病？"
    ]
    
    # 编码文档
    print("\n编码文档...")
    doc_embeddings = embedder.encode_documents(documents)
    print(f"文档向量形状: {doc_embeddings.shape}")
    
    # 编码查询
    print("\n编码查询...")
    query_embeddings = embedder.encode_queries(queries)
    print(f"查询向量形状: {query_embeddings.shape}")
    
    # 计算相似度
    print("\n计算相似度...")
    similarities = embedder.similarity(query_embeddings, doc_embeddings)
    
    print("\n查询-文档相似度矩阵:")
    print("=" * 60)
    for i, query in enumerate(queries):
        print(f"\n查询 {i+1}: {query}")
        for j, doc in enumerate(documents):
            print(f"  文档 {j+1} (相似度: {similarities[i][j]:.4f}): {doc}")
        
        # 找出最相关的文档
        most_similar_idx = np.argmax(similarities[i])
        print(f"  → 最相关文档: {documents[most_similar_idx]}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
