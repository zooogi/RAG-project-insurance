"""
Reranker模块 - 使用faiss进行向量检索，使用bge-reranker-large进行重排序
"""
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from app.embedder import Embedder
from app.chunker import Chunk, ChunkMetadata


class Reranker:
    """
    检索和重排序器
    
    功能：
    1. 使用faiss进行高效的向量检索（ranking）
    2. 使用bge-reranker-large进行精细重排序（reranking）
    3. 支持从chunks JSON文件加载数据
    """
    
    def __init__(
        self,
        embedder: Embedder,
        reranker_model_name: str = "BAAI/bge-reranker-large",
        reranker_model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_metadata: bool = True
    ):
        """
        初始化Reranker
        
        Args:
            embedder: Embedder实例，用于向量化
            reranker_model_name: reranker模型名称
            reranker_model_path: 本地reranker模型路径
            device: 设备类型 ('cuda', 'cpu' 或 None自动检测)
            use_metadata: 是否使用metadata进行加权
        """
        self.embedder = embedder
        self.use_metadata = use_metadata
        
        # 自动检测设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 加载reranker模型
        print(f"正在加载reranker模型: {reranker_model_path or reranker_model_name}")
        try:
            if reranker_model_path and os.path.exists(reranker_model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    reranker_model_path
                ).to(device)
                print(f"✓ 成功从本地加载reranker模型: {reranker_model_path}")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
                self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                    reranker_model_name
                ).to(device)
                print(f"✓ 成功加载reranker模型: {reranker_model_name}")
        except Exception as e:
            print(f"✗ Reranker模型加载失败: {e}")
            raise
        
        self.reranker_model.eval()
        
        # 初始化faiss索引（稍后构建）
        self.faiss_index = None
        self.chunks: List[Chunk] = []
        self.chunk_embeddings: Optional[np.ndarray] = None
        
    def load_chunks_from_json(self, json_path: Union[str, Path]) -> List[Chunk]:
        """
        从JSON文件加载chunks
        
        Args:
            json_path: chunks JSON文件路径
            
        Returns:
            Chunk列表
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Chunks文件不存在: {json_path}")
        
        print(f"正在加载chunks: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = []
        for item in chunks_data:
            # 从JSON数据重建Chunk对象
            metadata = ChunkMetadata(**item['metadata'])
            chunk = Chunk(
                chunk_id=item['chunk_id'],
                text=item['text'],
                metadata=metadata
            )
            chunks.append(chunk)
        
        print(f"✓ 成功加载 {len(chunks)} 个chunks")
        return chunks
    
    def build_index(
        self,
        chunks: List[Chunk],
        chunk_embeddings: Optional[np.ndarray] = None,
        index_type: str = "flat"
    ):
        """
        构建faiss索引
        
        Args:
            chunks: Chunk列表
            chunk_embeddings: 预计算的chunk向量（可选）
            index_type: faiss索引类型 ("flat" 或 "ivf")
        """
        self.chunks = chunks
        
        # 如果没有提供embeddings，则计算
        if chunk_embeddings is None:
            print("正在计算chunk embeddings...")
            texts = [chunk.text for chunk in chunks]
            self.chunk_embeddings = self.embedder.encode_documents(
                texts,
                show_progress_bar=True
            )
        else:
            self.chunk_embeddings = chunk_embeddings
        
        embedding_dim = self.chunk_embeddings.shape[1]
        
        # 归一化向量（用于余弦相似度）
        faiss.normalize_L2(self.chunk_embeddings)
        
        # 创建faiss索引
        if index_type == "flat":
            # 精确搜索，适合小到中等规模数据
            self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        elif index_type == "ivf":
            # 近似搜索，适合大规模数据
            nlist = min(100, len(chunks) // 10)  # 聚类中心数
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            self.faiss_index.train(self.chunk_embeddings)
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")
        
        # 添加向量到索引
        self.faiss_index.add(self.chunk_embeddings)
        
        print(f"✓ 成功构建faiss索引: {len(chunks)} 个chunks, 维度 {embedding_dim}")
        print(f"  索引类型: {index_type}, 索引大小: {self.faiss_index.ntotal}")
    
    def rank(
        self,
        query: str,
        top_k: int = 20
    ) -> List[Tuple[Chunk, float]]:
        """
        使用faiss进行向量检索（ranking）
        
        Args:
            query: 查询文本
            top_k: 返回top-k结果
            
        Returns:
            List of (chunk, similarity_score) tuples, 按分数降序排列
        """
        if self.faiss_index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        # 编码查询
        query_embedding = self.embedder.encode_queries([query], show_progress_bar=False)
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def rerank(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 5,
        batch_size: int = 16
    ) -> List[Tuple[Chunk, float]]:
        """
        使用bge-reranker-large进行重排序
        
        Args:
            query: 查询文本
            chunks: 待重排序的chunks
            top_k: 返回top-k结果
            batch_size: 批处理大小
            
        Returns:
            List of (chunk, rerank_score) tuples, 按分数降序排列
        """
        if not chunks:
            return []
        
        # 准备输入对
        pairs = [[query, chunk.text] for chunk in chunks]
        
        # 批处理计算分数
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                # 计算分数
                outputs = self.reranker_model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                
                scores.extend(batch_scores.tolist())
        
        # 结合metadata加权（可选）
        if self.use_metadata:
            scores = self._apply_metadata_boost(query, chunks, scores)
        
        # 排序并返回top_k
        chunk_score_pairs = list(zip(chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return chunk_score_pairs[:top_k]
    
    def _apply_metadata_boost(
        self,
        query: str,
        chunks: List[Chunk],
        scores: List[float]
    ) -> List[float]:
        """
        使用metadata对分数进行加权
        
        策略：
        1. section_path中包含查询关键词的chunk加分
        2. 表格和列表类型的chunk可能更重要
        3. 更高级别的标题可能更相关
        """
        query_keywords = set(query.lower().split())
        boosted_scores = []
        
        for chunk, score in zip(chunks, scores):
            boost = 0.0
            
            # 检查section_path中是否包含查询关键词
            section_text = ' '.join(chunk.metadata.section_path).lower()
            for keyword in query_keywords:
                if keyword in section_text:
                    boost += 0.1  # 每个匹配的关键词加0.1分
            
            # 表格和列表类型加权
            if chunk.metadata.has_table:
                boost += 0.05
            if chunk.metadata.has_list:
                boost += 0.03
            
            # 标题级别加权（更高级别的标题可能更相关）
            if chunk.metadata.heading_level <= 2:
                boost += 0.02
            
            boosted_scores.append(score + boost)
        
        return boosted_scores
    
    def search(
        self,
        query: str,
        rank_top_k: int = 20,
        rerank_top_k: int = 5,
        use_rerank: bool = True
    ) -> List[Tuple[Chunk, float, Dict[str, Any]]]:
        """
        完整的检索流程：ranking + reranking
        
        Args:
            query: 查询文本
            rank_top_k: ranking阶段返回的top-k数量
            rerank_top_k: reranking阶段返回的top-k数量
            use_rerank: 是否使用rerank
            
        Returns:
            List of (chunk, final_score, metadata) tuples
        """
        # 第一步：向量检索
        ranked_results = self.rank(query, top_k=rank_top_k)
        
        if not use_rerank or not ranked_results:
            # 不使用rerank，直接返回ranking结果
            return [
                (chunk, score, {"rank_score": score, "rerank_score": None})
                for chunk, score in ranked_results[:rerank_top_k]
            ]
        
        # 第二步：重排序
        chunks_to_rerank = [chunk for chunk, _ in ranked_results]
        reranked_results = self.rerank(query, chunks_to_rerank, top_k=rerank_top_k)
        
        # 合并结果，包含原始ranking分数和rerank分数
        final_results = []
        rank_scores_dict = {chunk.chunk_id: score for chunk, score in ranked_results}
        
        for chunk, rerank_score in reranked_results:
            rank_score = rank_scores_dict.get(chunk.chunk_id, 0.0)
            final_results.append((
                chunk,
                rerank_score,  # 使用rerank分数作为最终分数
                {
                    "rank_score": rank_score,
                    "rerank_score": rerank_score,
                    "metadata": {
                        "section_path": chunk.metadata.section_path,
                        "chunk_type": chunk.metadata.chunk_type,
                        "has_table": chunk.metadata.has_table,
                        "has_list": chunk.metadata.has_list
                    }
                }
            ))
        
        return final_results
    
    def get_index_info(self) -> Dict[str, Any]:
        """获取索引信息"""
        if self.faiss_index is None:
            return {"status": "索引未构建"}
        
        return {
            "index_type": type(self.faiss_index).__name__,
            "total_chunks": self.faiss_index.ntotal,
            "embedding_dim": self.chunk_embeddings.shape[1] if self.chunk_embeddings is not None else None,
            "device": self.device,
            "use_metadata": self.use_metadata
        }


# 便捷函数
def create_reranker(
    embedder: Embedder,
    reranker_model_name: str = "BAAI/bge-reranker-large",
    reranker_model_path: Optional[str] = None,
    **kwargs
) -> Reranker:
    """
    创建Reranker实例的便捷函数
    
    Args:
        embedder: Embedder实例
        reranker_model_name: reranker模型名称
        reranker_model_path: 本地reranker模型路径
        **kwargs: 其他参数
        
    Returns:
        Reranker实例
    """
    return Reranker(
        embedder=embedder,
        reranker_model_name=reranker_model_name,
        reranker_model_path=reranker_model_path,
        **kwargs
    )


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("测试 Reranker 模块")
    print("=" * 70)
    
    # 创建embedder
    from app.embedder import create_embedder
    embedder = create_embedder()
    
    # 创建reranker
    reranker = create_reranker(embedder)
    
    # 测试数据
    test_chunks = [
        "意外伤害保险理赔流程说明，需要准备身份证、医疗证明等材料",
        "重大疾病保险条款详解，包含30种重大疾病的保障范围",
        "车险理赔所需材料清单：驾驶证、行驶证、事故证明等",
        "人寿保险投保须知，年龄限制和健康告知要求"
    ]
    
    # 构建索引（使用简单的测试数据）
    chunks = []
    for i, text in enumerate(test_chunks):
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
    
    reranker.build_index(chunks)
    
    # 测试检索
    query = "如何申请意外险理赔？"
    print(f"\n查询: {query}")
    results = reranker.search(query, rank_top_k=10, rerank_top_k=3)
    
    print("\n检索结果:")
    for i, (chunk, score, info) in enumerate(results, 1):
        print(f"\n{i}. [分数: {score:.4f}]")
        print(f"   文本: {chunk.text[:100]}...")
        print(f"   Ranking分数: {info['rank_score']:.4f}")
        print(f"   Rerank分数: {info['rerank_score']:.4f}")
