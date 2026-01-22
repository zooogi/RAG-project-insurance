"""
RAG Pipeline 主流程模块
串联OCR -> 清洗 -> Chunk -> Embedding -> Rerank -> LLM的完整流程
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.config import RAGPipelineConfig
from app.ocr import DocumentProcessor
from app.chunker import SemanticChunker, Chunk
from app.embedder import Embedder, create_embedder
from app.reranker import Reranker, create_reranker
from app.llm import LLM, create_llm


class RAGPipeline:
    """
    RAG完整流程管道
    
    功能：
    1. 处理文档：OCR -> 清洗 -> Chunk -> Embedding索引构建
    2. 查询答案：Query -> Rerank检索 -> LLM生成答案
    """
    
    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        """
        初始化RAG Pipeline
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or RAGPipelineConfig()
        
        # 初始化各个组件（延迟加载，只在需要时创建）
        self.ocr_processor: Optional[DocumentProcessor] = None
        self.chunker: Optional[SemanticChunker] = None
        self.embedder: Optional[Embedder] = None
        self.reranker: Optional[Reranker] = None
        self.llm: Optional[LLM] = None
        
        # 存储已处理的chunks（用于构建索引）
        self.chunks: List[Chunk] = []
        self.index_built: bool = False
    
    # ==================== 组件初始化方法 ====================
    
    def _init_ocr_processor(self) -> DocumentProcessor:
        """初始化OCR处理器"""
        if self.ocr_processor is None:
            self.ocr_processor = DocumentProcessor(
                output_base_dir=str(self.config.ocr_output_dir),
                source=self.config.ocr_source,
                use_gpu=self.config.ocr_use_gpu,
                use_paddleocr_slim=self.config.ocr_use_paddleocr_slim
            )
        return self.ocr_processor
    
    def _init_chunker(self) -> SemanticChunker:
        """初始化Chunker"""
        if self.chunker is None:
            from app.text_cleaner import TextCleaner
            
            text_cleaner = None
            if self.config.enable_text_cleaning:
                text_cleaner = TextCleaner(
                    min_repeat_length=self.config.text_cleaner_min_repeat_length,
                    repeat_threshold=self.config.text_cleaner_repeat_threshold
                )
            
            self.chunker = SemanticChunker(
                target_chunk_size=self.config.chunker_target_size,
                max_chunk_size=self.config.chunker_max_size,
                min_chunk_size=self.config.chunker_min_size,
                overlap_size=self.config.chunker_overlap_size,
                enable_text_cleaning=self.config.enable_text_cleaning,
                text_cleaner=text_cleaner,
                save_cleaned_text=self.config.save_cleaned_text,
                cleaned_output_dir=self.config.cleaned_dir,
                enable_semantic_splitting=self.config.enable_semantic_splitting,
                enable_terminology=self.config.enable_terminology,
                terminology_file=self.config.terminology_file
            )
        return self.chunker
    
    def _init_embedder(self) -> Embedder:
        """初始化Embedder"""
        if self.embedder is None:
            self.embedder = create_embedder(
                model_name=self.config.embedder_model_name,
                model_path=self.config.embedder_model_path,
                device=self.config.embedder_device,
                use_mirror=self.config.embedder_use_mirror
            )
        return self.embedder
    
    def _init_reranker(self) -> Reranker:
        """初始化Reranker"""
        if self.reranker is None:
            embedder = self._init_embedder()
            self.reranker = create_reranker(
                embedder=embedder,
                reranker_model_name=self.config.reranker_model_name,
                reranker_model_path=self.config.reranker_model_path,
                device=self.config.reranker_device,
                use_metadata=self.config.reranker_use_metadata,
                use_mirror=self.config.reranker_use_mirror
            )
        return self.reranker
    
    def _init_llm(self) -> LLM:
        """初始化LLM"""
        if self.llm is None:
            self.llm = create_llm(
                model_name=self.config.llm_model_name,
                model_path=self.config.llm_model_path,
                device=self.config.llm_device,
                use_mirror=self.config.llm_use_mirror,
                max_new_tokens=self.config.llm_max_new_tokens,
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p,
                load_in_8bit=self.config.llm_load_in_8bit,
                load_in_4bit=self.config.llm_load_in_4bit
            )
        return self.llm
    
    # ==================== 文档处理流程 ====================
    
    def process_documents(
        self,
        input_path: Optional[Path] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        处理文档：OCR -> 清洗 -> Chunk
        
        Args:
            input_path: 输入路径（文件或目录），如果为None则使用config中的raw_data_dir
            overwrite: 是否覆盖已存在的处理结果
        
        Returns:
            处理结果字典，包含：
            - ocr_results: OCR处理结果列表
            - chunks: 生成的chunks列表
            - chunk_files: 保存的chunk JSON文件路径列表
        """
        print("=" * 70)
        print("📄 开始处理文档")
        print("=" * 70)
        
        if input_path is None:
            input_path = self.config.raw_data_dir
        
        input_path = Path(input_path)
        
        # 步骤1: OCR处理
        print("\n【步骤1/3】OCR处理...")
        ocr_processor = self._init_ocr_processor()
        
        if input_path.is_file():
            # 处理单个文件
            ocr_result = ocr_processor.process_file(
                input_path,
                extract_images=self.config.ocr_extract_images,
                extract_tables=self.config.ocr_extract_tables,
                overwrite=overwrite
            )
            ocr_results = [ocr_result]
        else:
            # 批量处理目录
            ocr_results = ocr_processor.batch_process(
                input_path,
                overwrite=overwrite
            )
        
        print(f"✓ OCR处理完成，共处理 {len(ocr_results)} 个文件")
        
        # 步骤2: Chunk处理
        print("\n【步骤2/3】文本清洗和分块...")
        chunker = self._init_chunker()
        
        all_chunks = []
        chunk_files = []
        
        # 从OCR结果中找到所有markdown文件
        md_files = []
        for ocr_result in ocr_results:
            if ocr_result.get('file_type') in ['pdf', 'image', 'csv']:
                # 尝试多种方式获取markdown文件路径
                md_path = None
                if 'files' in ocr_result and 'markdown' in ocr_result['files']:
                    md_path = Path(ocr_result['files']['markdown'])
                elif 'output_path' in ocr_result:
                    md_path = Path(ocr_result['output_path'])
                elif 'output_dir' in ocr_result:
                    # 从output_dir和file_name构建路径
                    output_dir = Path(ocr_result['output_dir'])
                    file_name = ocr_result.get('file_name', '')
                    # PDF文件可能在子目录中
                    if ocr_result.get('file_type') == 'pdf':
                        md_path = output_dir / file_name / "hybrid_auto" / f"{file_name}.md"
                    else:
                        md_path = output_dir / f"{file_name}.md"
                
                if md_path and md_path.exists() and md_path.suffix == '.md':
                    md_files.append(md_path)
                elif md_path:
                    print(f"⚠ 未找到markdown文件: {md_path}")
        
        # 处理每个markdown文件
        for md_file in md_files:
            print(f"\n处理文件: {md_file.name}")
            chunks = chunker.chunk_markdown_file(md_file)
            all_chunks.extend(chunks)
            
            # 保存chunks到JSON文件
            chunk_file = self.config.chunks_dir / f"{md_file.stem}_chunks.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [chunk.to_dict() for chunk in chunks],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            chunk_files.append(chunk_file)
            print(f"✓ 生成 {len(chunks)} 个chunks，已保存到: {chunk_file}")
        
        print(f"\n✓ 分块处理完成，共生成 {len(all_chunks)} 个chunks")
        
        # 步骤3: 构建Embedding索引
        print("\n【步骤3/3】构建Embedding索引...")
        if all_chunks:
            reranker = self._init_reranker()
            reranker.build_index(all_chunks)
            self.chunks = all_chunks
            self.index_built = True
            print(f"✓ 索引构建完成，共 {len(all_chunks)} 个chunks")
        else:
            print("⚠ 没有chunks可构建索引")
        
        print("\n" + "=" * 70)
        print("✓ 文档处理流程完成！")
        print("=" * 70)
        
        return {
            "ocr_results": ocr_results,
            "chunks": all_chunks,
            "chunk_files": chunk_files,
            "index_built": self.index_built
        }
    
    def load_chunks_from_files(
        self,
        chunk_files: Optional[List[Path]] = None
    ) -> List[Chunk]:
        """
        从JSON文件加载chunks并构建索引
        
        Args:
            chunk_files: chunk JSON文件路径列表，如果为None则从chunks_dir加载所有文件
        
        Returns:
            加载的chunks列表
        """
        if chunk_files is None:
            # 加载chunks_dir下的所有chunk文件
            chunk_files = list(self.config.chunks_dir.glob("*_chunks.json"))
        
        print(f"📂 从 {len(chunk_files)} 个文件加载chunks...")
        
        all_chunks = []
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_dicts = json.load(f)
            
            chunks = [Chunk.from_dict(chunk_dict) for chunk_dict in chunk_dicts]
            all_chunks.extend(chunks)
            print(f"✓ 从 {chunk_file.name} 加载了 {len(chunks)} 个chunks")
        
        # 构建索引
        if all_chunks:
            reranker = self._init_reranker()
            reranker.build_index(all_chunks)
            self.chunks = all_chunks
            self.index_built = True
            print(f"\n✓ 索引构建完成，共 {len(all_chunks)} 个chunks")
        
        return all_chunks
    
    # ==================== 查询流程 ====================
    
    def query(
        self,
        query: str,
        use_rerank: bool = True,
        return_chunks: bool = False
    ) -> Dict[str, Any]:
        """
        查询答案：检索 -> Rerank -> LLM生成
        
        Args:
            query: 用户查询文本
            use_rerank: 是否使用rerank
            return_chunks: 是否在结果中返回检索到的chunks
        
        Returns:
            结果字典，包含：
            - answer: LLM生成的答案
            - query: 原始查询
            - chunks: 检索到的chunks（如果return_chunks=True）
            - metadata: 其他元信息
        """
        if not self.index_built:
            raise ValueError("索引未构建！请先调用 process_documents() 或 load_chunks_from_files()")
        
        print("=" * 70)
        print(f"🔍 查询: {query}")
        print("=" * 70)
        
        # 步骤1: 检索
        print("\n【步骤1/2】检索相关chunks...")
        reranker = self._init_reranker()
        
        search_results = reranker.search(
            query,
            rank_top_k=self.config.reranker_rank_top_k,
            rerank_top_k=self.config.reranker_rerank_top_k,
            use_rerank=use_rerank
        )
        
        retrieved_chunks = [chunk for chunk, _, _ in search_results]
        print(f"✓ 检索到 {len(retrieved_chunks)} 个相关chunks")
        
        # 步骤2: LLM生成答案
        print("\n【步骤2/2】生成答案...")
        llm = self._init_llm()
        
        result = llm.answer(
            query,
            retrieved_chunks,
            max_context_length=self.config.llm_max_context_length
        )
        
        print(f"✓ 答案生成完成")
        
        # 构建返回结果
        response = {
            "answer": result["answer"],
            "query": query,
            "num_chunks_used": result["num_chunks_used"],
            "prompt_length": result["prompt_length"],
            "metadata": {
                "rank_top_k": self.config.reranker_rank_top_k,
                "rerank_top_k": self.config.reranker_rerank_top_k,
                "use_rerank": use_rerank
            }
        }
        
        if return_chunks:
            response["chunks"] = [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    "section_path": chunk.metadata.section_path,
                    "score": score,
                    "rank_score": info.get("rank_score"),
                    "rerank_score": info.get("rerank_score")
                }
                for chunk, score, info in search_results
            ]
        
        return response
    
    def get_status(self) -> Dict[str, Any]:
        """获取Pipeline当前状态"""
        return {
            "index_built": self.index_built,
            "num_chunks": len(self.chunks),
            "components_loaded": {
                "ocr_processor": self.ocr_processor is not None,
                "chunker": self.chunker is not None,
                "embedder": self.embedder is not None,
                "reranker": self.reranker is not None,
                "llm": self.llm is not None
            }
        }


# ==================== 便捷函数 ====================

def create_pipeline(config: Optional[RAGPipelineConfig] = None) -> RAGPipeline:
    """
    创建RAG Pipeline实例的便捷函数
    
    Args:
        config: 配置对象，如果为None则使用默认配置
    
    Returns:
        RAGPipeline实例
    """
    return RAGPipeline(config)


# ==================== 命令行接口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline - 完整的RAG流程")
    parser.add_argument(
        "mode",
        choices=["process", "query", "load"],
        help="运行模式: process(处理文档), query(查询), load(加载chunks)"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="输入路径（文件或目录，用于process模式）"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="查询文本（用于query模式）"
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        help="Chunks目录（用于load模式）"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径（JSON格式，可选）"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = RAGPipelineConfig()
    if args.config:
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            # 更新配置
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # 创建pipeline
    pipeline = create_pipeline(config)
    
    if args.mode == "process":
        # 处理文档
        input_path = Path(args.input) if args.input else None
        result = pipeline.process_documents(input_path)
        print(f"\n✓ 处理完成！共生成 {len(result['chunks'])} 个chunks")
    
    elif args.mode == "load":
        # 加载chunks
        if args.chunks_dir:
            config.chunks_dir = Path(args.chunks_dir)
        chunks = pipeline.load_chunks_from_files()
        print(f"\n✓ 加载完成！共 {len(chunks)} 个chunks")
    
    elif args.mode == "query":
        # 查询
        if not args.query:
            print("错误: query模式需要提供 --query 参数")
            exit(1)
        
        # 如果索引未构建，尝试加载chunks
        if not pipeline.index_built:
            print("索引未构建，尝试加载chunks...")
            pipeline.load_chunks_from_files()
        
        result = pipeline.query(args.query, return_chunks=True)
        print(f"\n答案: {result['answer']}")
        print(f"\n使用的chunks数量: {result['num_chunks_used']}")
