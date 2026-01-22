"""
RAG Pipeline 配置模块
定义整个RAG流程的配置参数
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGPipelineConfig:
    """
    RAG Pipeline 完整配置类
    
    包含从OCR到LLM的所有配置参数
    """
    # ==================== 路径配置 ====================
    # 数据目录
    raw_data_dir: Path = Path("data/raw_data")
    processed_dir: Path = Path("data/processed")
    cleaned_dir: Path = Path("data/cleaned")
    chunks_dir: Path = Path("data/chunks")
    
    # ==================== OCR配置 ====================
    ocr_output_dir: Optional[Path] = None  # 如果为None，使用processed_dir
    ocr_source: str = "modelscope"  # 'modelscope' 或 'huggingface'
    ocr_use_gpu: bool = True
    ocr_use_paddleocr_slim: bool = True  # 使用轻量模型节省显存
    ocr_extract_images: bool = True
    ocr_extract_tables: bool = True
    
    # ==================== 文本清洗配置 ====================
    enable_text_cleaning: bool = True
    text_cleaner_min_repeat_length: int = 20
    text_cleaner_repeat_threshold: int = 3
    save_cleaned_text: bool = True
    
    # ==================== Chunker配置 ====================
    chunker_target_size: int = 800
    chunker_max_size: int = 1500
    chunker_min_size: int = 200
    chunker_overlap_size: int = 100
    enable_semantic_splitting: bool = True
    enable_terminology: bool = True
    terminology_file: Optional[Path] = None
    
    # ==================== Embedder配置 ====================
    embedder_model_name: str = "BAAI/bge-large-zh-v1.5"
    embedder_model_path: Optional[str] = None
    embedder_device: Optional[str] = None  # None表示自动检测
    embedder_use_mirror: bool = True
    
    # ==================== Reranker配置 ====================
    reranker_model_name: str = "BAAI/bge-reranker-large"
    reranker_model_path: Optional[str] = None
    reranker_device: Optional[str] = None
    reranker_use_mirror: bool = True
    reranker_use_metadata: bool = True
    reranker_rank_top_k: int = 20  # ranking阶段返回的top-k
    reranker_rerank_top_k: int = 5  # reranking阶段返回的top-k
    
    # ==================== LLM配置 ====================
    llm_model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_model_path: Optional[str] = None
    llm_device: Optional[str] = None  # None表示自动检测，'cpu'强制使用CPU
    llm_use_mirror: bool = True
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_max_context_length: int = 3000  # RAG prompt的最大上下文长度
    llm_load_in_8bit: bool = False  # 使用8bit量化（显存减半，需要bitsandbytes）
    llm_load_in_4bit: bool = False  # 使用4bit量化（显存减少75%，需要bitsandbytes）
    
    def __post_init__(self):
        """初始化后处理：确保路径是Path对象"""
        # 转换路径为Path对象
        if isinstance(self.raw_data_dir, str):
            self.raw_data_dir = Path(self.raw_data_dir)
        if isinstance(self.processed_dir, str):
            self.processed_dir = Path(self.processed_dir)
        if isinstance(self.cleaned_dir, str):
            self.cleaned_dir = Path(self.cleaned_dir)
        if isinstance(self.chunks_dir, str):
            self.chunks_dir = Path(self.chunks_dir)
        if isinstance(self.ocr_output_dir, str):
            self.ocr_output_dir = Path(self.ocr_output_dir)
        elif self.ocr_output_dir is None:
            self.ocr_output_dir = self.processed_dir
        
        # 创建必要的目录
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)


# 默认配置实例
default_config = RAGPipelineConfig()
