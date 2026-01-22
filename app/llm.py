"""
LLM模块 - 使用Qwen2.5模型进行RAG答案生成
"""
import os
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from app.chunker import Chunk


# 类级别的模型缓存，避免重复加载占用显存
_model_cache: Dict[str, Tuple[Any, Any]] = {}  # key: (model_name, device, model_path), value: (tokenizer, model)


class LLM:
    """
    大语言模型类，用于RAG答案生成
    
    功能：
    1. 加载Qwen2.5模型
    2. 构建RAG prompt（查询 + 检索到的chunks）
    3. 生成答案
    4. 支持模型缓存机制
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_mirror: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        初始化LLM
        
        Args:
            model_name: 模型名称，默认使用Qwen2.5-3B-Instruct（显存需求约6-8GB）
                        可选：Qwen2.5-1.5B-Instruct（约3-4GB）、Qwen2.5-7B-Instruct（约14GB）
            model_path: 本地模型路径
            device: 设备类型 ('cuda', 'cpu' 或 None自动检测)
            use_mirror: 是否使用国内镜像源（hf-mirror.com）
            max_new_tokens: 最大生成token数
            temperature: 生成温度（控制随机性）
            top_p: nucleus sampling参数
            load_in_8bit: 使用8bit量化（显存减半，需要bitsandbytes库）
            load_in_4bit: 使用4bit量化（显存减少75%，需要bitsandbytes库）
        """
        self.model_name = model_name
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # 自动检测设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # 量化只能在CUDA上使用
        if (load_in_8bit or load_in_4bit) and device == 'cpu':
            print("⚠ 警告: 量化只能在CUDA设备上使用，已禁用量化")
            load_in_8bit = False
            load_in_4bit = False
        
        # 生成缓存key：模型名称+设备+路径+量化配置
        quant_suffix = ""
        if load_in_4bit:
            quant_suffix = "_4bit"
        elif load_in_8bit:
            quant_suffix = "_8bit"
        cache_key = f"{model_path or model_name}_{device}{quant_suffix}"
        
        # 检查模型缓存
        global _model_cache
        if cache_key in _model_cache:
            # 复用已加载的模型
            self.tokenizer, self.model = _model_cache[cache_key]
            print(f"✓ 复用已缓存的LLM模型: {cache_key}")
        else:
            # 设置国内镜像源（如果启用且不是从本地路径加载）
            original_hf_endpoint = None
            if use_mirror and not model_path:
                # 保存原始环境变量
                original_hf_endpoint = os.environ.get('HF_ENDPOINT')
                # 设置国内镜像
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                print("✓ 已配置使用国内镜像源: https://hf-mirror.com")
            
            # 加载模型
            print(f"正在加载LLM模型: {model_path or model_name}")
            try:
                # 准备加载参数
                load_kwargs = {
                    "trust_remote_code": True
                }
                
                # 量化配置（显存优化）
                if load_in_8bit or load_in_4bit:
                    try:
                        from transformers import BitsAndBytesConfig
                        if load_in_4bit:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            load_kwargs["quantization_config"] = quantization_config
                            print("✓ 启用4bit量化（显存减少约75%）")
                        elif load_in_8bit:
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                            load_kwargs["quantization_config"] = quantization_config
                            print("✓ 启用8bit量化（显存减少约50%）")
                    except ImportError:
                        print("⚠ 警告: bitsandbytes未安装，无法使用量化。安装: pip install bitsandbytes")
                        print("   继续使用标准加载方式...")
                        load_in_8bit = False
                        load_in_4bit = False
                
                # 设备配置
                if device == 'cuda' and not load_in_8bit and not load_in_4bit:
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
                elif device == 'cpu':
                    load_kwargs["torch_dtype"] = torch.float32
                
                # 加载模型
                if model_path and os.path.exists(model_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
                    if device == 'cpu' and not load_in_8bit and not load_in_4bit:
                        self.model = self.model.to(device)
                    print(f"✓ 成功从本地加载LLM模型: {model_path}")
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                    if device == 'cpu' and not load_in_8bit and not load_in_4bit:
                        self.model = self.model.to(device)
                    print(f"✓ 成功加载LLM模型: {model_name}")
                
                # 将模型加入缓存
                _model_cache[cache_key] = (self.tokenizer, self.model)
                print(f"✓ 模型已缓存，后续实例将复用此模型（节省显存）")
                
            except Exception as e:
                print(f"✗ LLM模型加载失败: {e}")
                raise
            finally:
                # 恢复原始环境变量
                if use_mirror and not model_path:
                    if original_hf_endpoint is not None:
                        os.environ['HF_ENDPOINT'] = original_hf_endpoint
                    elif 'HF_ENDPOINT' in os.environ:
                        del os.environ['HF_ENDPOINT']
        
        self.model.eval()
    
    def build_rag_prompt(
        self,
        query: str,
        chunks: List[Chunk],
        max_context_length: int = 3000
    ) -> str:
        """
        构建RAG prompt
        
        Args:
            query: 用户查询
            chunks: 检索到的chunks
            max_context_length: 最大上下文长度（字符数）
            
        Returns:
            构建好的prompt字符串
        """
        # 构建上下文
        context_parts = []
        current_length = 0
        
        for i, chunk in enumerate(chunks, 1):
            # 构建chunk文本，包含章节信息
            section_info = f"【{' > '.join(chunk.metadata.section_path)}】" if chunk.metadata.section_path else ""
            chunk_text = f"{section_info}\n{chunk.text}" if section_info else chunk.text
            
            chunk_with_number = f"文档片段{i}：\n{chunk_text}\n"
            
            # 检查是否超过最大长度
            if current_length + len(chunk_with_number) > max_context_length:
                break
            
            context_parts.append(chunk_with_number)
            current_length += len(chunk_with_number)
        
        context = "\n".join(context_parts)
        
        # 构建Qwen2.5的prompt格式
        prompt = f"""你是一个专业的保险知识助手。请根据以下文档内容回答用户的问题。

## 参考文档：

{context}

## 用户问题：

{query}

## 回答要求：

1. 基于参考文档中的内容回答问题
2. 如果文档中没有相关信息，请明确说明
3. 回答要准确、专业、易懂
4. 如果涉及具体条款，请引用文档中的原文

## 回答："""
        
        return prompt
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        """
        生成答案
        
        Args:
            prompt: 输入prompt
            max_new_tokens: 最大生成token数（如果为None则使用初始化时的值）
            temperature: 生成温度（如果为None则使用初始化时的值）
            top_p: nucleus sampling参数（如果为None则使用初始化时的值）
            
        Returns:
            生成的答案文本
        """
        max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        
        try:
            # 构建messages（Qwen2.5的格式）
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Tokenize
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            # 生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            print(f"✗ 生成答案失败: {e}")
            raise
    
    def answer(
        self,
        query: str,
        chunks: List[Chunk],
        max_context_length: int = 3000,
        **generate_kwargs
    ) -> Dict[str, Any]:
        """
        完整的RAG答案生成流程
        
        Args:
            query: 用户查询
            chunks: 检索到的chunks
            max_context_length: 最大上下文长度
            **generate_kwargs: 传递给generate的其他参数
            
        Returns:
            包含答案和元信息的字典
        """
        # 构建prompt
        prompt = self.build_rag_prompt(query, chunks, max_context_length)
        
        # 生成答案
        answer = self.generate(prompt, **generate_kwargs)
        
        return {
            "answer": answer,
            "query": query,
            "num_chunks_used": len(chunks),
            "prompt_length": len(prompt)
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit
        }


# 便捷函数
def create_llm(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    model_path: Optional[str] = None,
    use_mirror: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    **kwargs
) -> LLM:
    """
    创建LLM实例的便捷函数
    
    注意：模型会被缓存，多个实例共享同一个模型，节省显存。
    相同模型名称+设备+路径+量化配置的实例会复用同一个模型。
    
    Args:
        model_name: 模型名称
        model_path: 本地模型路径
        use_mirror: 是否使用国内镜像源（默认True）
        load_in_8bit: 使用8bit量化（显存减半）
        load_in_4bit: 使用4bit量化（显存减少75%）
        **kwargs: 其他参数
        
    Returns:
        LLM实例
    """
    return LLM(
        model_name=model_name,
        model_path=model_path,
        use_mirror=use_mirror,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        **kwargs
    )


def clear_model_cache():
    """
    清空模型缓存，释放显存
    
    注意：清空后，后续创建的LLM实例会重新加载模型
    """
    global _model_cache
    _model_cache.clear()
    print("✓ LLM模型缓存已清空")


def get_model_cache_info() -> Dict[str, Any]:
    """
    获取模型缓存信息
    
    Returns:
        缓存信息字典
    """
    global _model_cache
    return {
        "cached_models": list(_model_cache.keys()),
        "cache_count": len(_model_cache)
    }


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("测试 LLM 模块")
    print("=" * 70)
    
    # 创建LLM（使用3B模型，显存需求更低）
    llm = create_llm(model_name="Qwen/Qwen2.5-3B-Instruct")
    
    # 测试数据
    from app.chunker import Chunk, ChunkMetadata
    
    test_chunks = [
        Chunk(
            chunk_id="test_1",
            text="意外伤害保险理赔流程说明。当发生意外伤害事故时，被保险人需要准备以下材料：1. 身份证原件及复印件；2. 医疗诊断证明书；3. 医疗费用发票；4. 事故证明文件；5. 保险合同原件。理赔申请提交后，保险公司会在15个工作日内完成审核并支付理赔款。",
            metadata=ChunkMetadata(
                chunk_id="test_1",
                chunk_type="paragraph",
                section_path=["保险理赔"],
                heading_level=2,
                char_count=150,
                image_refs=[],
                source_file="test.md"
            )
        )
    ]
    
    # 测试生成
    query = "如何申请意外险理赔？"
    print(f"\n查询: {query}")
    
    result = llm.answer(query, test_chunks)
    print(f"\n答案: {result['answer']}")
    print(f"\n使用的chunks数量: {result['num_chunks_used']}")
