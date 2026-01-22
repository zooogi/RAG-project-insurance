# app/chunker.py

import re
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field

from app.text_cleaner import TextCleaner, SentenceInfo


# ============================================================================
# 语义切割相关类和函数
# ============================================================================

@dataclass
class SemanticAtom:
    """语义原子"""
    text: str
    semantic_type: str  # '给付', '免责', '条件', '定义', '其他'
    start_pos: int  # 在原文本中的起始位置
    end_pos: int  # 在原文本中的结束位置
    trigger_words: List[str] = field(default_factory=list)  # 触发词列表


class SemanticSplitter:
    """
    语义切割器 - 基于规则识别语义触发词并拆成语义原子
    """
    
    def __init__(self):
        """初始化语义切割器"""
        # 给付词汇模式
        self.payment_patterns = [
            r'保险人\s*(?:给付|赔付|支付|承担|负责)',
            r'按照\s*(?:合同|本条款|本协议)\s*(?:约定|规定)\s*(?:给付|赔付|支付)',
            r'给付\s*(?:保险金|赔偿金|金额)',
            r'赔付\s*(?:保险金|赔偿金|金额)',
            r'承担\s*(?:保险责任|赔偿责任)',
            r'负责\s*(?:给付|赔付|支付)',
        ]
        
        # 免责词汇模式
        self.exclusion_patterns = [
            r'但\s*(?:是|为|在|如|若|当)',
            r'除外',
            r'不\s*(?:承担|负责|给付|赔付|支付|予赔付)',
            r'不予\s*(?:赔付|给付|支付)',
            r'免除\s*(?:责任|赔偿责任)',
            r'不\s*(?:在|属于)\s*(?:保险责任|保障范围)',
            r'不在\s*(?:保险责任|保障范围)\s*(?:内|之内)',
        ]
        
        # 条件结构模式
        self.condition_patterns = [
            r'因\s*[^，。；]*?\s*导致',
            r'在\s*[^，。；]*?\s*情况下',
            r'当\s*[^，。；]*?\s*时',
            r'若\s*[^，。；]*?[，。；]',
            r'如\s*[^，。；]*?[，。；]',
            r'如果\s*[^，。；]*?[，。；]',
            r'倘若\s*[^，。；]*?[，。；]',
            r'只要\s*[^，。；]*?[，。；]',
            r'除非\s*[^，。；]*?[，。；]',
        ]
        
        # 定义模式（用于识别术语定义）
        self.definition_patterns = [
            r'[^，。；]*?\s*(?:指|是指|为|是指|系指|即)',
            r'[^，。；]*?\s*(?:包括|包含|涵盖|涉及)',
        ]
        
        # 编译正则表达式
        self.payment_regexes = [re.compile(pattern) for pattern in self.payment_patterns]
        self.exclusion_regexes = [re.compile(pattern) for pattern in self.exclusion_patterns]
        self.condition_regexes = [re.compile(pattern) for pattern in self.condition_patterns]
        self.definition_regexes = [re.compile(pattern) for pattern in self.definition_patterns]
    
    def identify_semantic_type(self, text: str) -> Tuple[str, List[str]]:
        """
        识别文本的语义类型
        
        Args:
            text: 文本内容
            
        Returns:
            (语义类型, 触发词列表)
        """
        trigger_words = []
        
        # 检查给付词汇
        for regex in self.payment_regexes:
            matches = regex.findall(text)
            if matches:
                trigger_words.extend(matches)
                return ('给付', trigger_words)
        
        # 检查免责词汇
        for regex in self.exclusion_regexes:
            matches = regex.findall(text)
            if matches:
                trigger_words.extend(matches)
                return ('免责', trigger_words)
        
        # 检查条件结构
        for regex in self.condition_regexes:
            matches = regex.findall(text)
            if matches:
                trigger_words.extend(matches)
                return ('条件', trigger_words)
        
        # 检查定义
        for regex in self.definition_regexes:
            matches = regex.findall(text)
            if matches:
                trigger_words.extend(matches)
                return ('定义', trigger_words)
        
        return ('其他', [])
    
    def split_into_semantic_atoms(self, text: str) -> List[SemanticAtom]:
        """
        将文本拆分成语义原子
        
        Args:
            text: 原始文本
            
        Returns:
            语义原子列表
        """
        atoms = []
        
        # 按句号、分号、换行等分割成句子
        sentences = re.split(r'([。；\n])', text)
        
        current_atom_text = ""
        current_atom_type = None
        current_trigger_words = []
        start_pos = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            
            # 跳过分隔符
            if sentence in ['。', '；', '\n']:
                if current_atom_text:
                    current_atom_text += sentence
                i += 1
                continue
            
            # 识别当前句子的语义类型
            semantic_type, trigger_words = self.identify_semantic_type(sentence)
            
            # 如果语义类型改变，保存当前原子并开始新原子
            if current_atom_type and semantic_type != current_atom_type and current_atom_type != '其他':
                # 保存当前原子
                atom_text = current_atom_text.strip()
                if atom_text:
                    atoms.append(SemanticAtom(
                        text=atom_text,
                        semantic_type=current_atom_type,
                        start_pos=start_pos,
                        end_pos=start_pos + len(atom_text),
                        trigger_words=current_trigger_words.copy()
                    ))
                
                # 开始新原子
                current_atom_text = sentence
                current_atom_type = semantic_type
                current_trigger_words = trigger_words
                start_pos = start_pos + len(atom_text) if atom_text else start_pos
            else:
                # 继续当前原子
                if not current_atom_type or current_atom_type == '其他':
                    current_atom_type = semantic_type
                    current_trigger_words = trigger_words
                    start_pos = start_pos if current_atom_text else start_pos
                
                current_atom_text += sentence
                if trigger_words:
                    current_trigger_words.extend(trigger_words)
            
            i += 1
        
        # 保存最后一个原子
        if current_atom_text.strip():
            atoms.append(SemanticAtom(
                text=current_atom_text.strip(),
                semantic_type=current_atom_type or '其他',
                start_pos=start_pos,
                end_pos=start_pos + len(current_atom_text.strip()),
                trigger_words=current_trigger_words
            ))
        
        # 如果没有识别到语义原子，返回整个文本作为一个原子
        if not atoms:
            atoms.append(SemanticAtom(
                text=text,
                semantic_type='其他',
                start_pos=0,
                end_pos=len(text),
                trigger_words=[]
            ))
        
        return atoms
    
    def split_long_atom(self, atom: SemanticAtom, max_length: int = 500) -> List[SemanticAtom]:
        """
        如果语义原子过长，按结构/长度切分（不跨语义）
        
        Args:
            atom: 语义原子
            max_length: 最大长度
            
        Returns:
            切分后的语义原子列表
        """
        if len(atom.text) <= max_length:
            return [atom]
        
        # 尝试按编号切分（如：1. 2. 3. 或 一、二、三、）
        numbered_pattern = r'[（(]?\s*[一二三四五六七八九十\d]+[、.)）]\s*'
        splits = re.split(numbered_pattern, atom.text)
        
        if len(splits) > 1:
            # 找到所有编号位置
            matches = list(re.finditer(numbered_pattern, atom.text))
            sub_atoms = []
            current_pos = 0
            
            for i, match in enumerate(matches):
                # 添加编号之前的内容
                if match.start() > current_pos:
                    prev_text = atom.text[current_pos:match.start()]
                    if prev_text.strip():
                        sub_atoms.append(SemanticAtom(
                            text=prev_text.strip(),
                            semantic_type=atom.semantic_type,
                            start_pos=atom.start_pos + current_pos,
                            end_pos=atom.start_pos + match.start(),
                            trigger_words=atom.trigger_words.copy() if i == 0 else []
                        ))
                
                # 添加编号和后续内容
                next_match = matches[i + 1] if i + 1 < len(matches) else None
                end_pos = next_match.start() if next_match else len(atom.text)
                numbered_text = atom.text[match.start():end_pos]
                
                if numbered_text.strip():
                    sub_atoms.append(SemanticAtom(
                        text=numbered_text.strip(),
                        semantic_type=atom.semantic_type,
                        start_pos=atom.start_pos + match.start(),
                        end_pos=atom.start_pos + end_pos,
                        trigger_words=atom.trigger_words.copy() if i == 0 else []
                    ))
                
                current_pos = end_pos
            
            # 添加最后剩余的内容
            if current_pos < len(atom.text):
                remaining_text = atom.text[current_pos:]
                if remaining_text.strip():
                    sub_atoms.append(SemanticAtom(
                        text=remaining_text.strip(),
                        semantic_type=atom.semantic_type,
                        start_pos=atom.start_pos + current_pos,
                        end_pos=atom.end_pos,
                        trigger_words=[]
                    ))
            
            # 递归处理仍然过长的子原子
            final_atoms = []
            for sub_atom in sub_atoms:
                if len(sub_atom.text) > max_length:
                    final_atoms.extend(self.split_long_atom(sub_atom, max_length))
                else:
                    final_atoms.append(sub_atom)
            
            return final_atoms
        
        # 如果没有编号，按标点符号切分
        punctuation_pattern = r'[，。；：]\s*'
        splits = re.split(punctuation_pattern, atom.text)
        
        if len(splits) > 1:
            sub_atoms = []
            current_text = ""
            current_pos = 0
            
            for split in splits:
                if len(current_text + split) <= max_length:
                    current_text += split
                else:
                    if current_text.strip():
                        sub_atoms.append(SemanticAtom(
                            text=current_text.strip(),
                            semantic_type=atom.semantic_type,
                            start_pos=atom.start_pos + current_pos,
                            end_pos=atom.start_pos + current_pos + len(current_text),
                            trigger_words=atom.trigger_words.copy() if len(sub_atoms) == 0 else []
                        ))
                    current_pos += len(current_text)
                    current_text = split
            
            if current_text.strip():
                sub_atoms.append(SemanticAtom(
                    text=current_text.strip(),
                    semantic_type=atom.semantic_type,
                    start_pos=atom.start_pos + current_pos,
                    end_pos=atom.end_pos,
                    trigger_words=[]
                ))
            
            return sub_atoms if sub_atoms else [atom]
        
        # 如果都无法切分，按固定长度切分（保留语义类型）
        sub_atoms = []
        for i in range(0, len(atom.text), max_length):
            sub_text = atom.text[i:i + max_length]
            sub_atoms.append(SemanticAtom(
                text=sub_text,
                semantic_type=atom.semantic_type,
                start_pos=atom.start_pos + i,
                end_pos=min(atom.start_pos + i + max_length, atom.end_pos),
                trigger_words=atom.trigger_words.copy() if i == 0 else []
            ))
        
        return sub_atoms


# ============================================================================
# 术语表相关类和函数
# ============================================================================

class InsuranceTerminology:
    """
    保险业务专业术语表
    """
    
    def __init__(self, terminology_file: Optional[Path] = None):
        """
        初始化术语表
        
        Args:
            terminology_file: 术语表文件路径（JSON格式），如果为None则使用默认术语表
        """
        if terminology_file and Path(terminology_file).exists():
            self.load_from_file(terminology_file)
        else:
            self.load_default_terminology()
    
    def load_default_terminology(self):
        """加载默认术语表"""
        # 格式：{规范术语: [变体1, 变体2, ...]}
        self.terminology = {
            "保险金": ["保险金", "保险金额", "保险给付", "保险赔付", "保险赔偿", "保险赔款"],
            "保险责任": ["保险责任", "保障责任", "保障范围", "保险保障", "保险范围"],
            "保险费": ["保险费", "保费", "保险费用", "保险费率"],
            "保险期间": ["保险期间", "保障期间", "保险期限", "保障期限", "保险有效期"],
            "等待期": ["等待期", "观察期", "免责期", "等待期间"],
            "犹豫期": ["犹豫期", "冷静期", "撤销期", "退保期"],
            "免赔额": ["免赔额", "免赔", "免赔金额", "免赔额度"],
            "保险金额": ["保险金额", "保额", "保险额度", "保障金额"],
            "现金价值": ["现金价值", "退保价值", "保单价值", "解约金"],
            "受益人": ["受益人", "保险受益人", "受益方"],
            "被保险人": ["被保险人", "被保人", "受保人"],
            "投保人": ["投保人", "要保人", "保单持有人"],
            "保险人": ["保险人", "保险公司", "承保公司", "承保方"],
            "意外伤害": ["意外伤害", "意外事故", "意外", "意外事件"],
            "疾病": ["疾病", "疾病状态", "患病", "罹患疾病"],
            "身故": ["身故", "死亡", "身死", "去世"],
            "伤残": ["伤残", "残疾", "失能", "丧失劳动能力"],
            "重大疾病": ["重大疾病", "重疾", "重大疾病保险", "重疾险"],
            "轻症": ["轻症", "轻度疾病", "轻症疾病"],
            "中症": ["中症", "中度疾病", "中症疾病"],
            "理赔": ["理赔", "索赔", "申请理赔", "申请索赔"],
            "给付": ["给付", "支付", "赔付", "赔偿", "支付保险金"],
            "免责": ["免责", "除外", "不承担", "不予赔付", "免除责任"],
            "续保": ["续保", "续期", "续保保费", "续保申请"],
            "退保": ["退保", "解除合同", "终止合同", "取消保险"],
            "保单": ["保单", "保险合同", "保险单", "保险凭证"],
            "条款": ["条款", "保险条款", "合同条款", "约定条款"],
        }
        
        # 构建反向索引：变体 -> 规范术语
        self.variant_to_standard = {}
        for standard_term, variants in self.terminology.items():
            for variant in variants:
                self.variant_to_standard[variant] = standard_term
    
    def load_from_file(self, file_path: Path):
        """
        从文件加载术语表
        
        Args:
            file_path: JSON文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.terminology = data
            # 构建反向索引
            self.variant_to_standard = {}
            for standard_term, variants in self.terminology.items():
                for variant in variants:
                    self.variant_to_standard[variant] = standard_term
    
    def save_to_file(self, file_path: Path):
        """
        保存术语表到文件
        
        Args:
            file_path: JSON文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.terminology, f, ensure_ascii=False, indent=2)
    
    def extract_terms(self, text: str) -> Set[str]:
        """
        从文本中提取匹配的规范术语
        
        Args:
            text: 文本内容
            
        Returns:
            匹配到的规范术语集合
        """
        found_terms = set()
        
        # 按长度降序排序变体，优先匹配长变体
        sorted_variants = sorted(
            self.variant_to_standard.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        
        for variant, standard_term in sorted_variants:
            # 对于中文，直接搜索（因为中文没有明确的词边界）
            if variant in text:
                found_terms.add(standard_term)
        
        return found_terms
    
    def match_query_terms(self, query: str) -> Set[str]:
        """
        匹配用户查询中的术语
        
        Args:
            query: 用户查询文本
            
        Returns:
            匹配到的规范术语集合
        """
        return self.extract_terms(query)
    
    def get_all_standard_terms(self) -> List[str]:
        """
        获取所有规范术语列表
        
        Returns:
            规范术语列表
        """
        return list(self.terminology.keys())
    
    def add_term(self, standard_term: str, variants: List[str]):
        """
        添加新术语
        
        Args:
            standard_term: 规范术语
            variants: 变体列表
        """
        self.terminology[standard_term] = variants
        for variant in variants:
            self.variant_to_standard[variant] = standard_term
    
    def remove_term(self, standard_term: str):
        """
        删除术语
        
        Args:
            standard_term: 规范术语
        """
        if standard_term in self.terminology:
            variants = self.terminology.pop(standard_term)
            for variant in variants:
                if variant in self.variant_to_standard:
                    del self.variant_to_standard[variant]


# ============================================================================
# Chunk相关数据结构
# ============================================================================


@dataclass
class ChunkMetadata:
    """Chunk元数据结构"""
    chunk_id: str
    chunk_type: str  # paragraph, table, list, mixed
    section_path: List[str]  # 标题层级路径
    heading_level: int  # 当前所属标题级别
    char_count: int
    image_refs: List[str]  # 图片引用路径
    source_file: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    has_table: bool = False
    has_list: bool = False
    skip_embedding: bool = False  # 整个chunk是否跳过embedding
    skip_sentences: Optional[List[int]] = None  # 跳过embedding的句子索引列表（相对于chunk内的句子）
    # 新增字段：语义切割相关
    semantic_type: Optional[str] = None  # 语义类型：'给付', '免责', '条件', '定义', '其他'
    clause_number: Optional[str] = None  # 条款编号
    is_core_section: bool = False  # 是否属于核心条款区
    trigger_words: Optional[List[str]] = None  # 语义触发词列表
    # 新增字段：术语相关
    key_terms: Optional[List[str]] = None  # 规范术语列表


@dataclass
class Chunk:
    """统一的Chunk结构"""
    chunk_id: str
    text: str  # 原始文本（包含所有句子，包括跳过embedding的）
    metadata: ChunkMetadata
    sentence_infos: Optional[List[SentenceInfo]] = None  # 句子信息列表（用于标记跳过embedding）

    def get_embedding_text(self) -> str:
        """
        获取用于embedding的文本（排除跳过embedding的句子）
        
        Returns:
            用于embedding的文本
        """
        if self.metadata.skip_embedding:
            return ""  # 整个chunk跳过embedding
        
        if not self.sentence_infos:
            return self.text  # 没有句子信息，返回原始文本
        
        # 过滤掉跳过embedding的句子
        embedding_sentences = [
            info.text for info in self.sentence_infos 
            if not info.skip_embedding
        ]
        
        return '\n'.join(embedding_sentences)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding_text": self.get_embedding_text(),  # 添加embedding文本
            "metadata": asdict(self.metadata)
        }
        # 添加句子信息（如果存在）
        if self.sentence_infos:
            result["sentence_infos"] = [
                {
                    "text": info.text,
                    "skip_embedding": info.skip_embedding,
                    "reason": info.reason
                }
                for info in self.sentence_infos
            ]
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chunk':
        """从字典格式创建Chunk对象"""
        # 重建metadata
        metadata_dict = data.get("metadata", {})
        metadata = ChunkMetadata(**metadata_dict)
        
        # 重建sentence_infos（如果存在）
        sentence_infos = None
        if "sentence_infos" in data:
            sentence_infos = [
                SentenceInfo(
                    text=info_dict["text"],
                    skip_embedding=info_dict.get("skip_embedding", False),
                    reason=info_dict.get("reason", "")
                )
                for info_dict in data["sentence_infos"]
            ]
        
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            metadata=metadata,
            sentence_infos=sentence_infos
        )


class SemanticChunker:
    """
    语义分块器 - 基于OCR产出的Markdown进行智能分块
    
    核心原则：
    1. 仅以OCR产出的Markdown作为输入源
    2. 最小必要的文本规范化
    3. 保留文档结构上下文（标题层级）
    4. 优先保证语义完整性
    5. 表格作为不可拆分的原子单元
    6. 识别并记录图片引用
    7. 统一的结构化输出格式
    """

    def __init__(
        self,
        target_chunk_size: int = 800,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        overlap_size: int = 100,
        enable_text_cleaning: bool = True,
        text_cleaner: Optional[TextCleaner] = None,
        save_cleaned_text: bool = True,
        cleaned_output_dir: Optional[Path] = None,
        enable_semantic_splitting: bool = True,
        semantic_splitter: Optional[SemanticSplitter] = None,
        enable_terminology: bool = True,
        terminology: Optional[InsuranceTerminology] = None,
        terminology_file: Optional[Path] = None
    ):
        """
        初始化分块器
        
        Args:
            target_chunk_size: 目标chunk大小（字符数）
            max_chunk_size: 最大chunk大小
            min_chunk_size: 最小chunk大小
            overlap_size: chunk之间的重叠大小
            enable_text_cleaning: 是否启用文本清洗
            text_cleaner: 文本清洗器实例（如果为None且enable_text_cleaning=True，则创建默认实例）
            save_cleaned_text: 是否保存清洗后的文本到文件
            cleaned_output_dir: 清洗后文本的输出目录（默认：data/cleaned/，保持与processed相同的目录结构）
            enable_semantic_splitting: 是否启用语义切割
            semantic_splitter: 语义切割器实例（如果为None且enable_semantic_splitting=True，则创建默认实例）
            enable_terminology: 是否启用术语提取
            terminology: 术语表实例（如果为None且enable_terminology=True，则创建默认实例）
            terminology_file: 术语表文件路径（可选）
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.enable_text_cleaning = enable_text_cleaning
        self.save_cleaned_text = save_cleaned_text
        self.enable_semantic_splitting = enable_semantic_splitting
        self.enable_terminology = enable_terminology
        
        if enable_text_cleaning:
            self.text_cleaner = text_cleaner or TextCleaner()
        else:
            self.text_cleaner = None
        
        if enable_semantic_splitting:
            self.semantic_splitter = semantic_splitter or SemanticSplitter()
        else:
            self.semantic_splitter = None
        
        if enable_terminology:
            self.terminology = terminology or InsuranceTerminology(terminology_file)
        else:
            self.terminology = None
        
        # 设置清洗后文本的输出目录
        if cleaned_output_dir is None:
            # 默认使用项目根目录下的 data/cleaned/
            project_root = Path(__file__).parent.parent
            self.cleaned_output_dir = project_root / "data" / "cleaned"
        else:
            self.cleaned_output_dir = Path(cleaned_output_dir)

    def normalize_text(self, text: str) -> str:
        """
        最小必要的文本规范化
        
        - 压缩连续空行（超过2个空行压缩为2个）
        - 移除行尾空格
        - 保留其他格式和结构
        """
        # 移除行尾空格
        lines = [line.rstrip() for line in text.split('\n')]
        
        # 压缩连续空行
        normalized_lines = []
        empty_count = 0
        
        for line in lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    normalized_lines.append(line)
            else:
                empty_count = 0
                normalized_lines.append(line)
        
        return '\n'.join(normalized_lines)

    def extract_heading_info(self, line: str) -> Optional[Tuple[int, str]]:
        """
        提取标题信息
        
        Returns:
            (level, title) 或 None
        """
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            return (level, title)
        return None

    def is_table_start(self, line: str) -> bool:
        """判断是否是表格开始"""
        return line.strip().startswith('<table')

    def is_table_end(self, line: str) -> bool:
        """判断是否是表格结束"""
        return '</table>' in line

    def extract_image_refs(self, text: str) -> List[str]:
        """
        提取图片引用
        
        支持格式：
        - ![](path/to/image.jpg)
        - ![alt text](path/to/image.jpg)
        """
        pattern = r'!\[.*?\]\((.*?)\)'
        return re.findall(pattern, text)

    def is_list_item(self, line: str) -> bool:
        """判断是否是列表项"""
        stripped = line.strip()
        # 无序列表
        if re.match(r'^[-*+]\s+', stripped):
            return True
        # 有序列表
        if re.match(r'^\d+\.\s+', stripped):
            return True
        # 特殊符号列表（如 $\bullet$）
        if re.match(r'^\$\\[a-zA-Z]+\$\s+', stripped):
            return True
        return False

    def parse_markdown(self, md_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        解析Markdown文件，提取结构化内容块
        
        Returns:
            (List of content blocks with metadata, cleaned_text)
            cleaned_text: 清洗后的文本（如果启用了清洗），否则为None
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        cleaned_text = None
        
        # 文本清洗（如果启用）
        if self.enable_text_cleaning and self.text_cleaner:
            content = self.text_cleaner.basic_clean(original_content)
            cleaned_text = content  # 保存清洗后的文本
        else:
            content = original_content
        
        # 文本规范化
        content = self.normalize_text(content)
        lines = content.split('\n')
        
        blocks = []
        current_section_path = []
        current_heading_level = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 处理标题
            heading_info = self.extract_heading_info(line)
            if heading_info:
                level, title = heading_info
                # 更新section path
                current_section_path = current_section_path[:level-1] + [title]
                current_heading_level = level
                
                blocks.append({
                    'type': 'heading',
                    'content': line,
                    'level': level,
                    'title': title,
                    'section_path': current_section_path.copy(),
                    'line_num': i
                })
                i += 1
                continue
            
            # 处理表格（作为原子单元）
            if self.is_table_start(line):
                table_lines = [line]
                i += 1
                while i < len(lines) and not self.is_table_end(lines[i-1]):
                    table_lines.append(lines[i])
                    i += 1
                
                table_content = '\n'.join(table_lines)
                blocks.append({
                    'type': 'table',
                    'content': table_content,
                    'section_path': current_section_path.copy(),
                    'heading_level': current_heading_level,
                    'line_num': i - len(table_lines)
                })
                continue
            
            # 处理列表
            if self.is_list_item(line):
                list_lines = [line]
                i += 1
                # 收集连续的列表项
                while i < len(lines):
                    next_line = lines[i]
                    # 空行或新的列表项继续
                    if next_line.strip() == '' or self.is_list_item(next_line):
                        list_lines.append(next_line)
                        i += 1
                    # 缩进的内容（列表项的延续）
                    elif next_line.startswith('  ') or next_line.startswith('\t'):
                        list_lines.append(next_line)
                        i += 1
                    else:
                        break
                
                list_content = '\n'.join(list_lines)
                blocks.append({
                    'type': 'list',
                    'content': list_content,
                    'section_path': current_section_path.copy(),
                    'heading_level': current_heading_level,
                    'line_num': i - len(list_lines)
                })
                continue
            
            # 处理普通段落
            if line.strip():
                para_lines = [line]
                i += 1
                # 收集连续的非空行（直到遇到空行、标题、表格或列表）
                while i < len(lines):
                    next_line = lines[i]
                    if (next_line.strip() == '' or 
                        self.extract_heading_info(next_line) or
                        self.is_table_start(next_line) or
                        self.is_list_item(next_line)):
                        break
                    para_lines.append(next_line)
                    i += 1
                
                para_content = '\n'.join(para_lines)
                blocks.append({
                    'type': 'paragraph',
                    'content': para_content,
                    'section_path': current_section_path.copy(),
                    'heading_level': current_heading_level,
                    'line_num': i - len(para_lines)
                })
                continue
            
            # 跳过空行
            i += 1
        
        return blocks, cleaned_text

    def _perform_sentence_splitting_and_denoising(
        self,
        blocks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        执行句级拆分和语义降噪（在chunking之前）
        
        Args:
            blocks: 内容块列表
            
        Returns:
            global_sentence_map: 句子到SentenceInfo的映射字典
        """
        global_sentence_map = {}  # sentence_text -> SentenceInfo
        
        if self.enable_text_cleaning and self.text_cleaner:
            # 收集所有文本块的内容和章节路径
            all_texts = []
            all_section_paths = []
            
            for block in blocks:
                if block['type'] != 'heading':  # 跳过标题
                    all_texts.append(block['content'])
                    all_section_paths.append(block.get('section_path', []))
            
            # 合并所有文本，进行全局语义降噪
            # 为每个句子分配章节路径（简化处理：使用对应文本块的章节路径）
            all_sentences = []
            section_paths_for_sentences = []
            for i, text in enumerate(all_texts):
                sentences_in_text = self.text_cleaner.split_into_sentences(text)
                for sentence in sentences_in_text:
                    all_sentences.append(sentence)
                    section_paths_for_sentences.append(all_section_paths[i])
            
            # 全局语义降噪
            global_sentence_infos = self.text_cleaner.semantic_denoise(
                all_sentences,
                section_paths_for_sentences
            )
            
            # 建立句子到SentenceInfo的映射（使用句子文本作为key）
            for sentence_info in global_sentence_infos:
                # 简化：使用句子文本作为key（实际应该考虑章节路径）
                key = sentence_info.text.strip()
                if key not in global_sentence_map:
                    global_sentence_map[key] = sentence_info
        
        return global_sentence_map

    def create_chunks_from_blocks(
        self,
        blocks: List[Dict[str, Any]],
        source_file: str,
        global_sentence_map: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        从内容块创建chunks，保证语义完整性
        
        策略：
        1. 表格作为独立chunk，不拆分
        2. 列表尽量保持完整
        3. 段落在超过max_size时才拆分
        4. 相邻小块可以合并（在同一section下）
        5. 使用预先计算的语义降噪结果（如果提供）
        
        Args:
            blocks: 内容块列表
            source_file: 源文件路径
            global_sentence_map: 预先计算的句子到SentenceInfo的映射（可选）
        """
        # 如果没有提供global_sentence_map，则自己计算（向后兼容）
        if global_sentence_map is None:
            global_sentence_map = self._perform_sentence_splitting_and_denoising(blocks)
        
        chunks = []
        buffer = []
        buffer_size = 0
        current_section_path = []
        current_heading_level = 0
        accumulated_images = []
        start_line = None
        
        def flush_buffer():
            """将buffer中的内容创建为chunk"""
            nonlocal buffer, buffer_size, accumulated_images, start_line
            
            if not buffer:
                return
            
            # 合并buffer内容
            text_parts = []
            has_table = False
            has_list = False
            end_line = None
            
            for block in buffer:
                text_parts.append(block['content'])
                if block['type'] == 'table':
                    has_table = True
                if block['type'] == 'list':
                    has_list = True
                if 'line_num' in block:
                    end_line = block['line_num']
            
            chunk_text = '\n\n'.join(text_parts)
            
            # 提取图片引用
            image_refs = self.extract_image_refs(chunk_text)
            image_refs.extend(accumulated_images)
            
            # 语义降噪：识别应该跳过embedding的句子
            sentence_infos = None
            skip_embedding = False
            
            if self.enable_text_cleaning and self.text_cleaner:
                # 对文本进行句级拆分
                sentences = self.text_cleaner.split_into_sentences(chunk_text)
                
                # 从全局映射中获取句子信息，如果没有则创建新的
                sentence_infos = []
                for sentence in sentences:
                    sentence_stripped = sentence.strip()
                    if sentence_stripped in global_sentence_map:
                        # 使用全局识别的信息
                        sentence_infos.append(global_sentence_map[sentence_stripped])
                    else:
                        # 新句子，检查是否是兜底话术
                        is_boilerplate = self.text_cleaner.is_boilerplate_sentence(sentence_stripped)
                        from app.text_cleaner import SentenceInfo
                        sentence_infos.append(SentenceInfo(
                            text=sentence,
                            skip_embedding=is_boilerplate,
                            reason="boilerplate" if is_boilerplate else ""
                        ))
                
                # 如果所有句子都跳过embedding，标记整个chunk跳过
                if sentence_infos and all(info.skip_embedding for info in sentence_infos):
                    skip_embedding = True
            
            # Step1: 语义切割（如果启用且不是表格）
            semantic_atoms = []
            if self.enable_semantic_splitting and self.semantic_splitter and not has_table:
                # 对文本进行语义切割
                atoms = self.semantic_splitter.split_into_semantic_atoms(chunk_text)
                
                # Step2: 长度控制 - 如果单个语义原子过长，进一步切分
                for atom in atoms:
                    if len(atom.text) > self.max_chunk_size:
                        sub_atoms = self.semantic_splitter.split_long_atom(atom, self.max_chunk_size)
                        semantic_atoms.extend(sub_atoms)
                    else:
                        semantic_atoms.append(atom)
            else:
                # 如果没有启用语义切割，将整个文本作为一个原子
                semantic_atoms = [SemanticAtom(
                    text=chunk_text,
                    semantic_type='其他',
                    start_pos=0,
                    end_pos=len(chunk_text),
                    trigger_words=[]
                )]
            
            # 为每个语义原子创建chunk
            for atom in semantic_atoms:
                atom_text = atom.text
                
                # 重新处理语义降噪（针对单个原子）
                atom_sentence_infos = None
                atom_skip_embedding = False
                
                if self.enable_text_cleaning and self.text_cleaner:
                    # 对原子文本进行句级拆分
                    sentences = self.text_cleaner.split_into_sentences(atom_text)
                    
                    # 从全局映射中获取句子信息
                    atom_sentence_infos = []
                    for sentence in sentences:
                        sentence_stripped = sentence.strip()
                        if sentence_stripped in global_sentence_map:
                            atom_sentence_infos.append(global_sentence_map[sentence_stripped])
                        else:
                            is_boilerplate = self.text_cleaner.is_boilerplate_sentence(sentence_stripped)
                            from app.text_cleaner import SentenceInfo
                            atom_sentence_infos.append(SentenceInfo(
                                text=sentence,
                                skip_embedding=is_boilerplate,
                                reason="boilerplate" if is_boilerplate else ""
                            ))
                    
                    if atom_sentence_infos and all(info.skip_embedding for info in atom_sentence_infos):
                        atom_skip_embedding = True
                else:
                    atom_sentence_infos = sentence_infos
                    atom_skip_embedding = skip_embedding
                
                # Step3: 术语提取
                key_terms = []
                if self.enable_terminology and self.terminology:
                    key_terms = list(self.terminology.extract_terms(atom_text))
                
                # 判断是否在核心条款区
                is_core_section = False
                if self.text_cleaner:
                    is_core_section = self.text_cleaner.is_core_section(current_section_path)
                
                # 提取条款编号（如果有）
                clause_number = None
                clause_pattern = r'第\s*[一二三四五六七八九十\d]+\s*[条条款项]'
                clause_match = re.search(clause_pattern, atom_text)
                if clause_match:
                    clause_number = clause_match.group(0)
                
                # 创建chunk
                chunk_id = str(uuid.uuid4())
                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    chunk_type='table' if has_table else ('list' if has_list else 'paragraph'),
                    section_path=current_section_path.copy(),
                    heading_level=current_heading_level,
                    char_count=len(atom_text),
                    image_refs=list(set(image_refs)),  # 去重
                    source_file=source_file,
                    start_line=start_line,
                    end_line=end_line,
                    has_table=has_table,
                    has_list=has_list,
                    skip_embedding=atom_skip_embedding,
                    skip_sentences=None,  # 这个字段暂时不用，信息在sentence_infos中
                    # 新增字段
                    semantic_type=atom.semantic_type,
                    clause_number=clause_number,
                    is_core_section=is_core_section,
                    trigger_words=atom.trigger_words,
                    key_terms=key_terms if key_terms else None
                )
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    text=atom_text,
                    metadata=metadata,
                    sentence_infos=atom_sentence_infos
                )
                chunks.append(chunk)
            
            # 重置buffer
            buffer = []
            buffer_size = 0
            accumulated_images = []
            start_line = None
        
        for block in blocks:
            block_type = block['type']
            block_content = block['content']
            block_size = len(block_content)
            
            # 更新section context
            if 'section_path' in block:
                current_section_path = block['section_path']
            if 'heading_level' in block:
                current_heading_level = block['heading_level']
            
            # 标题不加入chunk，但更新上下文
            if block_type == 'heading':
                # 如果buffer有内容，先flush
                if buffer:
                    flush_buffer()
                continue
            
            # 表格作为独立chunk（不参与语义切割，但添加metadata）
            if block_type == 'table':
                # 先flush现有buffer
                if buffer:
                    flush_buffer()
                
                # 表格单独成chunk
                buffer = [block]
                buffer_size = block_size
                start_line = block.get('line_num')
                flush_buffer()
                continue
            
            # 如果当前块超过max_size，需要特殊处理
            if block_size > self.max_chunk_size:
                # 先flush现有buffer
                if buffer:
                    flush_buffer()
                
                # 对于超大块，尝试按段落拆分
                if block_type == 'paragraph':
                    sentences = re.split(r'([。！？\n])', block_content)
                    temp_buffer = []
                    temp_size = 0
                    
                    for i in range(0, len(sentences), 2):
                        sentence = sentences[i]
                        if i + 1 < len(sentences):
                            sentence += sentences[i + 1]
                        
                        if temp_size + len(sentence) > self.max_chunk_size and temp_buffer:
                            # 创建chunk
                            buffer = [{
                                'type': 'paragraph',
                                'content': ''.join(temp_buffer),
                                'section_path': block['section_path'],
                                'heading_level': block['heading_level'],
                                'line_num': block.get('line_num')
                            }]
                            buffer_size = temp_size
                            start_line = block.get('line_num')
                            flush_buffer()
                            temp_buffer = []
                            temp_size = 0
                        
                        temp_buffer.append(sentence)
                        temp_size += len(sentence)
                    
                    if temp_buffer:
                        buffer = [{
                            'type': 'paragraph',
                            'content': ''.join(temp_buffer),
                            'section_path': block['section_path'],
                            'heading_level': block['heading_level'],
                            'line_num': block.get('line_num')
                        }]
                        buffer_size = temp_size
                        start_line = block.get('line_num')
                        flush_buffer()
                else:
                    # 列表等其他类型，直接作为一个chunk
                    buffer = [block]
                    buffer_size = block_size
                    start_line = block.get('line_num')
                    flush_buffer()
                continue
            
            # 检查是否需要flush buffer
            if buffer_size + block_size > self.target_chunk_size:
                # 如果加入当前块会超过target_size
                if buffer_size >= self.min_chunk_size:
                    # buffer已经足够大，flush
                    flush_buffer()
                elif buffer_size + block_size > self.max_chunk_size:
                    # 即使buffer很小，但加入后会超过max_size，也要flush
                    flush_buffer()
            
            # 将block加入buffer
            if not buffer:
                start_line = block.get('line_num')
            buffer.append(block)
            buffer_size += block_size
        
        # 处理剩余buffer
        if buffer:
            flush_buffer()
        
        return chunks

    def _save_cleaned_text(self, md_path: Path, cleaned_text: str) -> Optional[Path]:
        """
        保存清洗后的文本到文件
        
        Args:
            md_path: 原始Markdown文件路径
            cleaned_text: 清洗后的文本
            
        Returns:
            保存的文件路径，如果未保存则返回None
        """
        if not self.save_cleaned_text or not cleaned_text:
            return None
        
        try:
            # 计算相对于processed目录的相对路径
            md_path = Path(md_path).resolve()
            
            # 尝试找到processed目录
            processed_dirs = [
                Path(__file__).parent.parent / "data" / "processed",
                md_path.parent.parent.parent / "processed",  # 假设结构是 processed/xxx/xxx.md
            ]
            
            relative_path = None
            for processed_dir in processed_dirs:
                processed_dir = processed_dir.resolve()
                try:
                    relative_path = md_path.relative_to(processed_dir)
                    break
                except ValueError:
                    continue
            
            # 如果找不到processed目录，使用文件名
            if relative_path is None:
                relative_path = Path(md_path.name)
            
            # 构建输出路径：data/cleaned/相对路径
            output_path = self.cleaned_output_dir / relative_path
            
            # 创建目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            return output_path
        except Exception as e:
            # 如果保存失败，打印警告但不中断流程
            print(f"⚠️  警告: 保存清洗后文本失败: {e}")
            return None

    def chunk_markdown_file(self, md_path: Path) -> List[Chunk]:
        """
        对单个Markdown文件进行分块
        
        Args:
            md_path: Markdown文件路径
            
        Returns:
            Chunk列表
        """
        # 解析Markdown结构
        blocks, cleaned_text = self.parse_markdown(md_path)
        
        # 保存清洗后的文本（如果启用）
        if cleaned_text:
            saved_path = self._save_cleaned_text(md_path, cleaned_text)
            if saved_path:
                print(f"✓ 清洗后的文本已保存到: {saved_path}")
        
        # 在chunking之前执行句级拆分和语义降噪
        global_sentence_map = self._perform_sentence_splitting_and_denoising(blocks)
        
        # 创建chunks（传入预先计算的语义降噪结果）
        chunks = self.create_chunks_from_blocks(
            blocks,
            source_file=str(md_path),
            global_sentence_map=global_sentence_map
        )
        
        return chunks

    def chunk_directory(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        pattern: str = "**/*.md"
    ) -> Dict[str, List[Chunk]]:
        """
        批量处理目录下的所有Markdown文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录（可选，如果提供则保存JSON）
            pattern: 文件匹配模式
            
        Returns:
            {文件路径: chunks列表} 的字典
        """
        input_dir = Path(input_dir)
        results = {}
        
        # 查找所有匹配的Markdown文件
        md_files = list(input_dir.glob(pattern))
        
        print(f"找到 {len(md_files)} 个Markdown文件")
        
        for md_file in md_files:
            print(f"处理: {md_file}")
            try:
                chunks = self.chunk_markdown_file(md_file)
                results[str(md_file)] = chunks
                print(f"  生成 {len(chunks)} 个chunks")
                
                # 如果提供了输出目录，保存JSON文件
                if output_dir:
                    # 计算相对于input_dir的相对路径
                    try:
                        relative_path = md_file.relative_to(input_dir)
                        # 将.md扩展名改为_chunks.json
                        output_file = output_dir / relative_path.with_suffix('_chunks.json')
                        # 确保输出目录存在
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 保存JSON文件
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(
                                [chunk.to_dict() for chunk in chunks],
                                f,
                                ensure_ascii=False,
                                indent=2
                            )
                        print(f"  已保存到: {output_file}")
                    except ValueError:
                        # 如果无法计算相对路径，使用文件名
                        output_file = output_dir / f"{md_file.stem}_chunks.json"
                        output_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(
                                [chunk.to_dict() for chunk in chunks],
                                f,
                                ensure_ascii=False,
                                indent=2
                            )
                        print(f"  已保存到: {output_file}")
                    
            except Exception as e:
                print(f"  错误: {e}")
                continue
        
        return results

    def get_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        获取chunks的统计信息
        
        Args:
            chunks: Chunk列表
            
        Returns:
            统计信息字典
        """
        if not chunks:
            return {}
        
        char_counts = [chunk.metadata.char_count for chunk in chunks]
        chunk_types = [chunk.metadata.chunk_type for chunk in chunks]
        
        type_counts = {}
        for ct in chunk_types:
            type_counts[ct] = type_counts.get(ct, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(char_counts) / len(char_counts),
            'min_chunk_size': min(char_counts),
            'max_chunk_size': max(char_counts),
            'chunk_type_distribution': type_counts,
            'chunks_with_images': sum(1 for c in chunks if c.metadata.image_refs),
            'chunks_with_tables': sum(1 for c in chunks if c.metadata.has_table),
            'chunks_with_lists': sum(1 for c in chunks if c.metadata.has_list)
        }


# 便捷函数
def chunk_file(md_path: str, **kwargs) -> List[Chunk]:
    """
    便捷函数：对单个文件进行分块
    
    Args:
        md_path: Markdown文件路径
        **kwargs: SemanticChunker的初始化参数
        
    Returns:
        Chunk列表
    """
    chunker = SemanticChunker(**kwargs)
    return chunker.chunk_markdown_file(Path(md_path))


def chunk_directory(input_dir: str, output_dir: str = None, **kwargs) -> Dict[str, List[Chunk]]:
    """
    便捷函数：批量处理目录
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        **kwargs: SemanticChunker的初始化参数
        
    Returns:
        {文件路径: chunks列表} 的字典
    """
    chunker = SemanticChunker(**kwargs)
    return chunker.chunk_directory(Path(input_dir), Path(output_dir) if output_dir else None)


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        
        chunker = SemanticChunker(
            target_chunk_size=800,
            max_chunk_size=1500,
            min_chunk_size=200
        )
        
        if Path(input_path).is_file():
            # 处理单个文件
            chunks = chunker.chunk_markdown_file(Path(input_path))
            print(f"\n生成了 {len(chunks)} 个chunks")
            print("\n统计信息:")
            stats = chunker.get_statistics(chunks)
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        [chunk.to_dict() for chunk in chunks],
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                print(f"\n结果已保存到: {output_path}")
        else:
            # 处理目录
            results = chunker.chunk_directory(
                Path(input_path),
                Path(output_path) if output_path else None
            )
            print(f"\n总共处理了 {len(results)} 个文件")
    else:
        print("用法: python chunker.py <输入文件/目录> [输出文件/目录]")
