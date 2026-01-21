# app/text_cleaner.py

import re
from typing import List, Dict, Tuple, Set
from collections import Counter
from dataclasses import dataclass


@dataclass
class SentenceInfo:
    """句子信息，用于标记是否跳过embedding"""
    text: str
    skip_embedding: bool = False
    reason: str = ""  # 跳过原因：'boilerplate'（兜底话术）或 'repetitive'（重复话术）


class TextCleaner:
    """
    文本清洗器 - 专门处理OCR产出的Markdown文本
    
    功能：
    1. 基础清洗：去页眉页脚、页码、合并OCR断句
    2. 句级拆分：按标点拆分成最小单元
    3. 语义降噪：识别兜底话术和重复话术，标记跳过embedding
    """
    
    def __init__(
        self,
        min_repeat_length: int = 20,  # 页眉页脚最小重复长度
        repeat_threshold: int = 3,  # 重复话术出现次数阈值
        core_section_keywords: List[str] = None  # 核心条款区关键词
    ):
        """
        初始化文本清洗器
        
        Args:
            min_repeat_length: 页眉页脚识别的最小重复长度
            repeat_threshold: 重复话术出现次数阈值（超过此次数标记为低信息）
            core_section_keywords: 核心条款区关键词列表
        """
        self.min_repeat_length = min_repeat_length
        self.repeat_threshold = repeat_threshold
        self.core_section_keywords = core_section_keywords or [
            "保险责任", "保险金", "理赔", "给付", "保障范围", 
            "保险金额", "保险费", "保险期间", "等待期"
        ]
        
        # 兜底话术关键词和正则模式
        self.boilerplate_patterns = [
            r"本合同未尽事宜",
            r"保险人保留最终解释权",
            r"本合同的解释权归.*?所有",
            r"其他未尽事宜.*?约定",
            r"本条款.*?解释权",
            r"最终解释权.*?保险人",
            r"保险人.*?有权.*?解释",
            r"本附加合同.*?未尽事宜",
            r"其他.*?以.*?为准",
        ]
        
        # 编译正则表达式
        self.boilerplate_regexes = [re.compile(pattern) for pattern in self.boilerplate_patterns]
    
    def remove_page_numbers(self, text: str) -> str:
        """
        去除页码和目录重复
        
        识别模式：
        - 第xx页
        - 第 x 页
        - Page xx
        - 页码：xx
        """
        # 匹配页码模式
        patterns = [
            r'第\s*\d+\s*页',
            r'第\s*[一二三四五六七八九十百千万]+\s*页',
            r'Page\s*\d+',
            r'页码[：:]\s*\d+',
            r'^\s*\d+\s*$',  # 单独的数字行（可能是页码）
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 检查是否是页码行
            is_page_number = False
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_page_number = True
                    break
            
            # 如果不是页码行，保留
            if not is_page_number:
                cleaned_lines.append(line)
            # 如果是页码行，但行长度较长（可能是正常内容），也保留
            elif len(line.strip()) > 10:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_header_footer(self, text: str) -> str:
        """
        去除页眉页脚（找大段重复）
        
        策略：
        1. 统计每行的出现频率
        2. 如果某行出现多次且长度较长，可能是页眉页脚
        3. 检查是否在文档开头和结尾重复出现
        """
        lines = text.split('\n')
        if len(lines) < 10:  # 文档太短，不处理
            return text
        
        # 统计每行出现次数
        line_counter = Counter(lines)
        
        # 识别可能的页眉页脚
        header_footer_lines = set()
        
        # 检查开头和结尾的重复行
        header_candidates = lines[:min(5, len(lines) // 10)]  # 前5行或前10%
        footer_candidates = lines[-min(5, len(lines) // 10):]  # 后5行或后10%
        
        for line in header_candidates + footer_candidates:
            stripped = line.strip()
            # 如果这行出现多次，可能是页眉页脚
            # 对于在开头/结尾出现的行，降低长度要求（至少5个字符）
            min_length = max(5, self.min_repeat_length // 2) if line_counter[line] >= 3 else self.min_repeat_length
            if (line_counter[line] >= 2 and 
                len(stripped) >= min_length and
                stripped):  # 非空行
                header_footer_lines.add(line)
        
        # 过滤掉页眉页脚行
        # 被标记为页眉页脚的行应删除，除非只在中间出现一次且出现次数<2
        cleaned_lines = []
        header_footer_set = set(header_candidates + footer_candidates)
        middle_start = min(5, len(lines) // 10)
        middle_end = len(lines) - min(5, len(lines) // 10)
        
        # 统计每行在中间区域出现的次数
        middle_line_counter = Counter()
        for i in range(middle_start, middle_end):
            if i < len(lines):
                middle_line_counter[lines[i]] += 1
        
        for i, line in enumerate(lines):
            if line not in header_footer_lines:
                # 不在页眉页脚候选列表中，保留
                cleaned_lines.append(line)
            else:
                # 在页眉页脚候选列表中
                total_count = line_counter[line]
                middle_count = middle_line_counter[line]
                is_in_middle = middle_start <= i < middle_end
                
                # 只在中间出现一次且出现次数<2，保留
                # 条件：当前行在中间区域，且只在中间出现（middle_count == total_count），且总出现次数<2
                if is_in_middle and middle_count == total_count and total_count < 2:
                    cleaned_lines.append(line)
                # 否则删除（页眉页脚）
                # 不添加到cleaned_lines中，即删除
        
        return '\n'.join(cleaned_lines)
    
    def merge_ocr_breaks(self, text: str) -> str:
        """
        合并OCR断句
        
        逻辑：
        - 前一句没有标点（。！？；）
        - 后一句没有编号/title等（不以数字、字母、特殊符号开头）
        """
        # 先按行分割
        lines = text.split('\n')
        merged_lines = []
        
        i = 0
        while i < len(lines):
            current_line = lines[i].strip()
            
            # 如果当前行是空行或标题，直接保留
            if not current_line or current_line.startswith('#'):
                merged_lines.append(lines[i])
                i += 1
                continue
            
            # 检查当前行是否以标点结尾
            ends_with_punctuation = bool(re.search(r'[。！？；]$', current_line))
            
            # 如果当前行没有标点结尾，尝试合并下一行
            if not ends_with_punctuation and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                
                # 检查下一行是否是标题、列表、表格等
                is_special_line = (
                    next_line.startswith('#') or  # 标题
                    next_line.startswith('|') or  # 表格
                    next_line.startswith('<') or  # HTML标签
                    re.match(r'^[-*+]\s+', next_line) or  # 无序列表
                    re.match(r'^\d+\.\s+', next_line) or  # 有序列表
                    re.match(r'^[一二三四五六七八九十]+[、．.]', next_line) or  # 中文编号
                    not next_line  # 空行
                )
                
                # 如果下一行不是特殊行，可以合并
                if not is_special_line:
                    # 合并两行（用空格连接）
                    merged_line = current_line + ' ' + next_line
                    merged_lines.append(merged_line)
                    i += 2
                    continue
            
            # 否则保留当前行
            merged_lines.append(lines[i])
            i += 1
        
        return '\n'.join(merged_lines)
    
    def basic_clean(self, text: str, preserve_tables: bool = True, preserve_images: bool = True) -> str:
        """
        基础清洗
        
        Args:
            text: 原始文本
            preserve_tables: 是否保留表格
            preserve_images: 是否保留图片
        
        Returns:
            清洗后的文本
        """
        # 如果保留表格和图片，先提取它们
        table_blocks = []
        image_blocks = []
        
        if preserve_tables:
            # 提取表格块
            table_pattern = r'<table.*?</table>'
            table_matches = list(re.finditer(table_pattern, text, re.DOTALL))
            for i, match in enumerate(table_matches):
                placeholder = f"__TABLE_BLOCK_{i}__"
                table_blocks.append((placeholder, match.group(0)))
                text = text[:match.start()] + placeholder + text[match.end():]
        
        if preserve_images:
            # 提取图片块
            image_pattern = r'!\[.*?\]\(.*?\)'
            image_matches = list(re.finditer(image_pattern, text))
            for i, match in enumerate(image_matches):
                placeholder = f"__IMAGE_BLOCK_{i}__"
                image_blocks.append((placeholder, match.group(0)))
                text = text[:match.start()] + placeholder + text[match.end():]
        
        # 执行清洗
        text = self.remove_page_numbers(text)
        text = self.remove_header_footer(text)
        text = self.merge_ocr_breaks(text)
        
        # 恢复表格和图片
        for placeholder, content in table_blocks:
            text = text.replace(placeholder, content)
        
        for placeholder, content in image_blocks:
            text = text.replace(placeholder, content)
        
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        句级拆分：按标点拆分成最小单元
        
        支持的标点：。！？；\n（换行也作为句子分隔符）
        """
        # 先按换行分割
        paragraphs = text.split('\n')
        sentences = []
        
        for para in paragraphs:
            if not para.strip():
                sentences.append(para)  # 保留空行
                continue
            
            # 按标点分割
            # 使用正则表达式分割，保留分隔符
            parts = re.split(r'([。！？；])', para)
            
            current_sentence = ""
            for i, part in enumerate(parts):
                current_sentence += part
                # 如果遇到标点，结束当前句子
                if part in ['。', '！', '？', '；']:
                    if current_sentence.strip():
                        sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            # 处理最后剩余的文本
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
        
        return sentences
    
    def is_boilerplate_sentence(self, sentence: str) -> bool:
        """
        判断是否是兜底话术（规则触发）
        
        Returns:
            True if 是兜底话术
        """
        for regex in self.boilerplate_regexes:
            if regex.search(sentence):
                return True
        return False
    
    def is_core_section(self, section_path: List[str]) -> bool:
        """
        判断是否在核心条款区
        
        Args:
            section_path: 章节路径
        
        Returns:
            True if 在核心条款区
        """
        section_text = ' '.join(section_path)
        for keyword in self.core_section_keywords:
            if keyword in section_text:
                return True
        return False
    
    def identify_repetitive_sentences(
        self, 
        sentences: List[str],
        section_paths: List[List[str]] = None
    ) -> Set[int]:
        """
        识别重复话术（统计方法）
        
        Args:
            sentences: 句子列表
            section_paths: 每个句子对应的章节路径（可选）
        
        Returns:
            应该跳过embedding的句子索引集合
        """
        if section_paths is None:
            section_paths = [[] for _ in sentences]
        
        # 统计句子出现次数
        sentence_counter = Counter()
        sentence_indices = {}  # sentence -> [indices]
        
        for i, sentence in enumerate(sentences):
            normalized = sentence.strip()
            if normalized:
                sentence_counter[normalized] += 1
                if normalized not in sentence_indices:
                    sentence_indices[normalized] = []
                sentence_indices[normalized].append(i)
        
        # 找出重复次数超过阈值的句子
        skip_indices = set()
        
        for sentence, count in sentence_counter.items():
            if count >= self.repeat_threshold:
                # 如果重复次数超过阈值，标记所有出现位置为跳过
                indices = sentence_indices[sentence]
                for idx in indices:
                    # 检查是否在核心条款区
                    # 如果不在核心条款区，直接标记为跳过
                    if not self.is_core_section(section_paths[idx]):
                        skip_indices.add(idx)
                    # 即使在核心条款区，如果重复次数超过阈值，也标记为跳过
                    # 因为重复话术即使是核心条款，也会影响检索质量
                    else:
                        skip_indices.add(idx)
        
        return skip_indices
    
    def semantic_denoise(
        self,
        sentences: List[str],
        section_paths: List[List[str]] = None
    ) -> List[SentenceInfo]:
        """
        语义降噪：标记应该跳过embedding的句子
        
        Args:
            sentences: 句子列表
            section_paths: 每个句子对应的章节路径（可选）
        
        Returns:
            SentenceInfo列表，包含skip_embedding标记
        """
        if section_paths is None:
            section_paths = [[] for _ in sentences]
        
        sentence_infos = []
        
        # 第一步：识别兜底话术（规则触发）
        boilerplate_indices = set()
        for i, sentence in enumerate(sentences):
            if self.is_boilerplate_sentence(sentence):
                boilerplate_indices.add(i)
        
        # 第二步：识别重复话术（统计方法）
        repetitive_indices = self.identify_repetitive_sentences(sentences, section_paths)
        
        # 创建SentenceInfo列表
        for i, sentence in enumerate(sentences):
            skip_embedding = False
            reason = ""
            
            if i in boilerplate_indices:
                skip_embedding = True
                reason = "boilerplate"
            elif i in repetitive_indices:
                skip_embedding = True
                reason = "repetitive"
            
            sentence_infos.append(SentenceInfo(
                text=sentence,
                skip_embedding=skip_embedding,
                reason=reason
            ))
        
        return sentence_infos
    
    def clean_and_split(
        self,
        text: str,
        section_path: List[str] = None
    ) -> Tuple[str, List[SentenceInfo]]:
        """
        完整的清洗和拆分流程
        
        Args:
            text: 原始文本
            section_path: 当前章节路径（用于判断是否在核心条款区）
        
        Returns:
            (清洗后的文本, 句子信息列表)
        """
        # 基础清洗
        cleaned_text = self.basic_clean(text)
        
        # 句级拆分
        sentences = self.split_into_sentences(cleaned_text)
        
        # 语义降噪
        section_paths = [section_path or []] * len(sentences)
        sentence_infos = self.semantic_denoise(sentences, section_paths)
        
        return cleaned_text, sentence_infos
