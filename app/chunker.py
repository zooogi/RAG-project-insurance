# app/chunker.py

import re
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


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


@dataclass
class Chunk:
    """统一的Chunk结构"""
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": asdict(self.metadata)
        }


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
        overlap_size: int = 100
    ):
        """
        初始化分块器
        
        Args:
            target_chunk_size: 目标chunk大小（字符数）
            max_chunk_size: 最大chunk大小
            min_chunk_size: 最小chunk大小
            overlap_size: chunk之间的重叠大小
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

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

    def parse_markdown(self, md_path: Path) -> List[Dict[str, Any]]:
        """
        解析Markdown文件，提取结构化内容块
        
        Returns:
            List of content blocks with metadata
        """
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
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
        
        return blocks

    def create_chunks_from_blocks(
        self,
        blocks: List[Dict[str, Any]],
        source_file: str
    ) -> List[Chunk]:
        """
        从内容块创建chunks，保证语义完整性
        
        策略：
        1. 表格作为独立chunk，不拆分
        2. 列表尽量保持完整
        3. 段落在超过max_size时才拆分
        4. 相邻小块可以合并（在同一section下）
        """
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
            
            # 创建chunk
            chunk_id = str(uuid.uuid4())
            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                chunk_type='table' if has_table else ('list' if has_list else 'paragraph'),
                section_path=current_section_path.copy(),
                heading_level=current_heading_level,
                char_count=len(chunk_text),
                image_refs=list(set(image_refs)),  # 去重
                source_file=source_file,
                start_line=start_line,
                end_line=end_line,
                has_table=has_table,
                has_list=has_list
            )
            
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata
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
            
            # 表格作为独立chunk
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

    def chunk_markdown_file(self, md_path: Path) -> List[Chunk]:
        """
        对单个Markdown文件进行分块
        
        Args:
            md_path: Markdown文件路径
            
        Returns:
            Chunk列表
        """
        # 解析Markdown结构
        blocks = self.parse_markdown(md_path)
        
        # 创建chunks
        chunks = self.create_chunks_from_blocks(
            blocks,
            source_file=str(md_path)
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
                
                # 如果指定了输出目录，保存结果
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 生成输出文件名
                    relative_path = md_file.relative_to(input_dir)
                    output_file = output_dir / f"{relative_path.stem}_chunks.json"
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 保存为JSON
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(
                            [chunk.to_dict() for chunk in chunks],
                            f,
                            ensure_ascii=False,
                            indent=2
                        )
                    print(f"  保存到: {output_file}")
                    
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
