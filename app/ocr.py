"""
OCR模块 - 使用MineRU进行PDF文档解析和文本提取
"""
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
import shutil


class PDFProcessor:
    """PDF文档处理器，使用MineRU进行高质量文档解析"""
    
    def __init__(
        self,
        output_base_dir: str = "data/processed",
        source: str = "modelscope",
        use_gpu: bool = True
    ):
        """
        初始化PDF处理器
        
        Args:
            output_base_dir: 输出基础目录
            source: 模型源 ('modelscope' 或 'huggingface')
            use_gpu: 是否使用GPU加速
        """
        self.output_base_dir = Path(output_base_dir)
        self.source = source
        self.use_gpu = use_gpu
        
        # 创建输出目录
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查MineRU是否安装
        self._check_mineru_installed()
    
    def _check_mineru_installed(self):
        """检查MineRU是否已安装"""
        try:
            result = subprocess.run(
                ["mineru", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                print("✓ MineRU已安装")
            else:
                raise RuntimeError("MineRU未正确安装")
        except FileNotFoundError:
            raise RuntimeError(
                "MineRU未安装，请运行: pip install mineru>=2.7.0"
            )
        except Exception as e:
            raise RuntimeError(f"检查MineRU时出错: {e}")
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        extract_images: bool = True,
        extract_tables: bool = True
    ) -> Dict:
        """
        处理单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录（如果为None，自动生成）
            extract_images: 是否提取图片
            extract_tables: 是否提取表格
        
        Returns:
            包含处理结果的字典
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = self.output_base_dir / pdf_path.stem
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"开始处理PDF: {pdf_path.name}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
        
        # 构建MineRU命令
        cmd = [
            "mineru",
            "-p", str(pdf_path.absolute()),
            "-o", str(output_dir.absolute()),
            "--source", self.source
        ]
        
        # 执行MineRU处理
        try:
            print("正在运行MineRU...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            if result.returncode != 0:
                print(f"MineRU执行出错:\n{result.stderr}")
                raise RuntimeError(f"MineRU处理失败: {result.stderr}")
            
            print("✓ MineRU处理完成")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("MineRU处理超时（超过10分钟）")
        except Exception as e:
            raise RuntimeError(f"执行MineRU时出错: {e}")
        
        # 解析处理结果
        result_data = self._parse_mineru_output(pdf_path, output_dir)
        
        print(f"\n{'='*60}")
        print("处理完成！")
        print(f"{'='*60}\n")
        
        return result_data
    
    def _parse_mineru_output(
        self,
        pdf_path: Path,
        output_dir: Path
    ) -> Dict:
        """
        解析MineRU的输出结果
        
        Args:
            pdf_path: 原始PDF路径
            output_dir: MineRU输出目录
        
        Returns:
            解析后的结果字典
        """
        pdf_name = pdf_path.stem
        
        # MineRU的输出结构: output_dir/pdf_name/hybrid_auto/
        mineru_output = output_dir / pdf_name / "hybrid_auto"
        
        if not mineru_output.exists():
            raise RuntimeError(f"MineRU输出目录不存在: {mineru_output}")
        
        result = {
            "pdf_path": str(pdf_path.absolute()),
            "pdf_name": pdf_name,
            "output_dir": str(output_dir.absolute()),
            "mineru_output_dir": str(mineru_output.absolute()),
            "files": {}
        }
        
        # 读取Markdown文件
        md_file = mineru_output / f"{pdf_name}.md"
        if md_file.exists():
            with open(md_file, 'r', encoding='utf-8') as f:
                result["markdown"] = f.read()
                result["files"]["markdown"] = str(md_file)
            print(f"✓ 读取Markdown文件: {md_file.name}")
        
        # 读取内容列表JSON
        content_list_file = mineru_output / f"{pdf_name}_content_list.json"
        if content_list_file.exists():
            with open(content_list_file, 'r', encoding='utf-8') as f:
                result["content_list"] = json.load(f)
                result["files"]["content_list"] = str(content_list_file)
            print(f"✓ 读取内容列表: {content_list_file.name}")
        
        # 读取内容列表v2 JSON
        content_list_v2_file = mineru_output / f"{pdf_name}_content_list_v2.json"
        if content_list_v2_file.exists():
            with open(content_list_v2_file, 'r', encoding='utf-8') as f:
                result["content_list_v2"] = json.load(f)
                result["files"]["content_list_v2"] = str(content_list_v2_file)
            print(f"✓ 读取内容列表v2: {content_list_v2_file.name}")
        
        # 读取中间JSON
        middle_file = mineru_output / f"{pdf_name}_middle.json"
        if middle_file.exists():
            with open(middle_file, 'r', encoding='utf-8') as f:
                result["middle_data"] = json.load(f)
                result["files"]["middle"] = str(middle_file)
            print(f"✓ 读取中间数据: {middle_file.name}")
        
        # 读取模型JSON
        model_file = mineru_output / f"{pdf_name}_model.json"
        if model_file.exists():
            with open(model_file, 'r', encoding='utf-8') as f:
                result["model_data"] = json.load(f)
                result["files"]["model"] = str(model_file)
            print(f"✓ 读取模型数据: {model_file.name}")
        
        # 检查布局PDF
        layout_pdf = mineru_output / f"{pdf_name}_layout.pdf"
        if layout_pdf.exists():
            result["files"]["layout_pdf"] = str(layout_pdf)
            print(f"✓ 找到布局PDF: {layout_pdf.name}")
        
        # 检查原始PDF副本
        origin_pdf = mineru_output / f"{pdf_name}_origin.pdf"
        if origin_pdf.exists():
            result["files"]["origin_pdf"] = str(origin_pdf)
        
        # 统计信息
        if "content_list" in result:
            result["statistics"] = self._calculate_statistics(result["content_list"])
        
        return result
    
    def _calculate_statistics(self, content_list: List[Dict]) -> Dict:
        """
        计算文档统计信息
        
        Args:
            content_list: 内容列表
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_items": len(content_list),
            "text_items": 0,
            "list_items": 0,
            "image_items": 0,
            "table_items": 0,
            "pages": set(),
            "total_text_length": 0
        }
        
        for item in content_list:
            item_type = item.get("type", "")
            
            if item_type == "text":
                stats["text_items"] += 1
                if "text" in item:
                    stats["total_text_length"] += len(item["text"])
            elif item_type == "list":
                stats["list_items"] += 1
            elif item_type == "image":
                stats["image_items"] += 1
            elif item_type == "table":
                stats["table_items"] += 1
            
            if "page_idx" in item:
                stats["pages"].add(item["page_idx"])
        
        stats["total_pages"] = len(stats["pages"])
        stats["pages"] = sorted(list(stats["pages"]))
        
        return stats
    
    def extract_text(self, result: Dict) -> str:
        """
        从处理结果中提取纯文本
        
        Args:
            result: process_pdf返回的结果字典
        
        Returns:
            提取的纯文本
        """
        if "markdown" in result:
            return result["markdown"]
        
        if "content_list" not in result:
            raise ValueError("结果中没有可用的文本内容")
        
        # 从content_list提取文本
        text_parts = []
        for item in result["content_list"]:
            if item.get("type") == "text" and "text" in item:
                text_parts.append(item["text"])
            elif item.get("type") == "list" and "list_items" in item:
                for list_item in item["list_items"]:
                    text_parts.append(list_item)
        
        return "\n\n".join(text_parts)
    
    def extract_by_page(self, result: Dict) -> Dict[int, str]:
        """
        按页码提取文本
        
        Args:
            result: process_pdf返回的结果字典
        
        Returns:
            页码到文本的映射字典
        """
        if "content_list" not in result:
            raise ValueError("结果中没有content_list")
        
        pages_text = {}
        
        for item in result["content_list"]:
            page_idx = item.get("page_idx", 0)
            
            if page_idx not in pages_text:
                pages_text[page_idx] = []
            
            if item.get("type") == "text" and "text" in item:
                pages_text[page_idx].append(item["text"])
            elif item.get("type") == "list" and "list_items" in item:
                for list_item in item["list_items"]:
                    pages_text[page_idx].append(list_item)
        
        # 合并每页的文本
        return {
            page: "\n\n".join(texts)
            for page, texts in pages_text.items()
        }
    
    def save_text(
        self,
        result: Dict,
        output_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        保存提取的文本到文件
        
        Args:
            result: process_pdf返回的结果字典
            output_path: 输出文件路径（如果为None，自动生成）
        
        Returns:
            保存的文件路径
        """
        text = self.extract_text(result)
        
        if output_path is None:
            output_path = self.output_base_dir / "text" / f"{result['pdf_name']}.txt"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ 文本已保存到: {output_path}")
        return output_path
    
    def batch_process(
        self,
        pdf_dir: Union[str, Path],
        pattern: str = "*.pdf"
    ) -> List[Dict]:
        """
        批量处理PDF文件
        
        Args:
            pdf_dir: PDF文件所在目录
            pattern: 文件匹配模式
        
        Returns:
            处理结果列表
        """
        pdf_dir = Path(pdf_dir)
        pdf_files = list(pdf_dir.glob(pattern))
        
        if not pdf_files:
            print(f"在 {pdf_dir} 中没有找到匹配 {pattern} 的PDF文件")
            return []
        
        print(f"\n找到 {len(pdf_files)} 个PDF文件")
        
        results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n处理 {i}/{len(pdf_files)}: {pdf_file.name}")
            try:
                result = self.process_pdf(pdf_file)
                results.append(result)
            except Exception as e:
                print(f"✗ 处理失败: {e}")
                continue
        
        print(f"\n批量处理完成！成功: {len(results)}/{len(pdf_files)}")
        return results


# 便捷函数
def create_processor(
    output_base_dir: str = "data/processed",
    **kwargs
) -> PDFProcessor:
    """
    创建PDF处理器的便捷函数
    
    Args:
        output_base_dir: 输出基础目录
        **kwargs: 其他参数
    
    Returns:
        PDFProcessor实例
    """
    return PDFProcessor(output_base_dir=output_base_dir, **kwargs)


def process_single_pdf(
    pdf_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> Dict:
    """
    处理单个PDF文件的便捷函数
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        **kwargs: 其他参数
    
    Returns:
        处理结果字典
    """
    processor = create_processor(**kwargs)
    return processor.process_pdf(pdf_path, output_dir)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 OCR 模块")
    print("=" * 60)
    
    # 创建处理器
    processor = create_processor()
    
    # 测试单个PDF
    test_pdf = "data/pdf/保险基础知多少.pdf"
    
    if Path(test_pdf).exists():
        print(f"\n测试处理: {test_pdf}")
        
        try:
            result = processor.process_pdf(test_pdf)
            
            # 显示统计信息
            if "statistics" in result:
                print("\n文档统计:")
                for key, value in result["statistics"].items():
                    print(f"  {key}: {value}")
            
            # 提取并保存文本
            text = processor.extract_text(result)
            print(f"\n提取的文本长度: {len(text)} 字符")
            print(f"\n文本预览（前500字符）:")
            print("-" * 60)
            print(text[:500])
            print("-" * 60)
            
            # 保存文本
            text_file = processor.save_text(result)
            
            print("\n处理完成！")
            
        except Exception as e:
            print(f"\n处理失败: {e}")
    else:
        print(f"\n测试文件不存在: {test_pdf}")
        print("请确保 data/pdf/ 目录下有PDF文件")
