"""
PDF OCR处理模块
使用MineRU进行PDF文档解析和内容提取
"""

import os
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
from tqdm import tqdm


class PDFProcessor:
    """PDF文档处理器，使用MineRU进行OCR和内容提取"""
    
    def __init__(
        self,
        pdf_dir: str = "data/pdf",
        output_dir: str = "data/processed",
        temp_dir: str = "data/mineru_temp",
        source: str = "modelscope"
    ):
        """
        初始化PDF处理器
        
        Args:
            pdf_dir: PDF文件所在目录
            output_dir: 处理后文件的输出目录
            temp_dir: MineRU临时输出目录
            source: MineRU模型源（modelscope或huggingface）
        """
        self.pdf_dir = Path(pdf_dir)
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.source = source
        
        # 创建输出目录结构
        self._create_output_dirs()
        
        logger.info(f"PDF处理器初始化完成")
        logger.info(f"PDF目录: {self.pdf_dir}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def _create_output_dirs(self):
        """创建输出目录结构"""
        subdirs = ["markdown", "text", "json", "images", "tables"]
        for subdir in subdirs:
            (self.output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 创建临时目录
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def get_pdf_files(self) -> List[Path]:
        """
        获取PDF目录下的所有PDF文件
        
        Returns:
            PDF文件路径列表
        """
        if not self.pdf_dir.exists():
            logger.error(f"PDF目录不存在: {self.pdf_dir}")
            return []
        
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        return pdf_files
    
    def is_processed(self, pdf_name: str) -> bool:
        """
        检查PDF是否已经处理过
        
        Args:
            pdf_name: PDF文件名（不含扩展名）
        
        Returns:
            是否已处理
        """
        markdown_file = self.output_dir / "markdown" / f"{pdf_name}.md"
        json_file = self.output_dir / "json" / f"{pdf_name}_metadata.json"
        
        return markdown_file.exists() and json_file.exists()
    
    def process_single_pdf(
        self,
        pdf_path: Path,
        skip_if_exists: bool = True
    ) -> bool:
        """
        处理单个PDF文件
        
        Args:
            pdf_path: PDF文件路径
            skip_if_exists: 如果已处理是否跳过
        
        Returns:
            处理是否成功
        """
        pdf_name = pdf_path.stem
        logger.info(f"开始处理: {pdf_name}")
        
        # 检查是否已处理
        if skip_if_exists and self.is_processed(pdf_name):
            logger.info(f"文件已处理，跳过: {pdf_name}")
            return True
        
        try:
            # 1. 调用MineRU处理PDF
            success = self._run_mineru(pdf_path)
            if not success:
                logger.error(f"MineRU处理失败: {pdf_name}")
                return False
            
            # 2. 提取和保存内容
            success = self._extract_and_save(pdf_name)
            if not success:
                logger.error(f"内容提取失败: {pdf_name}")
                return False
            
            # 3. 清理临时文件
            self._cleanup_temp_files(pdf_name)
            
            logger.info(f"处理完成: {pdf_name}")
            return True
            
        except Exception as e:
            logger.error(f"处理PDF时出错 {pdf_name}: {str(e)}")
            return False
    
    def _run_mineru(self, pdf_path: Path) -> bool:
        """
        运行MineRU命令行工具
        
        Args:
            pdf_path: PDF文件路径
        
        Returns:
            是否成功
        """
        try:
            # 构建MineRU命令
            cmd = [
                "mineru",
                "-p", str(pdf_path.absolute()),
                "-o", str(self.temp_dir.absolute()),
                "--source", self.source
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode != 0:
                logger.error(f"MineRU执行失败: {result.stderr}")
                return False
            
            logger.info("MineRU处理完成")
            return True
            
        except FileNotFoundError:
            logger.error("未找到mineru命令，请确保已安装MineRU")
            return False
        except Exception as e:
            logger.error(f"执行MineRU时出错: {str(e)}")
            return False
    
    def _extract_and_save(self, pdf_name: str) -> bool:
        """
        从MineRU输出中提取内容并保存到processed目录
        
        Args:
            pdf_name: PDF文件名（不含扩展名）
        
        Returns:
            是否成功
        """
        try:
            # MineRU输出目录结构: temp_dir/pdf_name/hybrid_auto/
            mineru_output = self.temp_dir / pdf_name / "hybrid_auto"
            
            if not mineru_output.exists():
                logger.error(f"MineRU输出目录不存在: {mineru_output}")
                return False
            
            # 1. 保存Markdown文件
            md_source = mineru_output / f"{pdf_name}.md"
            if md_source.exists():
                md_dest = self.output_dir / "markdown" / f"{pdf_name}.md"
                shutil.copy2(md_source, md_dest)
                logger.info(f"保存Markdown: {md_dest}")
                
                # 同时生成纯文本版本
                self._save_as_text(md_source, pdf_name)
            else:
                logger.warning(f"未找到Markdown文件: {md_source}")
            
            # 2. 保存JSON元数据
            json_source = mineru_output / f"{pdf_name}_content_list.json"
            if json_source.exists():
                json_dest = self.output_dir / "json" / f"{pdf_name}_metadata.json"
                shutil.copy2(json_source, json_dest)
                logger.info(f"保存JSON元数据: {json_dest}")
                
                # 提取图片和表格信息
                self._extract_images_and_tables(json_source, pdf_name, mineru_output)
            else:
                logger.warning(f"未找到JSON文件: {json_source}")
            
            return True
            
        except Exception as e:
            logger.error(f"提取内容时出错: {str(e)}")
            return False
    
    def _save_as_text(self, md_path: Path, pdf_name: str):
        """
        将Markdown转换为纯文本并保存
        
        Args:
            md_path: Markdown文件路径
            pdf_name: PDF文件名
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 简单的Markdown到文本转换（移除一些Markdown标记）
            text_content = content.replace('#', '').replace('*', '').replace('`', '')
            
            text_dest = self.output_dir / "text" / f"{pdf_name}.txt"
            with open(text_dest, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            logger.info(f"保存纯文本: {text_dest}")
            
        except Exception as e:
            logger.warning(f"保存纯文本时出错: {str(e)}")
    
    def _extract_images_and_tables(
        self,
        json_path: Path,
        pdf_name: str,
        mineru_output: Path
    ):
        """
        从JSON元数据中提取图片和表格信息
        
        Args:
            json_path: JSON文件路径
            pdf_name: PDF文件名
            mineru_output: MineRU输出目录
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            images = []
            tables = []
            
            for item in content_list:
                if item.get('type') == 'image':
                    images.append(item)
                elif item.get('type') == 'table':
                    tables.append(item)
            
            # 保存图片信息
            if images:
                images_info = {
                    'pdf_name': pdf_name,
                    'total_images': len(images),
                    'images': images
                }
                images_dest = self.output_dir / "images" / f"{pdf_name}_images.json"
                with open(images_dest, 'w', encoding='utf-8') as f:
                    json.dump(images_info, f, ensure_ascii=False, indent=2)
                logger.info(f"保存图片信息: {images_dest} ({len(images)}张)")
            
            # 保存表格信息
            if tables:
                tables_info = {
                    'pdf_name': pdf_name,
                    'total_tables': len(tables),
                    'tables': tables
                }
                tables_dest = self.output_dir / "tables" / f"{pdf_name}_tables.json"
                with open(tables_dest, 'w', encoding='utf-8') as f:
                    json.dump(tables_info, f, ensure_ascii=False, indent=2)
                logger.info(f"保存表格信息: {tables_dest} ({len(tables)}个)")
            
        except Exception as e:
            logger.warning(f"提取图片和表格信息时出错: {str(e)}")
    
    def _cleanup_temp_files(self, pdf_name: str):
        """
        清理临时文件
        
        Args:
            pdf_name: PDF文件名
        """
        try:
            temp_pdf_dir = self.temp_dir / pdf_name
            if temp_pdf_dir.exists():
                shutil.rmtree(temp_pdf_dir)
                logger.info(f"清理临时文件: {temp_pdf_dir}")
        except Exception as e:
            logger.warning(f"清理临时文件时出错: {str(e)}")
    
    def process_all_pdfs(
        self,
        skip_if_exists: bool = True
    ) -> Tuple[int, int]:
        """
        批量处理所有PDF文件
        
        Args:
            skip_if_exists: 如果已处理是否跳过
        
        Returns:
            (成功数量, 失败数量)
        """
        pdf_files = self.get_pdf_files()
        
        if not pdf_files:
            logger.warning("没有找到PDF文件")
            return 0, 0
        
        success_count = 0
        fail_count = 0
        
        logger.info(f"开始批量处理 {len(pdf_files)} 个PDF文件")
        
        # 使用进度条
        for pdf_path in tqdm(pdf_files, desc="处理PDF"):
            if self.process_single_pdf(pdf_path, skip_if_exists):
                success_count += 1
            else:
                fail_count += 1
        
        logger.info(f"批量处理完成: 成功 {success_count}, 失败 {fail_count}")
        return success_count, fail_count
    
    def get_processing_summary(self) -> Dict:
        """
        获取处理结果摘要
        
        Returns:
            处理结果统计信息
        """
        summary = {
            'total_pdfs': len(self.get_pdf_files()),
            'processed_pdfs': 0,
            'markdown_files': len(list((self.output_dir / "markdown").glob("*.md"))),
            'text_files': len(list((self.output_dir / "text").glob("*.txt"))),
            'json_files': len(list((self.output_dir / "json").glob("*.json"))),
            'image_files': len(list((self.output_dir / "images").glob("*.json"))),
            'table_files': len(list((self.output_dir / "tables").glob("*.json")))
        }
        
        # 统计已处理的PDF数量
        for pdf_file in self.get_pdf_files():
            if self.is_processed(pdf_file.stem):
                summary['processed_pdfs'] += 1
        
        return summary


def main():
    """主函数示例"""
    # 配置日志
    logger.add(
        "logs/ocr_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    
    # 创建处理器
    processor = PDFProcessor()
    
    # 批量处理所有PDF
    success, fail = processor.process_all_pdfs(skip_if_exists=True)
    
    # 显示摘要
    summary = processor.get_processing_summary()
    logger.info("=" * 50)
    logger.info("处理摘要:")
    logger.info(f"  总PDF数量: {summary['total_pdfs']}")
    logger.info(f"  已处理数量: {summary['processed_pdfs']}")
    logger.info(f"  Markdown文件: {summary['markdown_files']}")
    logger.info(f"  文本文件: {summary['text_files']}")
    logger.info(f"  JSON元数据: {summary['json_files']}")
    logger.info(f"  图片信息: {summary['image_files']}")
    logger.info(f"  表格信息: {summary['table_files']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
