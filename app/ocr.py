"""
OCR模块 - 支持PDF、图片和CSV文件的处理
- PDF: 使用MineRU进行高质量解析
- 图片(JPG/PNG等): 使用PaddleOCR轻量模型进行OCR识别
- CSV: 使用pandas直接读取并转换为Markdown表格
"""
import os
import json
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# #region agent log
import json as json_lib
log_path = Path("/home/chanson/Zhang/RAG-保险项目/.cursor/debug.log")
def _debug_log(location, message, data, hypothesis_id="A"):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json_lib.dumps({
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(__import__("time").time() * 1000)
            }) + "\n")
    except: pass
_debug_log("ocr.py:13", "开始导入pandas模块", {}, "A")
# #endregion agent log

try:
    import pandas as pd
    # #region agent log
    _debug_log("ocr.py:15", "pandas导入成功", {"pandas_version": pd.__version__ if hasattr(pd, "__version__") else "unknown"}, "A")
    # #endregion agent log
    PANDAS_AVAILABLE = True
except ImportError as e:
    # #region agent log
    _debug_log("ocr.py:18", "pandas导入失败", {"error": str(e), "error_type": type(e).__name__}, "A")
    # #endregion agent log
    PANDAS_AVAILABLE = False
    pd = None

# PaddleOCR导入（可选，如果未安装会提示）
# #region agent log
_debug_log("ocr.py:22", "开始导入PaddleOCR模块", {}, "B")
# #endregion agent log
try:
    from paddleocr import PaddleOCR
    # #region agent log
    _debug_log("ocr.py:25", "PaddleOCR导入成功", {}, "B")
    # #endregion agent log
    PADDLEOCR_AVAILABLE = True
except ImportError as e:
    # #region agent log
    _debug_log("ocr.py:28", "PaddleOCR导入失败", {"error": str(e), "error_type": type(e).__name__}, "B")
    # #endregion agent log
    PADDLEOCR_AVAILABLE = False

# 类级别的PaddleOCR模型缓存，避免重复加载占用显存
_paddleocr_cache: Optional[Any] = None


class DocumentProcessor:
    """
    文档处理器，支持多种文件格式
    
    功能：
    1. PDF文件：使用MineRU处理
    2. 图片文件（JPG/PNG等）：使用PaddleOCR轻量模型OCR识别
    3. CSV文件：使用pandas读取并转换为Markdown表格
    """
    
    def __init__(
        self,
        output_base_dir: str = "data/processed",
        source: str = "modelscope",
        use_gpu: bool = True,
        use_paddleocr_slim: bool = True  # 使用轻量模型节省显存
    ):
        """
        初始化文档处理器
        
        Args:
            output_base_dir: 输出基础目录
            source: MineRU模型源 ('modelscope' 或 'huggingface')
            use_gpu: 是否使用GPU加速
            use_paddleocr_slim: 是否使用PaddleOCR轻量模型（节省显存）
        """
        self.output_base_dir = Path(output_base_dir)
        self.source = source
        self.use_gpu = use_gpu
        self.use_paddleocr_slim = use_paddleocr_slim
        
        # 创建输出目录
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查MineRU是否安装（用于PDF处理）
        self._check_mineru_installed()
        
        # 初始化PaddleOCR（延迟加载，只在需要时加载）
        self.paddleocr = None
    
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
                print("✓ MineRU已安装（用于PDF处理）")
            else:
                raise RuntimeError("MineRU未正确安装")
        except FileNotFoundError:
            print("⚠ MineRU未安装，PDF处理功能将不可用")
            print("  安装命令: pip install mineru>=2.7.0")
        except Exception as e:
            print(f"⚠ 检查MineRU时出错: {e}")
    
    def _get_paddleocr(self):
        """获取PaddleOCR实例（使用缓存）"""
        global _paddleocr_cache
        
        # #region agent log
        _debug_log("ocr.py:125", "_get_paddleocr函数入口", {"PADDLEOCR_AVAILABLE": PADDLEOCR_AVAILABLE, "cache_exists": _paddleocr_cache is not None}, "F")
        # #endregion agent log
        
        if not PADDLEOCR_AVAILABLE:
            # #region agent log
            _debug_log("ocr.py:130", "PaddleOCR不可用", {}, "F")
            # #endregion agent log
            raise RuntimeError(
                "PaddleOCR未安装，请运行: pip install paddleocr\n"
                "如果显存不足，可以使用CPU版本: pip install paddleocr --no-deps paddlepaddle"
            )
        
        if _paddleocr_cache is None:
            print("正在加载PaddleOCR（使用轻量模型，节省显存）...")
            # #region agent log
            _debug_log("ocr.py:137", "开始初始化PaddleOCR", {}, "F")
            # #endregion agent log
            try:
                # 使用基本参数，PaddleOCR会自动检测GPU是否可用
                # 移除use_gpu参数，因为某些版本的PaddleOCR不支持此参数
                _paddleocr_cache = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch'  # 中文
                )
                # #region agent log
                _debug_log("ocr.py:145", "PaddleOCR初始化成功", {}, "F")
                # #endregion agent log
                print("✓ PaddleOCR加载成功")
            except Exception as e:
                # #region agent log
                _debug_log("ocr.py:149", "PaddleOCR初始化失败", {"error": str(e), "error_type": type(e).__name__}, "F")
                # #endregion agent log
                error_msg = str(e)
                if "No module named 'paddle'" in error_msg or "No module named 'paddlepaddle'" in error_msg:
                    raise RuntimeError(
                        "PaddleOCR初始化失败：缺少PaddlePaddle依赖。\n"
                        "请安装PaddlePaddle：\n"
                        "  - CPU版本: pip install paddlepaddle\n"
                        "  - GPU版本: pip install paddlepaddle-gpu\n"
                        "或者安装完整依赖: pip install paddleocr paddlepaddle"
                    )
                print(f"✗ PaddleOCR加载失败: {e}")
                raise
        
        # #region agent log
        _debug_log("ocr.py:155", "_get_paddleocr函数返回", {"using_cache": _paddleocr_cache is not None}, "F")
        # #endregion agent log
        return _paddleocr_cache
    
    def process_file(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False
    ) -> Dict:
        """
        处理文件（自动识别文件类型）
        
        Args:
            file_path: 文件路径（PDF、图片或CSV）
            output_dir: 输出目录（如果为None，自动生成）
            overwrite: 如果文件已存在，是否覆盖（False则跳过）
            
        Returns:
            包含处理结果的字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名选择处理方法
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.process_pdf(file_path, output_dir, overwrite=overwrite)
        elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self.process_image(file_path, output_dir, overwrite=overwrite)
        elif suffix == '.csv':
            return self.process_csv(file_path, output_dir, overwrite=overwrite)
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        extract_images: bool = True,
        extract_tables: bool = True,
        overwrite: bool = False
    ) -> Dict:
        """
        处理PDF文件（使用MineRU）
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            extract_images: 是否提取图片
            extract_tables: 是否提取表格
            overwrite: 如果文件已存在，是否覆盖（False则跳过）
            
        Returns:
            处理结果字典
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
        
        # 检查是否已存在处理结果
        pdf_name = pdf_path.stem
        mineru_output = output_dir / pdf_name / "hybrid_auto"
        md_file = mineru_output / f"{pdf_name}.md"
        
        if md_file.exists() and not overwrite:
            print(f"\n{'='*60}")
            print(f"PDF文件已处理: {pdf_path.name}")
            print(f"Markdown文件已存在: {md_file}")
            print(f"跳过处理（如需重新处理请设置 overwrite=True）")
            print(f"{'='*60}\n")
            # 直接读取已存在的文件
            return self._parse_mineru_output(pdf_path, output_dir)
        
        print(f"\n{'='*60}")
        print(f"开始处理PDF: {pdf_path.name}")
        print(f"输出目录: {output_dir}")
        if overwrite and md_file.exists():
            print(f"⚠ 将覆盖已存在的文件")
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
                timeout=600
            )
            
            if result.returncode != 0:
                print(f"MineRU执行出错:\n{result.stderr}")
                raise RuntimeError(f"MineRU处理失败: {result.stderr}")
            
            print("✓ MineRU处理完成")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("MineRU处理超时（超过10分钟）")
        except FileNotFoundError:
            raise RuntimeError("MineRU未安装，请运行: pip install mineru>=2.7.0")
        except Exception as e:
            raise RuntimeError(f"执行MineRU时出错: {e}")
        
        # 解析处理结果
        result_data = self._parse_mineru_output(pdf_path, output_dir)
        
        print(f"\n{'='*60}")
        print("处理完成！")
        print(f"{'='*60}\n")
        
        return result_data
    
    def process_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False
    ) -> Dict:
        """
        处理图片文件（使用PaddleOCR）
        
        Args:
            image_path: 图片文件路径
            output_dir: 输出目录
            overwrite: 如果文件已存在，是否覆盖（False则跳过）
            
        Returns:
            处理结果字典
        """
        # #region agent log
        _debug_log("ocr.py:273", "process_image函数入口", {"image_path": str(image_path), "overwrite": overwrite}, "G")
        # #endregion agent log
        
        image_path = Path(image_path)
        
        # #region agent log
        _debug_log("ocr.py:280", "检查图片文件是否存在", {"image_path": str(image_path), "exists": image_path.exists()}, "G")
        # #endregion agent log
        
        if not image_path.exists():
            # #region agent log
            _debug_log("ocr.py:283", "图片文件不存在", {"image_path": str(image_path)}, "G")
            # #endregion agent log
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = self.output_base_dir / image_path.stem
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在处理结果
        md_file = output_dir / f"{image_path.stem}.md"
        if md_file.exists() and not overwrite:
            print(f"\n{'='*60}")
            print(f"图片文件已处理: {image_path.name}")
            print(f"Markdown文件已存在: {md_file}")
            print(f"跳过处理（如需重新处理请设置 overwrite=True）")
            print(f"{'='*60}\n")
            # 读取已存在的文件
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            return {
                "file_path": str(image_path.absolute()),
                "file_name": image_path.stem,
                "file_type": "image",
                "output_dir": str(output_dir.absolute()),
                "markdown": markdown_content,
                "files": {
                    "markdown": str(md_file)
                },
                "statistics": {
                    "total_lines": len(markdown_content.split('\n')),
                    "total_text_length": len(markdown_content)
                }
            }
        
        print(f"\n{'='*60}")
        print(f"开始处理图片: {image_path.name}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
        
        # 获取PaddleOCR实例
        # #region agent log
        _debug_log("ocr.py:342", "准备获取PaddleOCR实例", {}, "G")
        # #endregion agent log
        ocr = self._get_paddleocr()
        
        # #region agent log
        _debug_log("ocr.py:346", "PaddleOCR实例获取成功，准备调用ocr方法", {"image_path": str(image_path)}, "G")
        # #endregion agent log
        
        # 执行OCR识别
        print("正在执行OCR识别...")
        try:
            # #region agent log
            _debug_log("ocr.py:351", "调用ocr.ocr()方法", {"image_path": str(image_path), "cls": True}, "G")
            # #endregion agent log
            result = ocr.ocr(str(image_path), cls=True)
            
            # #region agent log
            _debug_log("ocr.py:354", "ocr.ocr()调用成功", {"result_type": type(result).__name__, "result_not_none": result is not None, "result_length": len(result) if result else 0}, "G")
            # #endregion agent log
            
            # 解析OCR结果
            text_lines = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]  # 提取文本
                        confidence = line[1][1]  # 提取置信度
                        if confidence > 0.5:  # 过滤低置信度结果
                            text_lines.append(text)
            
            # 构建Markdown内容
            markdown_content = "\n".join(text_lines)
            
            # 保存Markdown文件
            md_file = output_dir / f"{image_path.stem}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"✓ OCR识别完成，识别到 {len(text_lines)} 行文本")
            print(f"✓ Markdown文件已保存: {md_file}")
            
            result_data = {
                "file_path": str(image_path.absolute()),
                "file_name": image_path.stem,
                "file_type": "image",
                "output_dir": str(output_dir.absolute()),
                "markdown": markdown_content,
                "text_lines": text_lines,
                "files": {
                    "markdown": str(md_file)
                },
                "statistics": {
                    "total_lines": len(text_lines),
                    "total_text_length": len(markdown_content)
                }
            }
            
            print(f"\n{'='*60}")
            print("处理完成！")
            print(f"{'='*60}\n")
            
            return result_data
            
        except Exception as e:
            # #region agent log
            _debug_log("ocr.py:393", "OCR识别异常", {"error": str(e), "error_type": type(e).__name__}, "G")
            # #endregion agent log
            print(f"✗ OCR识别失败: {e}")
            raise
    
    def process_csv(
        self,
        csv_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        overwrite: bool = False
    ) -> Dict:
        """
        处理CSV文件（使用pandas读取并转换为Markdown）
        
        Args:
            csv_path: CSV文件路径
            output_dir: 输出目录
            overwrite: 如果文件已存在，是否覆盖（False则跳过）
            
        Returns:
            处理结果字典
        """
        # #region agent log
        _debug_log("ocr.py:321", "process_csv函数入口", {"csv_path": str(csv_path), "PANDAS_AVAILABLE": PANDAS_AVAILABLE}, "C")
        # #endregion agent log
        
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            # #region agent log
            _debug_log("ocr.py:328", "CSV文件不存在", {"csv_path": str(csv_path)}, "C")
            # #endregion agent log
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
        
        if not PANDAS_AVAILABLE:
            # #region agent log
            _debug_log("ocr.py:333", "pandas不可用，无法处理CSV", {}, "A")
            # #endregion agent log
            raise RuntimeError("pandas未安装，无法处理CSV文件。请运行: pip install pandas")
        
        # 确定输出目录
        if output_dir is None:
            output_dir = self.output_base_dir / csv_path.stem
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否已存在处理结果
        md_file = output_dir / f"{csv_path.stem}.md"
        if md_file.exists() and not overwrite:
            print(f"\n{'='*60}")
            print(f"CSV文件已处理: {csv_path.name}")
            print(f"Markdown文件已存在: {md_file}")
            print(f"跳过处理（如需重新处理请设置 overwrite=True）")
            print(f"{'='*60}\n")
            # 读取已存在的文件
            with open(md_file, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            # 尝试读取DataFrame（如果可能）
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(csv_path, encoding='gbk')
                except:
                    df = None
            return {
                "file_path": str(csv_path.absolute()),
                "file_name": csv_path.stem,
                "file_type": "csv",
                "output_dir": str(output_dir.absolute()),
                "markdown": markdown_content,
                "dataframe": df,
                "files": {
                    "markdown": str(md_file)
                },
                "statistics": {
                    "total_rows": len(df) if df is not None else 0,
                    "total_columns": len(df.columns) if df is not None else 0,
                    "column_names": list(df.columns) if df is not None else [],
                    "total_text_length": len(markdown_content)
                }
            }
        
        print(f"\n{'='*60}")
        print(f"开始处理CSV: {csv_path.name}")
        print(f"输出目录: {output_dir}")
        print(f"{'='*60}\n")
        
        try:
            # 读取CSV文件
            print("正在读取CSV文件...")
            # #region agent log
            _debug_log("ocr.py:350", "开始读取CSV文件", {"csv_path": str(csv_path)}, "C")
            # #endregion agent log
            # 尝试不同的编码
            try:
                df = pd.read_csv(csv_path, encoding='utf-8')
                # #region agent log
                _debug_log("ocr.py:354", "CSV读取成功(UTF-8)", {"rows": len(df), "columns": len(df.columns)}, "C")
                # #endregion agent log
            except UnicodeDecodeError as e:
                # #region agent log
                _debug_log("ocr.py:357", "UTF-8编码失败，尝试GBK", {"error": str(e)}, "C")
                # #endregion agent log
                try:
                    df = pd.read_csv(csv_path, encoding='gbk')
                    # #region agent log
                    _debug_log("ocr.py:361", "CSV读取成功(GBK)", {"rows": len(df), "columns": len(df.columns)}, "C")
                    # #endregion agent log
                except Exception as e2:
                    # #region agent log
                    _debug_log("ocr.py:364", "GBK编码失败，使用errors='ignore'", {"error": str(e2)}, "C")
                    # #endregion agent log
                    df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')
            
            # 转换为Markdown表格
            # #region agent log
            _debug_log("ocr.py:369", "开始转换为Markdown表格", {}, "D")
            # #endregion agent log
            try:
                markdown_table = df.to_markdown(index=False)
                # #region agent log
                _debug_log("ocr.py:372", "使用pandas.to_markdown成功", {"table_length": len(markdown_table)}, "D")
                # #endregion agent log
            except Exception as e:
                # #region agent log
                _debug_log("ocr.py:375", "pandas.to_markdown失败，尝试tabulate", {"error": str(e)}, "D")
                # #endregion agent log
                # 如果to_markdown不可用，手动构建Markdown表格
                try:
                    from tabulate import tabulate
                    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
                    # #region agent log
                    _debug_log("ocr.py:380", "使用tabulate成功", {"table_length": len(markdown_table)}, "D")
                    # #endregion agent log
                except ImportError as e2:
                    # #region agent log
                    _debug_log("ocr.py:382", "tabulate导入失败", {"error": str(e2)}, "D")
                    # #endregion agent log
                    raise RuntimeError("tabulate未安装，无法转换表格。请运行: pip install tabulate")
            
            # 添加标题
            markdown_content = f"# {csv_path.stem}\n\n{markdown_table}"
            
            # 保存Markdown文件
            md_file = output_dir / f"{csv_path.stem}.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            print(f"✓ CSV读取完成，共 {len(df)} 行，{len(df.columns)} 列")
            print(f"✓ Markdown文件已保存: {md_file}")
            
            result_data = {
                "file_path": str(csv_path.absolute()),
                "file_name": csv_path.stem,
                "file_type": "csv",
                "output_dir": str(output_dir.absolute()),
                "markdown": markdown_content,
                "dataframe": df,  # 保留DataFrame对象供后续使用
                "files": {
                    "markdown": str(md_file)
                },
                "statistics": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "column_names": list(df.columns),
                    "total_text_length": len(markdown_content)
                }
            }
            
            print(f"\n{'='*60}")
            print("处理完成！")
            print(f"{'='*60}\n")
            
            return result_data
            
        except Exception as e:
            print(f"✗ CSV处理失败: {e}")
            raise
    
    def _parse_mineru_output(
        self,
        pdf_path: Path,
        output_dir: Path
    ) -> Dict:
        """
        解析MineRU的输出结果（保持原有逻辑）
        """
        pdf_name = pdf_path.stem
        
        # MineRU的输出结构: output_dir/pdf_name/hybrid_auto/
        mineru_output = output_dir / pdf_name / "hybrid_auto"
        
        if not mineru_output.exists():
            raise RuntimeError(f"MineRU输出目录不存在: {mineru_output}")
        
        result = {
            "file_path": str(pdf_path.absolute()),
            "file_name": pdf_name,
            "file_type": "pdf",
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
        
        # 统计信息
        if "content_list" in result:
            result["statistics"] = self._calculate_statistics(result["content_list"])
        
        return result
    
    def _calculate_statistics(self, content_list: List[Dict]) -> Dict:
        """计算文档统计信息"""
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
        从处理结果中提取纯文本/Markdown
        
        Args:
            result: process_file返回的结果字典
            
        Returns:
            提取的文本/Markdown内容
        """
        if "markdown" in result:
            return result["markdown"]
        
        if result.get("file_type") == "csv" and "dataframe" in result:
            # CSV文件，返回Markdown表格
            return result["markdown"]
        
        if "content_list" in result:
            # PDF的content_list格式
            text_parts = []
            for item in result["content_list"]:
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif item.get("type") == "list" and "list_items" in item:
                    for list_item in item["list_items"]:
                        text_parts.append(list_item)
            return "\n\n".join(text_parts)
        
        raise ValueError("结果中没有可用的文本内容")
    
    def extract_by_page(self, result: Dict) -> Dict[int, str]:
        """
        按页码提取文本（仅对PDF有效）
        
        Args:
            result: process_file返回的结果字典（必须是PDF处理结果）
            
        Returns:
            页码到文本的映射字典，格式: {0: "第0页文本", 1: "第1页文本", ...}
        """
        if result.get("file_type") != "pdf":
            raise ValueError("extract_by_page方法仅支持PDF文件")
        
        if "content_list" not in result:
            raise ValueError("结果中没有content_list，无法按页提取")
        
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
            result: process_file返回的结果字典
            output_path: 输出文件路径（如果为None，自动生成到result的output_dir下）
            
        Returns:
            保存的文件路径
        """
        # 对于PDF文件，如果已经有MineRU生成的markdown文件，直接返回该路径，不创建新文件
        if result.get("file_type") == "pdf" and "files" in result and "markdown" in result["files"]:
            existing_md = Path(result["files"]["markdown"])
            if existing_md.exists():
                print(f"✓ PDF文件已有Markdown文件: {existing_md}")
                return existing_md
        
        text = self.extract_text(result)
        
        if output_path is None:
            # 使用result中的output_dir，而不是创建text文件夹
            output_dir = Path(result.get('output_dir', self.output_base_dir))
            file_name = result.get("file_name", "unknown")
            output_path = output_dir / f"{file_name}.md"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✓ 文本已保存到: {output_path}")
        return output_path
    
    def batch_process(
        self,
        input_dir: Union[str, Path],
        pattern: str = "*.*",
        overwrite: bool = False
    ) -> List[Dict]:
        """
        批量处理文件（支持PDF、图片、CSV）
        
        Args:
            input_dir: 输入目录
            pattern: 文件匹配模式
            overwrite: 如果文件已存在，是否覆盖（False则跳过）
            
        Returns:
            处理结果列表
        """
        # #region agent log
        _debug_log("ocr.py:770", "batch_process函数入口", {"input_dir": str(input_dir), "pattern": pattern, "overwrite": overwrite}, "H")
        # #endregion agent log
        
        input_dir = Path(input_dir)
        files = list(input_dir.glob(pattern))
        
        # 过滤出支持的文件类型
        supported_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.csv'}
        files = [f for f in files if f.suffix.lower() in supported_extensions]
        
        # #region agent log
        _debug_log("ocr.py:779", "文件列表过滤完成", {"total_files": len(files), "file_names": [f.name for f in files]}, "H")
        # #endregion agent log
        
        if not files:
            print(f"在 {input_dir} 中没有找到支持的文件（PDF/图片/CSV）")
            return []
        
        print(f"\n找到 {len(files)} 个文件")
        
        results = []
        skipped = 0
        for i, file_path in enumerate(files, 1):
            print(f"\n处理 {i}/{len(files)}: {file_path.name}")
            # #region agent log
            _debug_log("ocr.py:792", "开始处理文件", {"file_index": i, "total_files": len(files), "file_name": file_path.name, "file_suffix": file_path.suffix}, "H")
            # #endregion agent log
            try:
                result = self.process_file(file_path, overwrite=overwrite)
                # #region agent log
                _debug_log("ocr.py:795", "文件处理成功", {"file_name": file_path.name, "file_type": result.get("file_type")}, "H")
                # #endregion agent log
                results.append(result)
            except Exception as e:
                # #region agent log
                _debug_log("ocr.py:799", "文件处理异常", {"file_name": file_path.name, "error": str(e), "error_type": type(e).__name__}, "H")
                # #endregion agent log
                # 检查是否是跳过已存在文件的提示
                if "跳过处理" in str(e) or "已存在" in str(e):
                    skipped += 1
                    continue
                print(f"✗ 处理失败: {e}")
                continue
        
        print(f"\n批量处理完成！成功: {len(results)}/{len(files)}, 跳过: {skipped}")
        return results


# 保持向后兼容的类名
PDFProcessor = DocumentProcessor


# 便捷函数
def create_processor(
    output_base_dir: str = "data/processed",
    **kwargs
) -> DocumentProcessor:
    """
    创建文档处理器的便捷函数
    
    Args:
        output_base_dir: 输出基础目录
        **kwargs: 其他参数
        
    Returns:
        DocumentProcessor实例
    """
    return DocumentProcessor(output_base_dir=output_base_dir, **kwargs)


def process_single_file(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    **kwargs
) -> Dict:
    """
    处理单个文件的便捷函数（支持PDF、图片、CSV）
    
    Args:
        file_path: 文件路径
        output_dir: 输出目录
        overwrite: 如果文件已存在，是否覆盖（False则跳过）
        **kwargs: 其他参数
        
    Returns:
        处理结果字典
    """
    processor = create_processor(**kwargs)
    return processor.process_file(file_path, output_dir, overwrite=overwrite)


# 保持向后兼容的函数
def process_single_pdf(
    pdf_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    **kwargs
) -> Dict:
    """
    处理单个PDF文件的便捷函数（向后兼容）
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录
        overwrite: 如果文件已存在，是否覆盖（False则跳过）
        **kwargs: 其他参数
        
    Returns:
        处理结果字典
    """
    processor = create_processor(**kwargs)
    return processor.process_pdf(pdf_path, output_dir, overwrite=overwrite)


def clear_paddleocr_cache():
    """清空PaddleOCR缓存，释放显存"""
    global _paddleocr_cache
    _paddleocr_cache = None
    print("✓ PaddleOCR缓存已清空")


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试 OCR 模块")
    print("=" * 60)
    
    # 创建处理器
    processor = create_processor(use_paddleocr_slim=True)
    
    # 测试文件路径
    test_files = [
        "data/pdf/保险基础知多少.pdf",  # PDF
        "data/raw_data/保险图片.jpg",   # 图片
        "data/raw_data/insurance - 副本.csv"  # CSV
    ]
    
    for test_file in test_files:
        test_path = Path(test_file)
        if test_path.exists():
            print(f"\n测试处理: {test_file}")
            try:
                result = processor.process_file(test_path)
                print(f"✓ 处理成功，文件类型: {result.get('file_type')}")
                if "statistics" in result:
                    print(f"  统计信息: {result['statistics']}")
            except Exception as e:
                print(f"✗ 处理失败: {e}")
        else:
            print(f"\n测试文件不存在: {test_file}")
