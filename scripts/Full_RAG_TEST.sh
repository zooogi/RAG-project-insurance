#!/bin/bash
# 完整RAG流程测试脚本

echo "=========================================="
echo "RAG Pipeline 完整流程测试"
echo "=========================================="

# 测试文档处理流程（embedding 之前的流程）
# 前提：必须在 raw_data 目录下有原始数据！！！
echo ""
echo "【步骤1】处理文档（OCR -> 清洗 -> Chunk -> 索引构建）"
echo "命令: python -m app.main process --input data/raw_data"
echo ""
read -p "是否执行文档处理？(y/n): " process_choice
if [ "$process_choice" = "y" ]; then
    python -m app.main process --input data/raw_data
fi

echo ""
echo "=========================================="
echo "【步骤2】完整RAG流程测试（检索 -> Rerank -> LLM生成）"
echo "=========================================="

# 1. 测试pdf保险条款+内嵌数据表格（友邦保险）
echo ""
echo "测试1: PDF保险条款+表格查询"
echo "查询: 如果我今年45岁，友邦终身寿险可以选择哪些付费年限？"
python -m app.main query --query "如果我今年45岁，友邦终身寿险可以选择哪些付费年限？"

# 2. 测试表格数据（csv）
echo ""
echo "测试2: CSV表格数据查询"
echo "查询: 吸烟者 (smoker=yes) 的平均保险费用(Average charges)是多少？"
python -m app.main query --query "吸烟者 (smoker=yes) 的平均保险费用(Average charges)是多少？"

# 3. 纯图片测试
echo ""
echo "测试3: 图片OCR查询"
echo "查询: 在知识库里的一份图片合同说明，这款附加合同的保险期间是多久"
python -m app.main query --query "在知识库里的一份图片合同说明，这款附加合同的保险期间是多久"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
