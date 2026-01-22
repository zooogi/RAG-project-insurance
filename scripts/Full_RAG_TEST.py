### 测试文档处理流程（embedding 之前的流程）
##前提必须在raw data 有原始数据！！！
python -m app.main process --input data/raw_data

### 完整RAG流程测试

## 1. 测试pdf保险条款+内嵌数据表格（友邦保险）
python -m app.main query --query "如果我今年45岁，友邦终身寿险可以选择哪些付费年限？"

## 2. 测试表格数据（csv）
python -m app.main query --query "吸烟者 (smoker=yes) 的平均保险费用(Average charges)是多少？"

## 3. 纯图片测试
python -m app.main query --query "在知识库里的一份图片合同说明，这款附加合同的保险期间是多久"