# ColQwen2 本地部署（venv）

本项目已在本机完成一次实际部署验证，模型为 `vidore/colqwen2-v1.0`，可在 GPU 上输出图像与文本 embedding。

## 1. 进入虚拟环境

```bash
cd /home/coder/colqwen2
source .venv/bin/activate
```

## 2. 安装依赖（首次）

```bash
pip install -r requirements.txt
```

## 3. 运行 embedding 脚本

先准备一张图片，例如 `./sample.png`，然后执行：

```bash
python scripts/embed_colqwen2.py \
  --image ./sample.png \
  --query "这是一个什么场景？" \
  --output-json ./outputs/result.json
```

脚本会输出：
- `image_embeddings.shape`
- `query_embeddings.shape`
- `scores`（query 与 image 的多向量匹配分数）

并将均值池化后的 embedding 与分数保存到 JSON。

## 4. 参数说明

```bash
python scripts/embed_colqwen2.py -h
```

常用参数：
- `--device cuda:0`：指定设备（默认自动选 `cuda:0` 或 `cpu`）。
- `--dtype bfloat16|float16|float32`：模型精度（GPU 推荐 `bfloat16`）。
- `--attn auto|none|sdpa|eager|flash_attention_2`：注意力实现。

## 5. 说明

- 首次运行会从 Hugging Face 下载模型权重到本地缓存。
- 如需更快下载，建议设置 `HF_TOKEN`（可选）。

## 6. PDF -> ColQwen2 + Qdrant 多向量检索 Pipeline

脚本：`scripts/pdf_colqwen2_rag.py`

### 6.1 建索引（PDF 转页图 + 页级多向量写入 Qdrant）

```bash
python scripts/pdf_colqwen2_rag.py index \
  --pdf ./your.pdf \
  --index-dir ./outputs/your_pdf_index \
  --overwrite
```

产物：
- `outputs/your_pdf_index/pages/page_XXXX.png`：每页渲染后的图像
- `outputs/your_pdf_index/index_meta.json`：索引元数据（包含 `doc_id`、collection、Qdrant 连接信息）
- `outputs/your_pdf_index/qdrant_storage/`：本地 Qdrant 存储目录（默认）

默认是 **本地持久化 Qdrant**。如果你有远程 Qdrant 服务，也可以改成：

```bash
python scripts/pdf_colqwen2_rag.py index \
  --pdf ./your.pdf \
  --index-dir ./outputs/your_pdf_index \
  --qdrant-url http://localhost:6333 \
  --collection colqwen2_pages \
  --overwrite
```

### 6.2 检索（query 返回最相似页图）

```bash
python scripts/pdf_colqwen2_rag.py search \
  --index-dir ./outputs/your_pdf_index \
  --query "这份文档里哪里在讲费用报表？" \
  --top-k 3 \
  --copy-best ./outputs/best_page.png \
  --output-json ./outputs/search_results.json
```

输出内容：
- 控制台打印 TopK 页码、分数、页图路径
- `--copy-best` 会把第一条 query 的 Top1 页图拷贝到指定位置
- `--output-json` 保存结构化检索结果

可选：在同一个 collection 里指定文档过滤

```bash
python scripts/pdf_colqwen2_rag.py search \
  --index-dir ./outputs/your_pdf_index \
  --doc-id your_doc_id \
  --query "你的问题"
```

### 6.3 帮助

```bash
python scripts/pdf_colqwen2_rag.py -h
python scripts/pdf_colqwen2_rag.py index -h
python scripts/pdf_colqwen2_rag.py search -h
```
