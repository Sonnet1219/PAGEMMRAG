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
python scripts/pdf_colqwen2_rag.py ocr -h
python scripts/pdf_colqwen2_rag.py patch -h
python scripts/pdf_colqwen2_rag.py patch-index -h
python scripts/pdf_colqwen2_rag.py patch-search -h
python scripts/pdf_colqwen2_rag.py pipeline -h
```

### 6.4 MinerU Layout 分块裁剪（para block patch）

如果你已经跑完 MinerU OCR 并拿到 `*_middle.json`，可以把 layout 对应的 para block 裁成 patch：

```bash
python scripts/pdf_colqwen2_rag.py patch \
  --pdf ./outputs/2410.05779v3.pdf \
  --middle-json ./outputs/mineru_ocr_full_20260301_033146/2410.05779v3/ocr/2410.05779v3_middle.json \
  --output-dir ./outputs/mineru_patch_layout_v2_demo \
  --overwrite
```

输出结构：
- `para_blocks_raw/`：原始坐标裁剪 patch
- `para_blocks_padded/`：按长宽比规则 padding 后 patch
- `meta.json`：每个 patch 的页码、类型、bbox、像素坐标与尺寸
- `preview_*.jpg`：按类型和混合预览拼图

常用参数：
- `--types title text interline_equation image list table`
- `--dpi 300`
- `--margin-pt 2.0`
- `--max-aspect-ratio 120`
- `--min-short-edge 12`

### 6.5 MinerU OCR 原始输出（落地 middle/model/content_list/md）

```bash
python scripts/pdf_colqwen2_rag.py ocr \
  --pdf ./outputs/2410.05779v3.pdf \
  --output-dir ./outputs/mineru_ocr_full_demo \
  --method ocr \
  --backend pipeline \
  --mineru-device cuda:0 \
  --overwrite
```

说明：
- 默认优先使用 `./.venv_mineru/bin/mineru`（如果存在），避免和 ColQwen2 环境冲突。
- 会在 OCR 输出目录写入 `ocr_run_meta.json`（记录命令和关键产物路径）。

### 6.6 Patch -> ColQwen2 -> Qdrant（patch 索引）

```bash
python scripts/pdf_colqwen2_rag.py patch-index \
  --patch-dir ./outputs/mineru_patch_layout_v2_demo \
  --index-dir ./outputs/mineru_patch_index_demo \
  --collection colqwen2_patches \
  --overwrite
```

### 6.7 一键完整 Pipeline（OCR -> Patch -> Patch 索引）

```bash
python scripts/pdf_colqwen2_rag.py pipeline \
  --pdf ./outputs/2410.05779v3.pdf \
  --overwrite
```

默认输出：
- `outputs/mineru_ocr_full_<timestamp>/`
- `outputs/mineru_patch_layout_v2_<timestamp>/`
- `outputs/mineru_patch_index_<timestamp>/`

### 6.8 Patch 检索（query 返回最相似 patch 图）

```bash
python scripts/pdf_colqwen2_rag.py patch-search \
  --index-dir ./outputs/mineru_patch_index_demo \
  --query "实验设置部分在哪里" \
  --top-k 5 \
  --patch-types text table image \
  --copy-best ./outputs/best_patch.png \
  --output-json ./outputs/patch_search_results.json
```

可选：
- `--doc-id your_doc_id`：覆盖 metadata 中的文档 id。
- 不传 `--patch-types` 时会在该文档全部 patch 类型中检索。
