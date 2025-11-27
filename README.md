# 多模态检索增强生成 (RAG) 系统

本项目实现了一个基于 CLIP 和 FAISS 的多模态 RAG 框架，用于高效地从非结构化和半结构化的失物招领数据中检索信息，并利用 LLM 针对性地生成回答。
# 📚 项目简介：多模态检索增强生成 (RAG) 系统

本项目是一个专为**失物招领**场景设计的高级 RAG 系统，旨在通过结合文本和图像的检索能力，提供精确且详细的物品匹配和信息总结。

## 🎯 核心目标

根据用户输入的**文本描述**或**查询图像**，本项目能够从结构化的失物记录（包含文字描述、图片路径、分类等信息）中，检索出最相关的物品记录，并利用大型语言模型 (LLM) 生成清晰的匹配摘要和回复。

## 💡 技术亮点

| 要素 | 详情 | 作用 |
| :--- | :--- | :--- |
| **多模态嵌入** | **CLIP 模型** (`openai/clip-vit-base-patch32`) | 将文本和图像嵌入到统一的向量空间，实现跨模态搜索。 |
| **向量检索** | **FAISS** (L2 距离) + **L2 归一化** | 实现高效的近似最近邻搜索。通过 L2 归一化，L2 距离被精确转换为**余弦相似度**（核心转换公式：$\text{sim} = 1 - d^2 / 2$）。 |
| **检索策略** | **多块加权搜索** | 文本查询会被智能地分解为 `category`、`date`、`description` 等语义块，并根据权重（`CHUNK_WEIGHTS`）进行加权搜索，提高检索精度。 |
| **生成模型** | **本地 LLM** (`deepseek-r1:1.5b`) | 通过 **Ollama 平台**部署运行，用于接收检索到的上下文，并生成最终的、高质量的回复。 |
| **输入支持** | **文本** 和 **图像** | 系统原生支持文本描述和失物图片作为查询输入。 |

本项目通过精细的语义分块和加权检索策略，确保了在多模态数据背景下，能够实现高效、准确的检索增强问答。
## 📁 项目结构概览

.
├── data/                  # 待索引的原始数据 (描述文件和图片)  
├── faiss_index/           # 向量存储索引文件 (自动生成)  
├── multi_data_loader.py   # 数据处理：分块 (Chunking) 模块  
├── multi_embedding.py     # 嵌入：CLIP 模型和向量编码模块  
├── multi_retrieval.py     # 检索：多块查询、加权和得分聚合模块  
├── multi_generation.py    # 生成：RAG 上下文构建和 LLM 交互模块  
├── main_rag.py            # 主程序：初始化、查询执行流程控制  
├── diagnose.py            # 辅助：索引诊断与相似度转换验证  
└── requirements.txt       # Python 依赖

## 📜 模块功能与关键函数

### 1. `main_rag.py` (主程序入口)

负责整个 RAG 流程的协调和控制。

| 函数 | 功能描述 |
| :--- | :--- |
| `initialize_store()` | 检查 FAISS 索引是否存在。如果不存在，则调用数据加载和创建索引的函数；如果存在，则调用 `load_multimodal_index` 加载现有索引。 |
| `run_query()` | 根据传入的 `query_text` 或 `query_image_path` 调用相应的检索函数 (`multi_search` 或 `multi_search_image`)，然后调用 `multi_rag_fct` 执行增强和生成。 |

### 2. `multi_data_loader.py` (数据分块模块)

负责将原始文本文件结构化并分解为可检索的语义块（Document）。

| 类/函数 | 功能描述 |
| :--- | :--- |
| `ItemRecord` | **数据容器**：结构化表示单个失物条目（包括 `item_id`, `category`, `description`, `image_paths` 等）。 |
| `_parse_item_file()` | 读取原始文件，利用正则匹配提取结构化字段 (`category`, `date_found` 等)，生成 `ItemRecord` 对象。 |
| `_build_chunk_documents()` | 根据 `ItemRecord`，创建 LangChain 的 `Document` 列表。为每个字段（`category`, `date`, `description`, `image`）生成一个独立 Document，并带上多模态检索所需的元数据 (`item_id`, `chunk_type` 等)。 |
| `multi_chunk_associate()` | **核心函数**：遍历 `data_dir` 中的所有文本文件，执行解析和分块，返回所有语义块 `Document` 组成的列表。 |

### 3. `multi_embedding.py` (嵌入模块)

封装 CLIP 模型，实现文本和图像的向量编码，并负责 FAISS 索引的创建。

| 类/函数 | 功能描述 |
| :--- | :--- |
| `load_clip_model()` | 加载预训练的 CLIP 模型 (`openai/clip-vit-base-patch32`) 和处理器。 |
| `CLIPEmbeddingsWrapper(Embeddings)` | **嵌入封装器**：实现 LangChain 的 `Embeddings` 接口，包含 `embed_text` 和 `embed_image` 方法。 |
| `embed_text(text)` | 将文本通过 CLIP 文本编码器编码为向量，并进行 L2 归一化。 |
| `embed_image(image_path)` | 加载图像文件，通过 CLIP 图像编码器编码为向量，并进行 L2 归一化。 |
| `create_multimodal_store()` | **核心函数**：接收所有 `Document`，分别对文本块和图像块嵌入，将结果存入 `FAISS.from_embeddings`，并保存到本地。 |

### 4. `multi_retrieval.py` (检索模块)

处理查询解析、多块检索、L2 距离到余弦相似度的转换以及 Item 得分聚合。

| 变量/函数 | 功能描述 |
| :--- | :--- |
| `CHUNK_WEIGHTS` | **配置**：定义不同语义块类型（如 `image: 1.3`, `category: 1.2`）的得分权重，用于增强检索精度。 |
| `_distance_to_cosine_similarity()` | **关键转换**：将 FAISS 返回的 L2 距离 $d$ 转换为归一化向量的余弦相似度：$\text{sim} = 1 - d^2 / 2$。 |
| `_split_query_into_chunks()` | **查询解析**：将用户查询分解为加权的子查询块（`category`、`date`、`description`），以实现多维度搜索。 |
| `multi_search()` | **文本检索核心**：对每个查询块执行向量搜索，根据 `CHUNK_WEIGHTS` 调整得分，并将命中的块聚合到 Item 级别。 |
| `multi_search_image()` | **图像检索核心**：将查询图像嵌入为向量，执行基于向量的相似度搜索，并聚合得分。 |

### 5. `multi_generation.py` (生成模块)

负责将检索结果转化为 LLM 友好的上下文，并调用 LLM 生成最终答案。

| 函数 | 功能描述 |
| :--- | :--- |
| `normalize_scores_to_percentage()` | **得分归一化**：将原始向量相似度得分线性映射到 0-100 区间，用于 RAG 提示和用户展示。 |
| `format_multimodal_context()` | **上下文格式化**：从检索到的文档中提取和分离**文本内容**、**图像路径**、**原始得分**和**归一化得分**。 |
| `multi_rag_fct()` | **RAG 核心**：构建包含检索信息的增强 Prompt，发送给 LLM (`deepseek-r1:1.5b`)，并返回所需的检索上下文和最终响应。 |

### 6. `diagnose.py` (辅助诊断)

用于验证 L2 距离转换的准确性，并展示如何进一步处理相似度分数。

| 功能点 | 描述 |
| :--- | :--- |
| **L2 到余弦转换验证** | 加载索引，获取原始 FAISS L2 距离，并手动计算转换后的余弦相似度 (`1 - d^2/2`)。 |
| **Softmax 计算** | 将余弦相似度分数转换为 Softmax 概率分布，用于评估匹配的相对置信度。 |
















