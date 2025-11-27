# main_rag.py 示例

import os
from multi_data_loader import multi_chunk_associate
from multi_embedding import create_multimodal_store
# 从 multi_retrieval 导入两个搜索函数
from multi_retrieval import load_multimodal_index, multi_search, multi_search_image
from multi_generation import multi_rag_fct
from langchain_community.vectorstores import FAISS # 引入 FAISS 类型，用于类型提示

# --- 配置 ---
DATA_DIR = "./data" 
VECTOR_DB_PATH = "./faiss_index"
INDEX_NAME = "multimodal_clip_faiss"
LLM_MODEL = "deepseek-r1:1.5b" 

# --- 1. 准备数据和索引创建 ---
def initialize_store(data_dir=DATA_DIR, pthName=VECTOR_DB_PATH, idxName=INDEX_NAME) -> FAISS | None:
    """初始化或加载多模态向量存储。"""
    if not os.path.exists(f"{pthName}/{idxName}.faiss"):
        print("--- 步骤 1: 加载和分块数据 ---")
        all_documents = multi_chunk_associate(data_dir=data_dir)
        
        if not all_documents:
            print("没有找到任何文档，请检查 DATA_DIR。")
            return None
        
        print("--- 步骤 2: 创建多模态向量存储 ---")
        vector_store = create_multimodal_store(all_documents, pthName, idxName)
        return vector_store
    else:
        print("FAISS 索引已存在，跳过创建。")
        return load_multimodal_index(pthName, idxName)


# --- 2. RAG 查询执行  ---
def run_query(
    vector_store: FAISS,
    llmModelName: str = LLM_MODEL,
    query_text: str = None,
    query_image_path: str = None,
):
    """
    多模态 RAG 核心函数：根据文本或图像进行检索、增强、生成。
    """
    if not vector_store:
        print("无法执行查询，向量存储未初始化。")
        return
    
    if query_text and query_image_path:
        print("请仅提供文本查询或图像查询之一。")
        return
    
    retrieved_docs = []
    
    # --- 检索 (Retrieval) 逻辑 ---
    aggregated_scores = {}
    if query_text:
        print(f"\n--- 步骤 3: 执行文本 RAG 查询：'{query_text}' ---")
        retrieved_docs, aggregated_scores = multi_search(query_text, vector_store, k=6)
        query = query_text # RAG 提示使用的查询
    
    elif query_image_path:
        print(f"\n--- 步骤 3: 执行图像 RAG 查询：'{query_image_path}' ---")
        retrieved_docs, aggregated_scores = multi_search_image(query_image_path, vector_store, k=6)
        query = f"用户提供了一张图片（路径：{query_image_path}），请根据检索到的上下文信息，判断是否有与该图片相似的失物或描述？"
    
    else:
        print("未提供有效的查询文本或查询图像路径。")
        return

    # --- 增强和生成 (Augmentation & Generation) ---
    if not retrieved_docs:
        print("未检索到任何相关文档。")
        return

    # 调用 multi_generation.py 中的核心生成函数
    text_ctx, img_paths, scores, normalized_scores, item_rankings, response = multi_rag_fct(
        query,
        llmModelName,
        vector_store,
        retrieved_docs=retrieved_docs,
        aggregated_hits=aggregated_scores,
    )
    
    print("\n--- 检索到的上下文 ---")
    print(f"**检索到的图像路径 (已找回物品):**\n{img_paths}")
    print(f"**检索到的文本 (已找回物品描述):**\n{text_ctx}")
    
    if scores:
        print("\n--- 向量相似度分数（含 0-100 展示值） ---")
        for i, (raw, scaled) in enumerate(zip(scores, normalized_scores), 1):
            print(f"项目 {i}: 原始分数={raw:.6f}, 0-100 展示值={scaled:.2f}")
    
    if item_rankings:
        print("\n--- Item 综合得分 ---")
        for item in item_rankings:
            hits_desc = ", ".join(
                f"{hit['chunk_type']}->{hit['query_chunk']}({hit['adjusted_score']:.4f})"
                for hit in item["hits"]
            )
            hits_desc = hits_desc or "无命中详情"
            print(f"Item {item['item_id']}: 总分 {item['score']:.4f} | 命中: {hits_desc}")
    
    print("\n--- LLM 最终响应 ---")
    print(response)


if __name__ == "__main__":
    # 确保有一个 'data' 文件夹里面有您的文本和图像文件
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"请在 '{DATA_DIR}' 目录下放置您的已找回物品的 .html/.txt 描述和 .jpg/.png 文件。")
    
    vector_store = initialize_store()
    
    if vector_store:
        # --- 示例 1: 文本查询 ---
        # 假设用户描述了他丢失的物品
        
        text_query = input("输入文本信息：")
        run_query(vector_store, query_text=text_query)
        
        print("\n" + "="*50 + "\n")
        
        # --- 示例 2: 图像查询 ---
        # 假设用户上传了他丢失物品（如钥匙扣）的照片
        # 注意：您需要确保 'data/test_image.jpg' 文件实际存在于您的文件系统中以进行测试
        image_query_path = "data/lost_keychain.png"
        if os.path.exists(image_query_path):
             run_query(vector_store, query_image_path=image_query_path)
        else:
            print(f"跳过图像查询：测试图片 '{image_query_path}' 不存在。请替换为有效路径以进行测试。") 