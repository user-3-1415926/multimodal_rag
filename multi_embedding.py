# multi_embedding.py
import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
try:
    from langchain.schema import Document
except Exception:
    # 回退 Document 定义，满足本模块创建 Document 的简单需求
    class Document:
        def __init__(self, page_content: str, metadata: dict | None = None):
            self.page_content = page_content
            self.metadata = metadata or {}
from typing import List

# 定义使用的 CLIP 模型名称
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def load_clip_model():
    """加载 CLIP 模型和处理器。"""
    print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
    try:
        # 尝试从 Hugging Face Hub 加载，它会自动处理下载和缓存
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        # 修正：添加 use_fast=False 以匹配模型配置并消除警告
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=False) 
        
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        raise
        
    return model, processor

class CLIPEmbeddingsWrapper(Embeddings):
    """LangChain Embeddings 接口封装器，用于处理文本和图像。"""
    def __init__(self, clip_model, clip_processor):
        self.model = clip_model
        self.processor = clip_processor
        # 获取向量维度，用于返回零向量。
        # projection_dim 在不同模型/transformers 版本中位置可能不同，尽量稳健获取，
        # 如果无法从 config 获取，会在第一次 embed 时根据实际向量形状设置。
        self.vector_dim = getattr(self.model.config, "projection_dim", None)
        if self.vector_dim is None:
            # 常见备用属性
            self.vector_dim = getattr(self.model.config, "hidden_size", None)
        
    def embed_text(self, text: str) -> List[float]:
        """将文本编码为向量。"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        vec = text_features.squeeze(0).cpu().numpy()
        # 在首次运行时，如果 vector_dim 仍为 None，则使用实际向量维度
        if self.vector_dim is None:
            try:
                self.vector_dim = int(vec.shape[-1])
            except Exception:
                self.vector_dim = len(vec)

        # L2 归一化向量，确保余弦相似度计算有效
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            # 视为无效/零向量，返回显式的零向量（长度为 vector_dim）
            return [0.0] * int(self.vector_dim)
        vec = vec / norm
        # 强制为 float32，便于 FAISS 使用
        vec = vec.astype(np.float32)
        return vec.tolist()

    def embed_image(self, image_path: str) -> List[float]:
        """将图像编码为向量。"""
    
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        vec = image_features.squeeze(0).cpu().numpy()
        # 在首次运行时，如果 vector_dim 仍为 None，则使用实际向量维度
        if self.vector_dim is None:
            try:
                self.vector_dim = int(vec.shape[-1])
            except Exception:
                self.vector_dim = len(vec)

        norm = np.linalg.norm(vec)
        vec = vec / norm
        vec = vec.astype(np.float32)
        return vec.tolist()
    

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        # 必须实现，但实际上在 create_multimodal_store 中手动处理
        return [self.embed_text(doc) for doc in documents]

    def embed_query(self, text: str) -> List[float]:
        """查询嵌入使用文本编码器。"""
        return self.embed_text(text)


def create_multimodal_store(all_documents: List[Document], pthName: str, idxName: str):
    """
    将文本和图像文档编码并存储到统一的 FAISS 向量数据库中。
    """
    clip_model, clip_processor = load_clip_model()
    clip_embeddings = CLIPEmbeddingsWrapper(clip_model, clip_processor)
    
    # 1. 准备数据和向量
    texts = []       # 用于存储 page_content (即文本内容或图像路径)
    metadatas = []   # 用于存储 Document 的 metadata
    embeddings = []  # 用于存储 CLIP 向量
    
    for doc in all_documents:
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)

        chunk_type = doc.metadata.get("chunk_type", doc.metadata.get("type", "text"))
        if chunk_type == "image":
            embeddings.append(clip_embeddings.embed_image(doc.page_content))
        else:
            embeddings.append(clip_embeddings.embed_text(doc.page_content))
   
    vector_data_tuples = list(zip(texts, embeddings))
    # 将嵌入列表转换为 numpy 数组
    # embeddings_array = np.array(embeddings) # 这里不再需要转换为 numpy array

    # 2. 从向量、文本和元数据创建 FAISS 存储
    # LangChain 的 from_embeddings 预期接收的参数
    # 注意：faiss.from_embeddings 期望每项为 (text, embedding)，
    # 之前的实现把 (embedding, text) 传入，导致 Document.page_content 被赋值为 list（向量），
    # 从而触发 Pydantic 的字符串校验失败。这里修正顺序为 (text, embedding)。
    index = FAISS.from_embeddings(
        # 传入 (文本内容, 向量) 元组的列表
        text_embeddings=list(zip(texts, embeddings)), 
        embedding=clip_embeddings,
        metadatas=metadatas  # 传入单独的元数据列表
    )

    # 3. 保存向量存储 (其余代码保持不变)
    if not os.path.exists(pthName):
        os.makedirs(pthName)
        
    index.save_local(folder_path=pthName, index_name=idxName)
    print(f"Multimodal vector store saved successfully to {pthName}/{idxName}.faiss")

    return index