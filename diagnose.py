from multi_retrieval import load_multimodal_index, _distance_to_cosine_similarity
import numpy as np

# 加载索引（不加载 CLIP 模型，直接用 vs 的 embeddings 对象）
vs = load_multimodal_index('./faiss_index', 'multimodal_clip_faiss')

# 查询
query = '黑色 iPad 10 代平板电脑'

# 获取原始 FAISS 分数（L2 距离）
raw = vs.similarity_search_with_score(query, k=4)
print("=== Raw FAISS L2 distances ===")
dists = []
for i, (doc, score) in enumerate(raw):
    print(f"Item {i+1}: distance={score:.6f}")
    dists.append(score)

# 手动计算转换后的余弦相似度
print("\n=== Converted cosine similarity (1 - d^2/2) ===")
sims = []
for i, dist in enumerate(dists):
    sim = _distance_to_cosine_similarity(dist)
    print(f"Item {i+1}: distance={dist:.6f} -> similarity={sim:.6f} ({sim*100:.2f}%)")
    sims.append(sim)

# 计算这些相似度的 softmax
x = np.array(sims, dtype=np.float64)
x = x - np.max(x)
e = np.exp(x)
probs = e / np.sum(e)
print("\n=== Softmax probabilities ===")
for i, prob in enumerate(probs):
    print(f"Item {i+1}: {prob*100:.2f}%")
