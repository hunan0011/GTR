# construct_tree_kmeans_full.py
import os
import pickle as pkl
import numpy as np
import pandas as pd
import config as user_config

# --- 优先使用 cuML KMeans (GPU 加速) ---
try:
    from cuml.cluster import KMeans as cuMLKMeans
    print("cuML detected. Using GPU for clustering and mean computation.")
    USE_GPU = True
except ImportError:
    from sklearn.cluster import KMeans
    print("cuML not found. Falling back to sklearn KMeans (CPU).")
    USE_GPU = False

# ---------------- CONFIG ----------------
MEMMAP_PATH = user_config.MEMMAP_PATH
ID2OFFSET_PATH = "/hdd02/jiangyutao/Code/msmarco_docs_bge_id2offset.tsv"
EMBEDDING_DIM = user_config.EMBEDDING_DIM
NODE_BALANCE = user_config.NODE_BALANCE       # B
TREE_HEIGHT = user_config.TREE_HEIGHT         # H

TREE_DIR = user_config.TREE_DIR
OUTPUT_TREE_PATH = user_config.OUTPUT_TREE_PATH
CHILDREN_EMBEDDINGS_PATH = user_config.CHILDREN_EMBEDDINGS_PATH
ID2PATH = user_config.ID2PATH
LEAF2ID = getattr(user_config, 'LEAF2ID', f"{TREE_DIR}/leaf2docs.pkl")

RANDOM_SEED = user_config.RANDOM_SEED

# ==========================================================
# 配置：文档向量不归一化，节点质心归一化（用于内积搜索）
# ==========================================================
NORMALIZE_INPUT = False       # 文档向量保持原始模长
NORMALIZE_CENTROIDS = True    # 所有节点 embedding（质心）强制归一化

print(f"Config: Raw Input Vectors + Normalized Centroids (for Inner Product Search)")

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
np.random.seed(RANDOM_SEED)

# ==========================================================
#                       Helper Functions
# ==========================================================
def save_object(obj, path):
    with open(path, 'wb') as f:
        pkl.dump(obj, f)

def load_object(path):
    with open(path, 'rb') as f:
        return pkl.load(f)

def l2_normalize(arr):
    """L2 归一化，支持 numpy array 或单向量"""
    if arr.shape[0] == 0:
        return arr
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (arr / norm).astype(np.float32)

# ==========================================================
#                       TreeNode
# ==========================================================
class TreeNode(object):
    def __init__(self, node_id_str, item_embedding=None, layer=None):
        self.val = node_id_str
        self.embedding = item_embedding  # 单位向量
        self.parent = None
        self.children = []
        self.isleaf = False
        self.pids = []
        self.layer = layer
        self.node_id_int = None

    def add(self, node):
        self.children.append(node)

# ==========================================================
#                       TreeInitialize
# ==========================================================
class TreeInitialize(object):
    def __init__(self, pid_embeddings_raw, pids, blance_factor=NODE_BALANCE, tree_height=TREE_HEIGHT):
        # 不对输入向量归一化，直接使用原始嵌入
        self.embeddings = pid_embeddings_raw.astype(np.float32)  # memmap → readonly numpy
        self.pids = pids
        self.root = None
        self.blance_factor = blance_factor
        self.tree_height = tree_height
        self.leaf_dict = {}
        self.node_count = 0

    def _k_means_clustering(self, pid_embeddings, n_clusters):
        N = pid_embeddings.shape[0]
        K_actual = min(n_clusters, N)
        if K_actual <= 0:
            return None, None, 0

        # 在原始向量上聚类（欧氏距离）
        if USE_GPU:
            kmeans = cuMLKMeans(
                n_clusters=K_actual,
                max_iter=3000,
                n_init=100,
                random_state=RANDOM_SEED,
                output_type='numpy'
            )
            labels = kmeans.fit_predict(pid_embeddings)
            raw_centers = kmeans.cluster_centers_.astype(np.float32)
        else:
            kmeans = KMeans(
                n_clusters=K_actual,
                max_iter=3000,
                n_init=100,
                random_state=RANDOM_SEED
            )
            labels = kmeans.fit_predict(pid_embeddings)
            raw_centers = kmeans.cluster_centers_.astype(np.float32)

        # 关键：质心强制归一化
        centers = l2_normalize(raw_centers)

        return labels, centers, K_actual

    def _build_full_tree(self, node, pid_embeddings, pids, layer):
        N = pid_embeddings.shape[0]
        print(f"Layer {layer}, Node {node.val}: samples={N}")

        if layer >= self.tree_height - 1:
            node.isleaf = True
            node.pids = pids
            self.leaf_dict[node.val] = node
            return node

        labels, centers, K_actual = self._k_means_clustering(pid_embeddings, self.blance_factor)

        cluster_map = {k: [] for k in range(K_actual)}
        labels_np = labels.get() if hasattr(labels, 'get') else labels
        for i, label in enumerate(labels_np):
            cluster_map[label].append(i)

        for slot in range(self.blance_factor):
            child_id_str = node.val + "_" + str(slot)
            idx_list = cluster_map.get(slot, [])

            if len(idx_list) == 0:
                # 空簇：继承父节点已归一化的 embedding
                child = TreeNode(node_id_str=child_id_str, item_embedding=node.embedding, layer=layer + 1)
                child.parent = node
                self._build_full_tree(child, np.zeros((0, EMBEDDING_DIM), dtype=np.float32), np.array([], dtype=np.int64), layer + 1)
            else:
                idx_array = np.array(idx_list, dtype=np.int64)
                child_embedding = centers[slot]  # 已归一化
                child_pids_chunk = pid_embeddings[idx_array]
                child_pids = pids[idx_array]

                child = TreeNode(node_id_str=child_id_str, item_embedding=child_embedding, layer=layer + 1)
                child.parent = node
                self._build_full_tree(child, child_pids_chunk, child_pids, layer + 1)
            node.add(child)
        return node

    def clustering_tree(self):
        print(f"\n===== Start building Tree (H={self.tree_height}, B={self.blance_factor}) =====")
        root = TreeNode('0', layer=0)

        # 根节点：全局平均向量 → 归一化
        if self.embeddings.shape[0] > 0:
            if USE_GPU:
                global_kmeans = cuMLKMeans(n_clusters=1, random_state=RANDOM_SEED, output_type='numpy')
                global_kmeans.fit(self.embeddings)
                root_emb = l2_normalize(global_kmeans.cluster_centers_)[0]
            else:
                mean_vec = np.mean(self.embeddings, axis=0).reshape(1, -1)
                root_emb = l2_normalize(mean_vec)[0]
        else:
            root_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        root.embedding = root_emb
        self.root = self._build_full_tree(root, self.embeddings, self.pids, layer=0)
        print("===== Tree build complete =====")
        return self.root

    def assign_node_ids(self):
        q = [self.root]
        next_id = 0
        while q:
            node = q.pop(0)
            node.node_id_int = next_id
            next_id += 1
            for child in node.children:
                q.append(child)
        self.node_count = next_id
        print(f"Assigned integer IDs to {self.node_count} nodes.")

# ==========================================================
#                       Post-processing
# ==========================================================
def build_children_id_embedding_map(tree):
    print("\n[Mapping] Building children ID embedding map...")
    mapping = {}
    q = [tree.root]
    while q:
        node = q.pop(0)
        if node.isleaf:
            continue
        children_info = []
        for idx, ch in enumerate(node.children):
            children_info.append({
                "child_id": ch.node_id_int,
                "child_index": idx,
                "embedding": ch.embedding.tolist()  # 单位向量
            })
            q.append(ch)
        mapping[node.node_id_int] = children_info
    return mapping

def build_docid2path_and_leaf2docs(tree, pid_to_docid):
    print("\n[docid2paths] building mappings...")
    docid2path = {}
    leaf2docs = {}
    total_leaves = 0
    empty_leaves = 0

    def dfs(node, cur_path):
        nonlocal total_leaves, empty_leaves
        current_path = cur_path + [node.node_id_int]
        if node.isleaf:
            total_leaves += 1
            if len(node.pids) == 0:
                empty_leaves += 1
            cur_docs = []
            for pid in node.pids:
                if (did := pid_to_docid.get(pid)):
                    docid2path[did] = current_path
                    cur_docs.append(did)
            if cur_docs:
                leaf2docs[node.node_id_int] = cur_docs
            return
        for ch in node.children:
            dfs(ch, current_path)

    dfs(tree.root, [])
    print(f"Total leaves: {total_leaves}, Empty leaves: {empty_leaves}")
    return docid2path, leaf2docs

# ================= MAIN =================
if __name__ == '__main__':
    os.makedirs(TREE_DIR, exist_ok=True)
    if TREE_HEIGHT < 1:
        raise ValueError("TREE_HEIGHT must be at least 1")

    print(f"Loading ID mapping from {ID2OFFSET_PATH}...")
    id_map = pd.read_csv(ID2OFFSET_PATH, sep='\t')
    pid_to_docid = dict(zip(id_map['offset'], id_map['docid']))

    print(f"Loading Memmap from {MEMMAP_PATH}...")
    pid_embeddings_raw = np.memmap(MEMMAP_PATH, dtype=np.float32, mode="r").reshape(-1, EMBEDDING_DIM)
    pids_all = np.arange(pid_embeddings_raw.shape[0])
    print(f"Total pid num: {len(pids_all)}")

    tree_initializer = TreeInitialize(
        pid_embeddings_raw, pids_all,
        blance_factor=NODE_BALANCE,
        tree_height=TREE_HEIGHT
    )
    tree = tree_initializer.clustering_tree()

    tree_initializer.assign_node_ids()

    print(f"Saving GMTree object to {OUTPUT_TREE_PATH}...")
    save_object(tree_initializer, OUTPUT_TREE_PATH)

    children_map = build_children_id_embedding_map(tree_initializer)
    save_object(children_map, CHILDREN_EMBEDDINGS_PATH)
    print(f"Saved children mapping to {CHILDREN_EMBEDDINGS_PATH}")

    docid2path_map, leaf2docs_map = build_docid2path_and_leaf2docs(tree_initializer, pid_to_docid)
    save_object(docid2path_map, ID2PATH)
    save_object(leaf2docs_map, LEAF2ID)
    print(f"Saved mappings to {ID2PATH} and {LEAF2ID}")

    print("All done (Raw document vectors + Normalized centroids for Inner Product Search).")