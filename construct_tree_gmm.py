import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import config

# ==========================================================
#               GPU-ONLY PyCave GMM
# ==========================================================
try:
    from pycave.bayes import GaussianMixture as TorchGMM
except ImportError:
    raise ImportError("PyCave is required. CPU GMM is disabled.")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required. CPU GMM is disabled.")

print("Using PyCave GMM (GPU only)")
print(f"Device: {torch.cuda.get_device_name(0)}")

# ==========================================================
#                       CONFIG
# ==========================================================
MEMMAP_PATH = config.MEMMAP_PATH
ID2OFFSET_PATH = config.ID2OFFSET
EMBEDDING_DIM = config.EMBEDDING_DIM
NODE_BALANCE = config.NODE_BALANCE
TREE_HEIGHT = config.TREE_HEIGHT
PROB_THRESHOLD = config.PROB_THRESHOLD

TREE_DIR = config.TREE_DIR
OUTPUT_TREE_PATH = config.OUTPUT_TREE_PATH
CHILDREN_EMBEDDINGS_PATH = config.CHILDREN_EMBEDDINGS_PATH
ID2PATH = config.ID2PATH
LEAF2ID = config.LEAF2ID

RANDOM_SEED = config.RANDOM_SEED

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

# ==========================================================
#                       Utils
# ==========================================================
def save_object(obj, path):
    with open(path, "wb") as f:
        pkl.dump(obj, f)

def l2_normalize(x):
    if len(x) == 0:
        return x
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (x / norm).astype(np.float32)

# ==========================================================
#                       TreeNode
# ==========================================================
class TreeNode:
    def __init__(self, node_id_str, embedding=None, layer=0):
        self.val = node_id_str
        self.embedding = embedding      # Mean (Center)
        self.variance = None            # 【新增】Variance
        self.layer = layer
        self.parent = None
        self.children = []
        self.isleaf = False
        self.pids = []
        self.node_id_int = None

    def add(self, child):
        self.children.append(child)

# ==========================================================
#                 Tree Initialize (Soft GMM)
# ==========================================================
class TreeInitialize:
    def __init__(self, embeddings, pids, blance_factor, tree_height):
        self.embeddings = embeddings.astype(np.float32)
        self.pids = pids
        self.B = blance_factor
        self.H = tree_height
        self.root = None
        self.leaf_dict = {}
        self.node_count = 0

    def _gmm(self, X):
        N = X.shape[0]
        K = min(self.B, N)
        if K == 0:
            return None, None, None, 0

        gmm = TorchGMM(
            num_components=K,
            covariance_type="diag",
            covariance_regularization=1e-6,
            convergence_tolerance=1e-5,
            batch_size=4096,
            trainer_params=dict(
                accelerator="gpu",
                devices=1,
                max_epochs=500,
                enable_progress_bar=False,
                logger=False
            )
        )

        gmm.fit(X)
        probs = gmm.predict_proba(X).cpu().numpy()
        
        # 获取 Means 并归一化 (保持与原逻辑一致)
        centers = l2_normalize(gmm.model_.means.cpu().numpy())
        
        # 【新增】获取 Variances
        # PyCave 的 covariances 存储在 model_.covariances 中
        # 加上极小值防止数值问题，虽然 PyCave 内部已有 regularization
        variances = gmm.model_.covariances.cpu().numpy() + 1e-9

        return probs, centers, variances, K

    def _build(self, node, X, pids, layer):
        print(f"Layer {layer}, Node {node.val}, Samples {len(pids)}")

        if layer == self.H - 1:
            node.isleaf = True
            node.pids = pids
            self.leaf_dict[node.val] = node
            return node

        # 【修改】接收 variances
        probs, centers, variances, K = self._gmm(X)

        cluster_map = [[] for _ in range(K)]
        for i in range(len(pids)):
            idxs = np.where(probs[i] >= PROB_THRESHOLD)[0]
            if len(idxs) == 0:
                idxs = [np.argmax(probs[i])]
            for k in idxs:
                cluster_map[k].append(i)

        for k in range(self.B):
            child_id = f"{node.val}_{k}"
            idxs = cluster_map[k] if k < K else []
            
            if len(idxs) == 0:
                # 空节点：继承父节点 embedding，方差设为默认值 (例如 1.0)
                child = TreeNode(child_id, node.embedding, layer + 1)
                child.variance = np.ones_like(node.embedding) # 【新增】默认方差
                self._build(child, np.zeros((0, EMBEDDING_DIM), np.float32), np.array([], np.int64), layer + 1)
            else:
                idxs = np.asarray(idxs)
                # 正常节点：使用 GMM 计算出的 center 和 variance
                child = TreeNode(child_id, centers[k], layer + 1)
                child.variance = variances[k] # 【新增】存储方差
                self._build(child, X[idxs], pids[idxs], layer + 1)
            
            child.parent = node
            node.add(child)

        return node

    def clustering_tree(self):
        root = TreeNode("0", layer=0)
        # 根节点均值
        root.embedding = l2_normalize(self.embeddings.mean(0, keepdims=True))[0]
        # 根节点方差 (可以使用全局方差，这里简单设为 1)
        root.variance = np.ones(EMBEDDING_DIM, dtype=np.float32)
        
        self.root = self._build(root, self.embeddings, self.pids, 0)
        return self.root

    def assign_node_ids(self):
        q, nid = [self.root], 0
        while q:
            n = q.pop(0)
            n.node_id_int = nid
            nid += 1
            q.extend(n.children)
        self.node_count = nid

# ==========================================================
#                 Post Processing
# ==========================================================
def build_children_map(tree):
    mapping = {}
    q = [tree.root]
    while q:
        n = q.pop(0)
        if not n.isleaf:
            mapping[n.node_id_int] = [
                dict(
                    child_id=c.node_id_int, 
                    child_index=i, 
                    embedding=c.embedding.tolist(),
                    variance=c.variance.tolist() # 【新增】保存方差到字典
                )
                for i, c in enumerate(n.children)
            ]
            q.extend(n.children)
    return mapping

def build_docid_paths(tree, pid2docid):
    docid2path, leaf2docs = {}, {}
    total = 0

    def dfs(node, path):
        nonlocal total 
        
        cur = path + [node.node_id_int]
        if node.isleaf:
            if len(node.pids) > 0:
                leaf2docs[node.node_id_int] = []
                for pid in node.pids:
                    did = pid2docid.get(pid)
                    if did is not None: 
                        did_str = str(did)
                        docid2path.setdefault(did_str, []).append(cur)
                        leaf2docs[node.node_id_int].append(did_str)
                        total += 1
            return
        for c in node.children:
            dfs(c, cur)

    dfs(tree.root, [])
    print(f"Total paths generated: {total}")
    return docid2path, leaf2docs

# ==========================================================
#                           MAIN
# ==========================================================
if __name__ == "__main__":
    os.makedirs(TREE_DIR, exist_ok=True)

    # 从 config 读取 DATA_TYPE，默认为 'doc'
    DATA_TYPE = getattr(config, 'DATA_TYPE', 'doc')
    print(f"Data Type: {DATA_TYPE}")
    print(f"Loading IDs from {ID2OFFSET_PATH}...")

    if DATA_TYPE == "doc":
        id_map = pd.read_csv(ID2OFFSET_PATH, sep="\t")
        pid2docid = dict(zip(id_map["offset"], id_map["docid"]))
    elif DATA_TYPE == "passage":
        id_map = pd.read_csv(ID2OFFSET_PATH, sep="\t")
        pid2docid = dict(zip(id_map.index, id_map["pid"]))

    print(f"Loaded {len(pid2docid)} ID mappings.")

    X = np.memmap(MEMMAP_PATH, dtype=np.float32, mode="r").reshape(-1, EMBEDDING_DIM)
    
    if len(pid2docid) != X.shape[0]:
        print(f"[Warning] ID count ({len(pid2docid)}) != Memmap rows ({X.shape[0]}). Truncating to minimum.")
        valid_count = min(len(pid2docid), X.shape[0])
        pids = np.arange(valid_count)
    else:
        pids = np.arange(len(X))

    tree_init = TreeInitialize(X, pids, NODE_BALANCE, TREE_HEIGHT)
    tree = tree_init.clustering_tree()
    tree_init.assign_node_ids()

    save_object(tree_init, OUTPUT_TREE_PATH)
    save_object(build_children_map(tree_init), CHILDREN_EMBEDDINGS_PATH)

    docid2path, leaf2docs = build_docid_paths(tree_init, pid2docid)
    save_object(docid2path, ID2PATH)
    save_object(leaf2docs, LEAF2ID)

    print(f"Soft-GMM Tree build finished ({DATA_TYPE} mode).")