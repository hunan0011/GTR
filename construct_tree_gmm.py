# construct_tree_gmm.py
import os
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import config
import gc

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
# [修改 1] 获取数据类型
DATA_TYPE = getattr(config, 'DATA_TYPE', 'doc')

MEMMAP_PATH = config.MEMMAP_PATH
ID2OFFSET_PATH = config.ID2OFFSET
EMBEDDING_DIM = config.EMBEDDING_DIM
NODE_BALANCE = config.NODE_BALANCE
TREE_HEIGHT = config.TREE_HEIGHT
PROB_THRESHOLD = config.PROB_THRESHOLD

TREE_DIR = config.TREE_DIR
CHILDREN_EMBEDDINGS_PATH = config.CHILDREN_EMBEDDINGS_PATH
ID2PATH = config.ID2PATH
LEAF2ID = config.LEAF2ID

RANDOM_SEED = config.RANDOM_SEED

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"Config: DATA_TYPE={DATA_TYPE}")

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
        self.embedding = embedding      
        self.variance = None            
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
        self.embeddings = embeddings 
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
            batch_size=8152,
            trainer_params=dict(
                accelerator="gpu",
                devices=1,
                max_epochs=500,
                enable_progress_bar=False,
                logger=False
            )
        )

        gmm.fit(torch.from_numpy(X)) 
        probs = gmm.predict_proba(torch.from_numpy(X)).cpu().numpy()
        centers = l2_normalize(gmm.model_.means.cpu().numpy())
        variances = gmm.model_.covariances.cpu().numpy() + 1e-9

        return probs, centers, variances, K

    def _build(self, node, X, pids, layer):
        print(f"Layer {layer}, Node {node.val}, Samples {len(pids)}")

        if layer == self.H - 1:
            node.isleaf = True
            node.pids = pids
            self.leaf_dict[node.val] = node
            return node

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
                child = TreeNode(child_id, node.embedding, layer + 1)
                child.variance = np.ones_like(node.embedding)
                self._build(child, np.zeros((0, EMBEDDING_DIM), np.float32), np.array([], np.int64), layer + 1)
            else:
                idxs = np.asarray(idxs)
                child = TreeNode(child_id, centers[k], layer + 1)
                child.variance = variances[k]
                self._build(child, X[idxs], pids[idxs], layer + 1)
            
            child.parent = node
            node.add(child)

        return node

    def clustering_tree(self):
        root = TreeNode("0", layer=0)
        root.embedding = l2_normalize(np.mean(self.embeddings, axis=0, keepdims=True))[0]
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
                    variance=c.variance.tolist() if c.variance is not None else []
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

    print(f"Loading IDs from {ID2OFFSET_PATH} (Mode: {DATA_TYPE})...")
    id_map = pd.read_csv(ID2OFFSET_PATH, sep="\t")

    # [修改 2] 修复 KeyError: 'docid'
    pid2docid = {}
    
    if DATA_TYPE == "doc":
        if "docid" in id_map.columns:
            pid2docid = dict(zip(id_map["offset"], id_map["docid"].astype(str)))
        else:
            print("[Error] DATA_TYPE='doc' but 'docid' missing.")
            exit(1)
            
    elif DATA_TYPE == "passage":
        # 优先找 'pid'，其次找 'docid'
        target_col = 'pid' if 'pid' in id_map.columns else 'docid'
        
        if target_col not in id_map.columns:
             # 如果连 pid 都没有，尝试 fallback 到第一列
             print(f"[Warn] Column '{target_col}' not found. Using first column as ID.")
             target_col = id_map.columns[0]
        
        # 检查 offset
        if "offset" in id_map.columns:
            pid2docid = dict(zip(id_map["offset"], id_map[target_col].astype(str)))
        else:
            print("[Info] No 'offset' column, using DataFrame index.")
            pid2docid = dict(zip(id_map.index, id_map[target_col].astype(str)))
            
    else:
        print(f"[Error] Unknown DATA_TYPE: {DATA_TYPE}")
        exit(1)

    print(f"Loaded {len(pid2docid)} ID mappings.")

    X = np.memmap(MEMMAP_PATH, dtype=np.float32, mode="r").reshape(-1, EMBEDDING_DIM)
    
    if len(pid2docid) != X.shape[0]:
        print(f"[Warning] ID count ({len(pid2docid)}) != Memmap rows ({X.shape[0]}). Truncating to minimum.")
        valid_count = min(len(pid2docid), X.shape[0])
        pids = np.arange(valid_count)
        X_view = X[:valid_count]
    else:
        pids = np.arange(len(X))
        X_view = X

    # 1. 建树
    tree_init = TreeInitialize(X_view, pids, NODE_BALANCE, TREE_HEIGHT)
    tree = tree_init.clustering_tree()
    tree_init.assign_node_ids()

    # 2. 生成必要的映射文件
    print("--- Generating required auxiliary files ---")
    save_object(build_children_map(tree_init), CHILDREN_EMBEDDINGS_PATH)
    print(f"Saved children embeddings to {CHILDREN_EMBEDDINGS_PATH}")

    docid2path, leaf2docs = build_docid_paths(tree_init, pid2docid)
    save_object(docid2path, ID2PATH)
    save_object(leaf2docs, LEAF2ID)
    print(f"Saved mappings to {ID2PATH} and {LEAF2ID}")

    # 3. 不保存 gmtree.pkl
    print("\nSkipping saving full tree object (gmtree.pkl).")
    print(f"Soft-GMM Tree build finished ({DATA_TYPE} mode).")