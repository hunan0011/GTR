### construct_tree_kmeans.py

- children_id_embeddings.pkl
  **结构说明**
  - 外层字典的 key 是父节点的整数 ID（node_id_int）。
  - 外层字典的 value 是一个长度为 NODE_BALANCE 的列表，表示该父节点的所有子节点信息。
  - 每个子节点信息是一个字典，包含：
    - "child_id"：子节点的整数 ID（node_id_int）。
    - "child_index"：子节点在父节点中的顺序索引（0 到 B-1）。
    - "embedding"：子节点的 embedding 向量，类型为 list[float]，长度为 EMBEDDING_DIM。

~~~ python
{
  0: [
    {"child_id": 1, "child_index": 0, "embedding": [0.1, 0.2, ..., 0.123]},
    {"child_id": 2, "child_index": 1, "embedding": [0.3, 0.4, ..., 0.456]},
    ...
  ],
  1: [...],
  ...
}
~~~

- docid2path.pkl
  **结构说明**
  - key 是文档的字符串 ID（即 docid）。
  - value 是一个整数列表，表示该文档从根节点到叶节点的完整路径。
  - 路径长度固定为 TREE_HEIGHT，包含根节点（ID 为 0）和叶节点。

~~~python
{
  "D123456": [0, 3, 7, 15, 31],
  "D789012": [0, 1, 5, 12, 25],
  ...
}
~~~

---

超参数影响：

对于H，和B单选择，尽量保证泛化能力，要保证尽量每一个叶子结点下文档数量超过300个

MSMARCO Passage H = 5 B = 13 k-mean :

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Passage H = 5 B = 13 P_threshold = 0.077 :

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Passage H = 5 B = 13 P_threshold = 0.039 :

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Passage H = 5 B = 13 P_threshold = 0.019 :

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Passage H = 5 B = 13 P_threshold = 0.010 :

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Doc H = 5 B = 10  k-mean:

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Doc H = 5 B = 10 P_threshold = 0.1 :

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Doc H = 5 B = 10 P_threshold =0.05 :

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Doc H = 5 B = 10 P_threshold = 0.025:

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

MSMARCO Doc H = 5 B = 10 P_threshold = 0.0125:

| BEAM_SIZE | MRR@100 | Recall@100 | AQT (ms/q) |
| --------- | ------- | ---------- | ---------- |
| 10        |         |            |            |
| 20        |         |            |            |
| 30        |         |            |            |
| 40        |         |            |            |
| 50        |         |            |            |

存储空间开销比例：

| P_theshold | 比例大小 |
| ---------- | -------- |
| 原始       | 1        |
| 0.1        |   1.12       |
| 0.05       |          |
| 0.025      |          |
| 0.0125     |          |

存储空间开销比例：

| P_theshold | 比例大小 |
| ---------- | -------- |
| 原始       | 1        |
| 0.1        |          |
| 0.05       |          |
| 0.025      |          |
| 0.0125     |          |

---

消融实验：

| 方法               | MRR@100 | Recall@100 |
| ------------------ | ------- | ---------- |
| 普通聚类树         |         |            |
| 端到端联合优化树   |         |            |
| 传统的单聚类训练树 |         |            |
| 本方法+MLP         |         |            |

---

对比实验 MsMarco Doc Passage：

| 方法    | Recall@100 | MRR@100 | AQT  |
| ------- | ---------- | ------- | ---- |
| IVFPQ   |            |         |      |
| IVFFlat |            |         |      |
| HNSW    |            |         |      |
| ...     |            |         |      |
| ...     |            |         |      |
| JPQ     |            |         |      |
| JTR     |            |         |      |

