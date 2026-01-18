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