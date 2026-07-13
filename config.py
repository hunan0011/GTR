DATA_TYPE = "doc"

MEMMAP_PATH = "/home/power/jiangyutao/msmarco_docs_bge_embeddings.memmap"
ID2OFFSET = "/home/power/jiangyutao/msmarco_docs_bge_id2offset.tsv"
EMBEDDING_DIM = 768

TREE_DIR = "./tree/cluster_docs_tree"

CHILDREN_EMBEDDINGS_PATH = f"{TREE_DIR}/children_id_embeddings.pkl"
ID2PATH = f"{TREE_DIR}/docs_id2path.pkl"
LEAF2ID = f"{TREE_DIR}/leaf2docs.pkl"

RANDOM_SEED = 42

TRAIN_QUERIES = "/home/power/jiangyutao/data/doc/dataset/msmarco-doctrain-queries.tsv"
TRAIN_QRELS = "/home/power/jiangyutao/data/doc/dataset/msmarco-doctrain-qrels.tsv"

OUTPUT_DIR = "./output"

MODEL_NAME = "./models/bge-base-en-v1.5"
POOLING = "cls"
MAX_SEQ_LEN = 512

LEARNING_RATE1 = 1e-5
LEARNING_RATE2 = 5e-5
WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0

SAVE_INTERVAL = 1
BATCH_SIZE = 128
EPOCHS = 15

DEV_QUERYS = "/home/power/jiangyutao/data/doc/dataset/msmarco-docdev-queries.tsv"
DEV_QRELS = "/home/power/jiangyutao/data/doc/dataset/msmarco-docdev-qrels.tsv"

TREC_QUERYS_20 = "/home/power/jiangyutao/data/doc/dataset/msmarco-doc-test2020-queries.tsv"
TREC_QUERL_20 = "/home/power/jiangyutao/data/doc/dataset/2020qrels-docs.txt"

TREC_QUERYS_19 = "/home/power/jiangyutao/data/doc/dataset/msmarco-test2019-queries.tsv"
TREC_QUERL_19 = "/home/power/jiangyutao/data/doc/dataset/2019qrels-docs.txt"

NEG_NUM = 127

PROB_THRESHOLD = 0.0001

NODE_BALANCE = 10
TREE_HEIGHT = 5