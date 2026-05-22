DATA_TYPE = "passage"

MEMMAP_PATH = "./data/msmarco/passage/msmarco-passages-bge.memmap"
ID2OFFSET = "./data/msmarco/passage/msmarco-passages-bge_id2offset.tsv"
EMBEDDING_DIM = 768

PROJECT_ROOT = "."
TREE_DIR = "./tree/cluster_docs_tree"

OUTPUT_TREE_PATH = f"{TREE_DIR}/gmtree.pkl"
CHILDREN_EMBEDDINGS_PATH = f"{TREE_DIR}/children_id_embeddings.pkl"
ID2PATH = f"{TREE_DIR}/docs_id2path.pkl"
LEAF2ID = f"{TREE_DIR}/leaf2docs.pkl"

RANDOM_SEED = 42

TRAIN_QUERIES = "./data/msmarco/passage/queries.train.tsv"
TRAIN_QRELS = "./data/msmarco/passage/qrels.train.tsv"

OUTPUT_DIR = "./output"

MODEL_NAME = "./models/bge-base-en-v1.5"
POOLING = "cls"                     # cls or mean
MAX_SEQ_LEN = 512                   # you can change if needed

LEARNING_RATE1 = 1e-5
LEARNING_RATE2 = 5e-5
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 2000
GRAD_CLIP = 1.0

SAVE_INTERVAL = 1
BATCH_SIZE = 128

MAX_CANDIDATES = 100                # beam search pool
ROUTING_BEAM_SIZE = 20

RESUME = True
EPOCHS = 15

EVAL_TOPK = 100                     # retrieve top 100 documents for eval

DEV_QUERYS = "./data/msmarco/passage/queries.dev.small.tsv"
DEV_QRELS = "./data/msmarco/passage/qrels.dev.tsv"

TREC_QUERYS_20 = "./data/msmarco/passage/msmarco-pass-test2020-queries.tsv"
TREC_QUERL_20 = "./data/msmarco/passage/2020qrels-pass.txt"

TREC_QUERYS_19 = "./data/msmarco/passage/msmarco-test2019-queries.tsv"
TREC_QUERL_19 = "./data/msmarco/passage/2019qrels-pass.txt"

NEG_NUM = 127

PROB_THRESHOLD = 0.000077

NODE_BALANCE = 13

TREE_HEIGHT = 5