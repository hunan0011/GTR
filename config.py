
import os
DATA_TYPE = "passage"
MEMMAP_PATH = "/home/power/jiangyutao/GTR/msmarco-passages-bge.memmap"
ID2OFFSET = "/home/power/jiangyutao/GTR/msmarco-passages-bge_id2offset.tsv"
EMBEDDING_DIM = 768
NODE_BALANCE = 13
TREE_HEIGHT = 5
PROB_THRESHOLD = 0.077
PROJECT_ROOT = "/home/power/jiangyutao/GTR"
TREE_DIR = f"{PROJECT_ROOT}/tree/cluster_docs_tree"
OUTPUT_TREE_PATH = f"{TREE_DIR}/gmtree.pkl"
CHILDREN_EMBEDDINGS_PATH = f"{TREE_DIR}/children_id_embeddings.pkl" 
ID2PATH = f"{TREE_DIR}/docs_id2path.pkl" 
LEAF2ID = f"{TREE_DIR}/leaf2docs.pkl"
RANDOM_SEED = 42



DOC_TRAIN_QRELS = "/home/power/jiangyutao/GTR/passages/qrels.train.tsv"
DOC_TRAIN_QUERIES = "/home/power/jiangyutao/GTR/passages/queries.train.cleaned.tsv"
OUTPUT_DIR = "./output"

MODEL_NAME = "/home/power/jiangyutao/GTR/bge-base-en-v1.5"
POOLING = "cls"                     # cls or mean
MAX_SEQ_LEN = 512                   # you can change if needed

LEARNING_RATE1 = 1e-5
LEARNING_RATE2 = 1e-5
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 2000
NUM_EPOCHS = 3
GRAD_CLIP = 1.0

SAVE_INTERVAL = 1
MODEL_SAVE_PATH = "/home/jiangda/jiangyutao/Code/output/checkpoint-final.pt"
BATCH_SIZE = 32


MAX_CANDIDATES = 100   # beam search pool
ROUTING_BEAM_SIZE = 20

RESUME=True
EPOCHS = 15

EVAL_TOPK = 100           # retrieve top 100 documents for eval

DEV_PASSAGE_QUERYS = "/home/power/jiangyutao/GTR/passages/queries.dev.small.tsv"
DEV_PASSAGE_QRELS = "/home/power/jiangyutao/GTR/passages/qrels.dev.small.tsv"

BIAS_NUM = 1              # number of bias heads in DAAB


NEG_NUM = 63