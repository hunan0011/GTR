import os
DATA_TYPE = "passage"
MEMMAP_PATH = "/home/jiangda/jiangyutao/Code/msmarco-passages-bge.memmap"
ID2OFFSET = "/home/jiangda/jiangyutao/Code/msmarco-passages-bge_id2offset.tsv"
EMBEDDING_DIM = 768
NODE_BALANCE = 13
TREE_HEIGHT = 5
PROB_THRESHOLD = 0.039
PROJECT_ROOT = "/home/jiangda/jiangyutao/Code"
TREE_DIR = f"{PROJECT_ROOT}/tree/cluster_passages_tree"
OUTPUT_TREE_PATH = f"{TREE_DIR}/gmtree.pkl"
CHILDREN_EMBEDDINGS_PATH = f"{TREE_DIR}/children_id_embeddings.pkl" 
ID2PATH = f"{TREE_DIR}/passages_id2path.pkl" 
LEAF2ID = f"{TREE_DIR}/leaf2passages.pkl"
RANDOM_SEED = 42



# TOP100_PATH = "/home/jiangda/jiangyutao/github/DRhard/data/doc/dataset/msmarco-doctrain-top100"
DOC_TRAIN_QRELS = "/home/jiangda/jiangyutao/Code/passages/qrels.train.tsv"
DOC_TRAIN_QUERIES = "/home/jiangda/jiangyutao/Code/passages/queries.train.cleaned.tsv"
DOC_PATH = "/home/jiangda/jiangyutao/Code/passages/collection.tsv"
# DOC_LOOKUP = "/home/jiangda/jiangyutao/github/DRhard/data/doc/dataset/msmarco-docs-lookup.tsv"
OUTPUT_DIR = "./output"

MODEL_NAME = "/home/jiangda/jiangyutao/Code/bge-base-en-v1.5"
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
EPOCHS = 13

EVAL_TOPK = 100           # retrieve top 100 documents for eval
DEV_DOC_TRAIN_QUERIES = "/home/jiangda/jiangyutao/Code/passages/queries.dev.small.tsv"
DEV_DOC_TRAIN_QRELS = "/home/jiangda/jiangyutao/Code/passages/qrels.dev.small.tsv"

BIAS_NUM = 1              # number of bias heads in DAAB


NEG_NUM = 63