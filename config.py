'''
Author: “Mia” “welshcorgi@foxmail.com”
Date: 2026-01-25 15:57:39
LastEditors: “Mia” “welshcorgi@foxmail.com”
LastEditTime: 2026-01-28 23:15:25
FilePath: /jiangyutao/GTR/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
DATA_TYPE = "doc"
MEMMAP_PATH = "/home/power/jiangyutao/GTR/msmarco_docs_bge_embeddings.memmap"
ID2OFFSET = "/home/power/jiangyutao/GTR/msmarco_docs_bge_id2offset.tsv"
EMBEDDING_DIM = 768
NODE_BALANCE = 10
TREE_HEIGHT = 5
PROB_THRESHOLD = 0.0075
PROJECT_ROOT = "/home/power/jiangyutao/GTR"
TREE_DIR = f"{PROJECT_ROOT}/tree/cluster_docs_tree"
OUTPUT_TREE_PATH = f"{TREE_DIR}/gmtree.pkl"
CHILDREN_EMBEDDINGS_PATH = f"{TREE_DIR}/children_id_embeddings.pkl" 
ID2PATH = f"{TREE_DIR}/docs_id2path.pkl" 
LEAF2ID = f"{TREE_DIR}/leaf2docs.pkl"
RANDOM_SEED = 42



# TOP100_PATH = "/home/jiangda/jiangyutao/github/DRhard/data/doc/dataset/msmarco-doctrain-top100"
DOC_TRAIN_QRELS = "/home/power/jiangyutao/GTR/docs/msmarco-doctrain-qrels.tsv"
DOC_TRAIN_QUERIES = "/home/power/jiangyutao/GTR/docs/msmarco-doctrain-queries.tsv"
DOC_PATH = "/home/power/jiangyutao/GTR/docs/collection.tsv"
# DOC_LOOKUP = "/home/jiangda/jiangyutao/github/DRhard/data/doc/dataset/msmarco-docs-lookup.tsv"
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
DEV_DOC_TRAIN_QUERIES = "/home/power/jiangyutao/GTR/docs/msmarco-docdev-queries.tsv"
DEV_DOC_TRAIN_QRELS = "/home/power/jiangyutao/GTR/docs/msmarco-docdev-qrels.tsv"

BIAS_NUM = 1              # number of bias heads in DAAB


NEG_NUM = 63