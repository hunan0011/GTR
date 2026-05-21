'''
Author: “Mia” “welshcorgi@foxmail.com”
Date: 2026-04-26 19:54:57
LastEditors: “Mia” “welshcorgi@foxmail.com”
LastEditTime: 2026-05-09 17:30:24
FilePath: /jiangyutao/GTR/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
DATA_TYPE = "passage"
MEMMAP_PATH = "/home/power/jiangyutao/GTR/msmarco-passages-bge.memmap"
ID2OFFSET = "/home/power/jiangyutao/GTR/msmarco-passages-bge_id2offset.tsv"
EMBEDDING_DIM = 768

PROJECT_ROOT = "/home/power/jiangyutao/GTR"
TREE_DIR = f"{PROJECT_ROOT}/tree/cluster_docs_tree"
OUTPUT_TREE_PATH = f"{TREE_DIR}/gmtree.pkl"
CHILDREN_EMBEDDINGS_PATH = f"{TREE_DIR}/children_id_embeddings.pkl" 
ID2PATH = f"{TREE_DIR}/docs_id2path.pkl" 
LEAF2ID = f"{TREE_DIR}/leaf2docs.pkl"
RANDOM_SEED = 42



DOC_TRAIN_QUERIES = "/home/power/jiangyutao/GTR/passage/dataset/queries.train.tsv"
DOC_TRAIN_QRELS = "/home/power/jiangyutao/GTR/passage/dataset/qrels.train.tsv"
OUTPUT_DIR = "./output"

MODEL_NAME = "/home/power/jiangyutao/GTR/bge-base-en-v1.5" 
POOLING = "cls"                     # cls or mean
MAX_SEQ_LEN = 512                   # you can change if needed

LEARNING_RATE1 = 1e-5
LEARNING_RATE2 = 5e-5
WEIGHT_DECAY = 1e-3
WARMUP_STEPS = 2000
NUM_EPOCHS = 3
GRAD_CLIP = 1.0

SAVE_INTERVAL = 1
BATCH_SIZE = 128


MAX_CANDIDATES = 100   # beam search pool
ROUTING_BEAM_SIZE = 20

RESUME=True
EPOCHS = 15

EVAL_TOPK = 100           # retrieve top 100 documents for eval

DEV_PASSAGE_QUERYS = "/home/power/jiangyutao/GTR/passage/dataset/queries.dev.small.tsv"
DEV_PASSAGE_QRELS = "/home/power/jiangyutao/GTR/passage/dataset/qrels.dev.small.tsv"


TREC_QUERYS_20= "/home/power/jiangyutao/GTR/passage/dataset/msmarco-pass-test2020-queries.tsv"
TREC_QUERL_20 = "/home/power/jiangyutao/GTR/passage/dataset/2020qrels-pass.txt"

TREC_QUERYS_19= "/home/power/jiangyutao/GTR/passage/dataset/msmarco-test2019-queries.tsv"
TREC_QUERL_19 = "/home/power/jiangyutao/GTR/passage/dataset/2019qrels-pass.txt"

BIAS_NUM = 1              # number of bias heads in DAAB


NEG_NUM = 127

PROB_THRESHOLD = 0.000077

NODE_BALANCE = 13

TREE_HEIGHT = 5
