# GTR: GMM-based Tree Indexing for End-to-End Dense Retrieval

## Introduction

To improve the robustness and efficiency of tree-based dense retrieval, we propose GTR, which stands for GMM-based Tree Indexing for end-to-end dense Retrieval. GTR aims to alleviate the irreversible routing errors caused by hard partitioning and the representation mismatch between query encoders and tree-based indexes. To achieve this goal, GTR replaces deterministic clustering with a GMM-based probabilistic tree, where documents are assigned to multiple semantic branches according to posterior probabilities. However, probabilistic indexing alone cannot fully address the discrepancy between dynamic query embeddings and static tree nodes. Therefore, we further design a Dual-MLP routing mechanism to project queries and index nodes into a shared routing space. Based on these components, GTR jointly optimizes the query encoder, tree index, and routing module in an end-to-end manner, leading to a more robust effectiveness-efficiency trade-off for large-scale dense retrieval.

## Preparation

GTR is evaluated on the MS MARCO Passage Ranking and Document Ranking datasets. The passage collection contains 8,841,823 passages, while the document collection contains 3,213,835 documents. We use [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) as the dense encoder backbone for generating query and document representations. The official MS MARCO download links are available from the [MS MARCO ranking dataset page](https://microsoft.github.io/msmarco/Datasets.html).

### Dataset

To download the required MS MARCO files, run:

```bash
bash scripts/download_msmarco.sh
```

### Encoder

We use [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) as the dense encoder backbone. The model can be automatically downloaded through Hugging Face during embedding generation. It can also be downloaded manually by:

```bash
pip install -U huggingface_hub

huggingface-cli download BAAI/bge-base-en-v1.5 \
  --local-dir ./models/bge-base-en-v1.5 \
  --local-dir-use-symlinks False
```

## Running GTR

### Pipeline Overview

After preparing the MS MARCO datasets and the BGE encoder, GTR can be executed in four main steps: **embedding preparation**, **GMM-based tree construction**, **end-to-end training**, and **inference**.

### Embedding Preparation

Before constructing the tree index, encode the MS MARCO corpus into dense document embeddings using the BGE encoder:

```bash
python generate_embeddings.py
```

This step encodes the MS MARCO corpus into dense document embeddings and prepares the corresponding query embeddings for training and evaluation.

### Construct the GMM-based Tree Index

After obtaining the document embeddings, construct the GMM-based probabilistic tree index:

```bash
python construct_tree_gmm.py
```

This step builds the hierarchical GMM-based tree structure and assigns documents to leaf buckets according to posterior probabilities.

### Train GTR

After the tree index has been constructed, train the GTR model with the joint optimization objective:

```bash
python train.py
```

This step jointly optimizes the query encoder, tree node embeddings, and Dual-MLP routing module.

### Run Inference

Finally, run inference with the trained model and the constructed tree index:

```bash
python inference.py
```

This step performs hierarchical beam search over the GMM-based tree index and generates the final retrieval results.

### Hyperparameter Configuration

To test the impact of different hyperparameter settings, modify the corresponding values in `config.py` before running the pipeline. 
