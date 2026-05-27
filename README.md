# GTR: GMM-based Tree Indexing for End-to-End Dense Retrieval

## Introduction

GTR is an experimental dense retrieval pipeline based on GMM-based tree indexing. It uses a probabilistic tree so that documents can be assigned to multiple semantic branches according to posterior probabilities, rather than relying on a single hard partition. The model also uses a Dual-MLP routing module to project query embeddings and index-node embeddings into a shared routing space. The query encoder, tree index, and routing module are trained together for tree-based dense retrieval.

## Preparation

GTR is evaluated on the MS MARCO Passage Ranking and Document Ranking datasets. The passage collection contains 8,841,823 passages, and the document collection contains 3,213,835 documents. The dense encoder backbone is [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5). Official MS MARCO downloads are available from the [MS MARCO ranking dataset page](https://microsoft.github.io/msmarco/Datasets.html).

### Dataset

To download the required MS MARCO files, run:

```bash
bash scripts/download_msmarco.sh
```

### Encoder

[BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) is used as the dense encoder backbone. The model can be downloaded automatically through Hugging Face during embedding generation, or manually with:

```bash
pip install -U huggingface_hub

huggingface-cli download BAAI/bge-base-en-v1.5 \
  --local-dir ./models/bge-base-en-v1.5 \
  --local-dir-use-symlinks False
```

## Running GTR

### Pipeline Overview

After preparing the MS MARCO datasets and the BGE encoder, run the pipeline in four stages: **embedding preparation**, **GMM-based tree construction**, **end-to-end training**, and **inference**.

### Embedding Preparation

Before constructing the tree index, encode the MS MARCO corpus into dense document embeddings using the BGE encoder:

```bash
python generate_embeddings.py
```

The script encodes the MS MARCO corpus into dense document embeddings and prepares query embeddings for training and evaluation.

### Construct the GMM-based Tree Index

After obtaining the document embeddings, construct the GMM-based probabilistic tree index:

```bash
python construct_tree_gmm.py
```

The script builds the hierarchical GMM-based tree and assigns documents to leaf buckets according to posterior probabilities.

### Train GTR

After the tree index has been constructed, train the GTR model with the joint optimization objective:

```bash
python train.py
```

Training optimizes the query encoder, tree node embeddings, and Dual-MLP routing module.

### Run Inference

Run inference with the trained model and the constructed tree index:

```bash
python inference.py
```

Inference runs hierarchical beam search over the GMM-based tree index and writes the retrieval results.

### Hyperparameter Configuration

To test the impact of different hyperparameter settings, modify the corresponding values in `config.py` before running the pipeline. 
