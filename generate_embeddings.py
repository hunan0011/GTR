import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import config


def get_input_path():
    if config.DATA_TYPE == "passage":
        return "./data/msmarco/passage/collection.tsv"
    if config.DATA_TYPE == "document":
        return "./data/msmarco/document/msmarco-docs.tsv"
    raise ValueError(f"Unsupported DATA_TYPE: {config.DATA_TYPE}")


def count_rows(tsv_path):
    return sum(1 for _ in open(tsv_path, "r", encoding="utf-8", errors="ignore"))


def load_chunks(tsv_path, chunk_size):
    if config.DATA_TYPE == "passage":
        return pd.read_csv(
            tsv_path,
            sep="\t",
            header=None,
            names=["docid", "text"],
            dtype=str,
            chunksize=chunk_size,
            on_bad_lines="skip",
        )

    return pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["docid", "url", "title", "body"],
        dtype=str,
        chunksize=chunk_size,
        on_bad_lines="skip",
    )


def prepare_texts(chunk_df):
    chunk_df["docid"] = chunk_df["docid"].astype(str).str.strip()

    if config.DATA_TYPE == "passage":
        texts = chunk_df["text"].fillna("").astype(str).str.strip().tolist()
        docids = chunk_df["docid"].tolist()
        return docids, texts

    title = chunk_df["title"].fillna("").astype(str).str.strip()
    body = chunk_df["body"].fillna("").astype(str).str.strip()

    texts = (title + " " + body).str.strip().tolist()
    docids = chunk_df["docid"].tolist()
    return docids, texts


def encode_batch(texts, tokenizer, model, device):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LEN,
    ).to(device)

    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().astype("float32")


def main():
    input_path = get_input_path()

    os.makedirs(os.path.dirname(config.MEMMAP_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(config.ID2OFFSET), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModel.from_pretrained(config.MODEL_NAME)
    model.to(device)
    model.eval()

    total_docs = count_rows(input_path)
    embedding_dim = model.config.hidden_size

    embeddings_memmap = np.memmap(
        config.MEMMAP_PATH,
        dtype="float32",
        mode="w+",
        shape=(total_docs, embedding_dim),
    )

    offset = 0
    chunk_size = 100000

    with open(config.ID2OFFSET, "w", encoding="utf-8") as mapping_file:
        mapping_file.write("docid\toffset\n")

        with torch.no_grad():
            for chunk_df in tqdm(load_chunks(input_path, chunk_size), total=(total_docs + chunk_size - 1) // chunk_size):
                docids, texts = prepare_texts(chunk_df)

                for i in range(0, len(texts), config.BATCH_SIZE):
                    batch_texts = texts[i:i + config.BATCH_SIZE]
                    batch_embeddings = encode_batch(batch_texts, tokenizer, model, device)

                    start = offset + i
                    end = start + len(batch_texts)
                    embeddings_memmap[start:end] = batch_embeddings

                for docid in docids:
                    mapping_file.write(f"{docid}\t{offset}\n")
                    offset += 1

                embeddings_memmap.flush()

    print(f"Saved embeddings to: {config.MEMMAP_PATH}")
    print(f"Saved id2offset to: {config.ID2OFFSET}")
    print(f"Total encoded documents: {offset}")


if __name__ == "__main__":
    main()