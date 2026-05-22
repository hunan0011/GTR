#!/bin/bash

set -e

DATA_DIR="./data/msmarco"

PASSAGE_DIR="${DATA_DIR}/passage"
DOCUMENT_DIR="${DATA_DIR}/document"

mkdir -p "${PASSAGE_DIR}"
mkdir -p "${DOCUMENT_DIR}"

echo "=========================================="
echo "Downloading MS MARCO Passage Ranking files"
echo "=========================================="

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz \
  -O "${PASSAGE_DIR}/collection.tar.gz"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz \
  -O "${PASSAGE_DIR}/queries.tar.gz"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv \
  -O "${PASSAGE_DIR}/qrels.train.tsv"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv \
  -O "${PASSAGE_DIR}/qrels.dev.tsv"

echo "Extracting MS MARCO Passage Ranking files"

tar -xzvf "${PASSAGE_DIR}/collection.tar.gz" -C "${PASSAGE_DIR}"
tar -xzvf "${PASSAGE_DIR}/queries.tar.gz" -C "${PASSAGE_DIR}"

echo "==========================================="
echo "Downloading MS MARCO Document Ranking files"
echo "==========================================="

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz \
  -O "${DOCUMENT_DIR}/msmarco-docs.tsv.gz"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-queries.tsv.gz \
  -O "${DOCUMENT_DIR}/msmarco-doctrain-queries.tsv.gz"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-doctrain-qrels.tsv.gz \
  -O "${DOCUMENT_DIR}/msmarco-doctrain-qrels.tsv.gz"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-queries.tsv.gz \
  -O "${DOCUMENT_DIR}/msmarco-docdev-queries.tsv.gz"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-docdev-qrels.tsv.gz \
  -O "${DOCUMENT_DIR}/msmarco-docdev-qrels.tsv.gz"

echo "Extracting MS MARCO Document Ranking files"

gunzip -kf "${DOCUMENT_DIR}/msmarco-docs.tsv.gz"
gunzip -kf "${DOCUMENT_DIR}/msmarco-doctrain-queries.tsv.gz"
gunzip -kf "${DOCUMENT_DIR}/msmarco-doctrain-qrels.tsv.gz"
gunzip -kf "${DOCUMENT_DIR}/msmarco-docdev-queries.tsv.gz"
gunzip -kf "${DOCUMENT_DIR}/msmarco-docdev-qrels.tsv.gz"

echo "=========================================="
echo "Downloading TREC DL Passage files"
echo "=========================================="

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz \
  -O "${PASSAGE_DIR}/msmarco-test2019-queries.tsv.gz"

wget -c https://trec.nist.gov/data/deep/2019qrels-pass.txt \
  -O "${PASSAGE_DIR}/2019qrels-pass.txt"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz \
  -O "${PASSAGE_DIR}/msmarco-pass-test2020-queries.tsv.gz"

wget -c https://trec.nist.gov/data/deep/2020qrels-pass.txt \
  -O "${PASSAGE_DIR}/2020qrels-pass.txt"

echo "Extracting TREC DL Passage query files"

gunzip -kf "${PASSAGE_DIR}/msmarco-test2019-queries.tsv.gz"
gunzip -kf "${PASSAGE_DIR}/msmarco-pass-test2020-queries.tsv.gz"

echo "=========================================="
echo "Downloading TREC DL Document files"
echo "=========================================="

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz \
  -O "${DOCUMENT_DIR}/msmarco-test2019-queries.tsv.gz"

wget -c https://trec.nist.gov/data/deep/2019qrels-docs.txt \
  -O "${DOCUMENT_DIR}/2019qrels-docs.txt"

wget -c https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz \
  -O "${DOCUMENT_DIR}/msmarco-test2020-queries.tsv.gz"

wget -c https://trec.nist.gov/data/deep/2020qrels-docs.txt \
  -O "${DOCUMENT_DIR}/2020qrels-docs.txt"

echo "Extracting TREC DL Document query files"

gunzip -kf "${DOCUMENT_DIR}/msmarco-test2019-queries.tsv.gz"
gunzip -kf "${DOCUMENT_DIR}/msmarco-test2020-queries.tsv.gz"

echo "=========================================="
echo "MS MARCO and TREC DL download completed."
echo "Passage data path: ${PASSAGE_DIR}"
echo "Document data path: ${DOCUMENT_DIR}"
echo "=========================================="