#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vectorize.py  ·  TUM program data vectorization
------------------------------------------
读取 JSONL 文件 →
  · 使用 OpenAI 进行向量化
  · 使用 FAISS 建立索引
  · 保存向量和索引

Usage
-----
python vectorize.py \
    --in_file data/processed/tum_program_docs.jsonl \
    --out_dir data/vectors \
    --model text-embedding-3-small
"""
import os
import json
import argparse
import pathlib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Get the workspace root directory (2 levels up from this script)


def load_docs_to_lc(in_file):
    """JSONL → List[langchain.docstore.document.Document]"""
    docs = []
    with open(in_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(
                page_content=obj["text"],
                metadata={**obj["metadata"], "id": obj["id"]}
            ))
    return docs

def main():
    WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", default=WORKSPACE_ROOT / "data/processed/tum_program_docs.jsonl",
                    help="输入 JSONL")
    ap.add_argument("--out_dir", default=WORKSPACE_ROOT / "data/embeddings",
                    help="向量库输出目录")
    ap.add_argument("--model",   default="text-embedding-3-small",
                    help="OpenAI 嵌入模型")
    args = ap.parse_args()

    print("🗂  Loading JSONL …")
    lc_docs = load_docs_to_lc(args.in_file)
    print(f"Loaded {len(lc_docs)} documents")

    print("🔄  Embedding & building FAISS store …")
    embedder = OpenAIEmbeddings(model=args.model)
    vectordb  = FAISS.from_documents(lc_docs, embedder)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(out_dir))
    print(f"✅ Vector store saved to {out_dir} "
          f"(files: index.faiss, index.pkl)")

if __name__ == "__main__":
    main() 