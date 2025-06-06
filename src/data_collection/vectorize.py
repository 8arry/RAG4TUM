#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vectorize.py  ·  TUM program data vectorization
------------------------------------------
Read JSONL file →
  · Use Google Gemini for vectorization
  · Build FAISS index
  · Save vectors and index

Usage
-----
python vectorize.py \
    --in_file data/processed/tum_program_docs.jsonl \
    --out_dir data/vectors \
    --model models/embedding-001
"""
import os
import json
import argparse
import pathlib
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
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
                    help="Input JSONL file")
    ap.add_argument("--out_dir", default=WORKSPACE_ROOT / "data/embeddings",
                    help="Vector store output directory")
    ap.add_argument("--model",   default="models/embedding-001",
                    help="Google Gemini embedding model")
    args = ap.parse_args()

    print("🗂  Loading JSONL …")
    lc_docs = load_docs_to_lc(args.in_file)
    print(f"Loaded {len(lc_docs)} documents")

    print("🔄  Embedding & building FAISS store …")
    embedder = GoogleGenerativeAIEmbeddings(model=args.model, google_api_key=api_key)
    vectordb  = FAISS.from_documents(lc_docs, embedder)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(out_dir))
    print(f"✅ Vector store saved to {out_dir} "
          f"(files: index.faiss, index.pkl)")

if __name__ == "__main__":
    main() 