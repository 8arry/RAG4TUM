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
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Tuple
import time
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# Get the workspace root directory (2 levels up from this script)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class DocumentVectorizer:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.dimension = 1536  # OpenAI text-embedding-3-small 的维度

    def get_embedding(self, text: str) -> np.ndarray:
        """获取单个文本的向量表示"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def batch_get_embeddings(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """批量获取文本的向量表示"""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            current_batch = (i // batch_size) + 1
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [np.array(item.embedding, dtype=np.float32) 
                                  for item in response.data]
                embeddings.extend(batch_embeddings)
                print(f"Processed batch {current_batch}/{total_batches} ({i + len(batch)}/{len(texts)} documents)")
                time.sleep(1)  # 避免 API 限制
            except Exception as e:
                print(f"Error processing batch {current_batch}/{total_batches}: {e}")
        return embeddings

def load_documents(file_path: str) -> Tuple[List[Dict], List[str]]:
    """加载 JSONL 文件中的文档"""
    documents = []
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
            texts.append(doc['text'])
    return documents, texts

def create_faiss_index(embeddings: List[np.ndarray], dimension: int) -> faiss.Index:
    """创建 FAISS 索引"""
    vecs = np.vstack(embeddings).astype("float32")
    # 归一化 → cosine
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    index = faiss.IndexFlatIP(dimension)
    index.add(vecs)
    return index

def save_index_and_docs(index: faiss.Index, documents: List[Dict], out_dir: str):
    """保存索引和文档"""
    os.makedirs(out_dir, exist_ok=True)
    
    # 保存 FAISS 索引
    faiss.write_index(index, os.path.join(out_dir, "tum_programs.index"))
    
    # 保存文档元数据
    with open(os.path.join(out_dir, "tum_programs_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", 
                       default=os.path.join(WORKSPACE_ROOT, "data", "processed", "tum_program_docs.jsonl"),
                       help="输入 JSONL 文件路径")
    parser.add_argument("--out_dir", 
                       default=os.path.join(WORKSPACE_ROOT, "data", "embeddings"),
                       help="输出目录")
    parser.add_argument("--model", 
                       default="text-embedding-3-small",
                       help="OpenAI 模型名称")
    args = parser.parse_args()

    # 加载文档
    print("Loading documents...")
    documents, texts = load_documents(args.in_file)
    print(f"Loaded {len(documents)} documents")

    # 初始化向量化器
    vectorizer = DocumentVectorizer(model_name=args.model)

    # 获取向量表示
    print("Getting embeddings...")
    embeddings = vectorizer.batch_get_embeddings(texts)
    print(f"Got {len(embeddings)} embeddings")

    # 创建 FAISS 索引
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings, vectorizer.dimension)
    print(f"Index created with {index.ntotal} vectors")

    # 保存索引和文档
    print("Saving index and documents...")
    save_index_and_docs(index, documents, args.out_dir)
    print(f"Saved to {args.out_dir}")

if __name__ == "__main__":
    main() 