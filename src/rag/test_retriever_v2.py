#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_retriever_v2.py  ·  FAISS + BGE-reranker demo
-------------------------------------------------
Usage:
python test_retriever_v2.py \
    --index index/faiss  \
    --meta  data/vectors/tum_programs_metadata.json \
    --query "How to apply for Information Engineering Master program?" \
    --slug information-engineering \
    --degree msc \
    --top_k 8
"""

import argparse, json, faiss, numpy as np, re
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings


# ---------- load helpers ----------
def load_metadata(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def cosine_normalize(index_path, dim):
    index = faiss.read_index(index_path)
    # 若 index 类型不是 IP，需要转换
    if not isinstance(index, faiss.IndexFlatIP):
        raise ValueError("请用 IndexFlatIP 保存索引")
    return index

def load_vector_db(index_dir, meta):
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS(index=faiss.read_index(f"{index_dir}/tum_programs.index"),
                 docstore=None,
                 index_to_docstore_id=None,
                 embedding_function=emb,
                 normalize_L2=True,
                 documents=meta)
import pathlib
# ---------- main ----------
if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    DEFAULT_INDEX_DIR = ROOT / "data" / "embeddings"
    ap.add_argument("--index", default=str(DEFAULT_INDEX_DIR), help="向量库目录（含 index.faiss + index.pkl）")
    ap.add_argument("--query", required=True, help="查询问题")
    ap.add_argument("--slug",  help="程序 slug 过滤")
    ap.add_argument("--degree", help="学位过滤 msc/bsc/ba/ma")
    ap.add_argument("--category", help="info/apply/keydata")
    ap.add_argument("--top_k", type=int, default=8)
    args = ap.parse_args()

    # 1) 加载向量库 + 元数据
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = FAISS.load_local(
        args.index,     # 目录里有 index.faiss + index.pkl
        emb,
        allow_dangerous_deserialization=True   # ← 告诉 LangChain：我信任这个文件
    )

    # 2) 构造 filter
    def meta_filter(meta):
        if args.slug   and meta.get("slug")    != args.slug:   return False
        if args.degree and meta.get("degree")  != args.degree: return False
        if args.category and meta.get("category") != args.category: return False
        return True

    # 3) 初步向量召回
    dense_docs = vectordb.similarity_search_with_score(args.query, k=40)
    dense_docs = [(d, 1-s) for d, s in dense_docs]           # 转为“相似度”分
    dense_docs = [(d, s) for d, s in dense_docs if meta_filter(d.metadata)]

    if not dense_docs:
        print("❗ 没有候选候选片段，放宽过滤或检查 slug/degree")
        exit()

    # 4) Cross-Encoder 重排
    reranker = CrossEncoder("BAAI/bge-reranker-large")
    pairs = [[args.query, d.page_content] for d, _ in dense_docs]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, dense_docs), reverse=True)[:args.top_k]

    # 5) 打印
    print(f"\n查询: {args.query}\n" + "=" * 80 + "\n")
    print("搜索结果 (Top-{})".format(args.top_k))
    print("-" * 80)
    for rank, (score, (doc, sim)) in enumerate(reranked, 1):
        meta = doc.metadata
        print(f"\n结果 {rank}:")
        print(f"程序: {meta['program']} ({meta.get('degree','')})")
        print(f"类别: {meta['category']} · {meta['section']}")
        print(f"相似度(重排) : {score:.3f}   原始sim: {sim:.3f}")
        excerpt = re.sub(r'\s+', ' ', doc.page_content)[:400]
        print(f"内容: {excerpt}...")
        print("-" * 80)
