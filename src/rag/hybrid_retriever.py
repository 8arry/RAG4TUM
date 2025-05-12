# hybrid_retriever.py
import argparse, pickle, re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--index_dir", default=str(ROOT / "data" / "embeddings"))
ap.add_argument("--query", required=True)
ap.add_argument("--top_k", type=int, default=8)
ap.add_argument("--slug", "--program")
ap.add_argument("--degree")
args = ap.parse_args()

# ---------- Load dense vectordb ----------
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.load_local(args.index_dir, emb,
                            allow_dangerous_deserialization=True)

# ---------- Load BM25 ----------
with open(f"{args.index_dir}/bm25.pkl", "rb") as f:
    bm25_pack = pickle.load(f)
bm25   = bm25_pack["bm25"]
docs   = bm25_pack["docs"]

def meta_ok(meta):
    if args.slug and meta.get("slug") != args.slug: return False
    if args.degree and meta.get("degree") != args.degree: return False
    return True

# ---------- Dense 召回 ----------
dense = vectordb.similarity_search_with_score(args.query, k=40)
dense = [(d, 1-s) for d, s in dense if meta_ok(d.metadata)]  # sim=1-dist

# ---------- BM25 召回 ----------
tokens = re.findall(r"\w+", args.query.lower())
bm25_scores = bm25.get_scores(tokens)
top_idx = bm25_scores.argsort()[-40:][::-1]
sparse = [(docs[i], bm25_scores[i]/bm25_scores.max())  # 0-1 归一
          for i in top_idx if meta_ok(docs[i].metadata)]

# ---------- 合并去重 ----------
merged = {}
for doc, score in dense + sparse:
    key = doc.metadata["id"] if "id" in doc.metadata else id(doc)
    merged[key] = (doc, max(score, merged.get(key, (None, 0))[1]))

merged_docs = list(merged.values())

# ---------- Cross-Encoder 重排 ----------
reranker = CrossEncoder("BAAI/bge-reranker-large")
pairs    = [[args.query, d.page_content] for d, _ in merged_docs]
ce_scores = reranker.predict(pairs)

reranked = sorted(zip(ce_scores, merged_docs), reverse=True)[:args.top_k]

# ---------- 输出 ----------
print(f"\nQuery: {args.query}\n" + "="*60)
for rk, (rscore, (doc, base_score)) in enumerate(reranked, 1):
    m = doc.metadata
    excerpt = re.sub(r"\s+", " ", doc.page_content)[:300]
    print(f"\n#{rk}  CE {rscore:.3f}  dense/BM25 {base_score:.3f}")
    print(f"{m.get('program')} ({m.get('degree')}) · {m.get('category')} · {m.get('section')}")
    print(excerpt, "…")
