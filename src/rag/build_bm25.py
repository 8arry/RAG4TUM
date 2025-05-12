# build_bm25.py
import json, pickle, pathlib, re
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCSTORE_DIR = ROOT / "data" / "embeddings"

# 1. 读取 LangChain 向量库，获取文本 + metadata
emb = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = FAISS.load_local(DOCSTORE_DIR, emb,
                            allow_dangerous_deserialization=True)
docs = list(vectordb.docstore._dict.values())

# 2. 构建 BM25
def tokenize(text):
    return re.findall(r"\w+", text.lower())

corpus_tokens = [tokenize(d.page_content) for d in docs]
bm25 = BM25Okapi(corpus_tokens)

# 3. 序列化
with open(f"{DOCSTORE_DIR}/bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "docs": docs}, f)

print(f"✅ BM25 index saved ({len(docs)} docs)")
