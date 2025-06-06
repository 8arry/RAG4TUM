# build_bm25.py
import json, pickle, pathlib, re, os
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCSTORE_DIR = ROOT / "data" / "embeddings"

# 1. Load LangChain vector store, get text + metadata
emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vectordb = FAISS.load_local(DOCSTORE_DIR, emb,
                            allow_dangerous_deserialization=True)
docs = list(vectordb.docstore._dict.values())

# 2. Build BM25 index
def tokenize(text):
    return re.findall(r"\w+", text.lower())

corpus_tokens = [tokenize(d.page_content) for d in docs]
bm25 = BM25Okapi(corpus_tokens)

# 3. Serialize and save
with open(f"{DOCSTORE_DIR}/bm25.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "docs": docs}, f)

print(f"✅ BM25 index saved ({len(docs)} docs)")
