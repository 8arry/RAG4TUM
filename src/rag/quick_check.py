# quick_check.py  —— 直接在 ipython / python -c 里跑
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Get the absolute path to the embeddings directory
current_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "data", "embeddings")

db = FAISS.load_local(embeddings_dir,
                      OpenAIEmbeddings(model="text-embedding-3-small"),
                      allow_dangerous_deserialization=True)

hits = [d for d in db.docstore._dict.values()
        if "information engineering" in d.metadata.get("program","").lower()]

print(len(hits), "chunks found")
print({d.metadata["slug"] for d in hits})
