# quick_check.py  —— 直接在 ipython / python -c 里跑
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2] # Assuming this script is in RAG4TUM/src/rag or similar
DOCSTORE_DIR = ROOT / "data" / "embeddings"

print(f"Loading FAISS index from: {DOCSTORE_DIR}")
db = FAISS.load_local(str(DOCSTORE_DIR),
                      OpenAIEmbeddings(model="text-embedding-3-small"),
                      allow_dangerous_deserialization=True)

print("FAISS index loaded.")

# Target metadata
TARGET_SLUG = "information-engineering-at-tum-campus-heilbronn"
TARGET_DEGREE = "msc"
TARGET_CATEGORY = "apply"

found_matching_docs = False
print(f"\nSearching for documents with metadata:\n  Slug: {TARGET_SLUG}\n  Degree: {TARGET_DEGREE}\n  Category: {TARGET_CATEGORY}\n")

for doc_id, doc in db.docstore._dict.items():
    metadata = doc.metadata
    if (
        metadata.get("slug", "").lower() == TARGET_SLUG
        and metadata.get("degree", "").lower() == TARGET_DEGREE
        and metadata.get("category", "").lower() == TARGET_CATEGORY
    ):
        print("\n---------------- MATCH FOUND -----------------")
        print(f"Document ID: {doc_id}")
        print(f"Metadata: {metadata}")
        print(f"Page Content:\n{doc.page_content}")
        print("---------------------------------------------")
        found_matching_docs = True

if not found_matching_docs:
    print("\nNo documents found matching all target criteria (slug, degree, AND category='apply').")

# Let's also check for the 'info' category for the same slug and degree to compare
print(f"\nSearching for documents with metadata:\n  Slug: {TARGET_SLUG}\n  Degree: {TARGET_DEGREE}\n  Category: info\n")
found_info_docs = False
for doc_id, doc in db.docstore._dict.items():
    metadata = doc.metadata
    if (
        metadata.get("slug", "").lower() == TARGET_SLUG
        and metadata.get("degree", "").lower() == TARGET_DEGREE
        and metadata.get("category", "").lower() == "info"
    ):
        print("\n---------------- INFO MATCH FOUND -----------------")
        print(f"Document ID: {doc_id}")
        print(f"Metadata: {metadata}")
        print(f"Page Content:\n{doc.page_content}")
        print("---------------------------------------------")
        found_info_docs = True

if not found_info_docs:
    print("\nNo documents found for slug='information-engineering', degree='msc', AND category='info'.") 