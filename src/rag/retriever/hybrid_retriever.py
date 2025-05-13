import pickle, re, pathlib, yaml
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class HybridRetriever:
    def __init__(self, cfg_path: str = "../config/retriever.yaml"):
        # 获取项目根目录（src 的父目录）
        root_dir = pathlib.Path(__file__).parent.parent.parent.parent
        # 读取配置
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        # 将相对路径转换为绝对路径（相对于项目根目录）
        index_dir = root_dir / "data" / "embeddings"
        cfg["index_dir"] = str(index_dir)
        self.cfg = cfg
        
        # dense vectordb
        self.emb = OpenAIEmbeddings(model=cfg["embed_model"])
        self.vdb = FAISS.load_local(cfg["index_dir"], self.emb,
                                    allow_dangerous_deserialization=True)
        # bm25
        with open(pathlib.Path(cfg["index_dir"]) / "bm25.pkl", "rb") as f:
            pack = pickle.load(f)
        self.bm25: BM25Okapi = pack["bm25"]
        self.docs             = pack["docs"]
        # reranker
        self.rerank = CrossEncoder(cfg["reranker_model"])

    # -------- helpers --------
    @staticmethod
    def _tokenize(text: str):
        return re.findall(r"\w+", text.lower())

    def _bm25_search(self, query: str, k: int) -> List[Tuple[object, float]]:
        tok = self._tokenize(query)
        scores = self.bm25.get_scores(tok)
        idxs = scores.argsort()[-k:][::-1]
        mx = scores.max() or 1
        return [(self.docs[i], scores[i] / mx) for i in idxs]

    # -------- main entry --------
    def retrieve(self, query: str, filters: Dict[str, str]):
        """
        Hybrid 检索：Dense + BM25 → Filter → Cross-Encoder 重排
        returns: List[Tuple[ce_score, (Document, base_score)]]
        """

        # 1) Dense 召回
        dense = self.vdb.similarity_search_with_score(query, k=self.cfg["n_dense"])
        dense = [(d, 1 - s) for d, s in dense]           # 距离→相似度 0-1

        # 2) Sparse 召回
        sparse = self._bm25_search(query, self.cfg["n_sparse"])

        # 3) Merge 取最大分
        merged: Dict[str, Tuple[object, float]] = {}
        for doc, score in dense + sparse:
            key = doc.metadata.get("id") or id(doc)      # 首选 id；否则对象地址
            if key not in merged or score > merged[key][1]:
                merged[key] = (doc, score)

        # ---------- DEBUG ----------
        print(f"—— DEBUG ——\ndense raw: {len(dense)}  sparse raw: {len(sparse)}")
        print("merged raw:", len(merged))

        # 4) Filter
        def ok(meta):
            # slug ⊂
            if filters["slug"] and filters["slug"].lower() not in meta.get("slug", "").lower():
                return False
            # degree ==
            if filters["degree"] and filters["degree"].lower() != meta.get("degree", "").lower():
                return False
            # category in {..}
            if filters["category"]:
                wanted = {c.strip().lower() for c in filters["category"].split(",")}
                if meta.get("category", "").lower() not in wanted:
                    return False
            return True

        merged = [v for v in merged.values() if ok(v[0].metadata)]
        if not merged:
            print("after filter: 0  ❗ no candidate left")
            return []

        print("after filter:", len(merged),
            " first 3 cats:", [v[0].metadata.get("category") for v in merged[:3]])

        # 5) Cross-Encoder 重排
        top_m = min(self.cfg.get("top_m", 20), len(merged))
        pairs  = [[query, doc.page_content] for doc, _ in merged[:top_m]]
        logits = self.rerank.predict(pairs, batch_size=8, convert_to_numpy=True)

        reranked = sorted(zip(logits, merged[:top_m]), reverse=True)
        top_k = self.cfg.get("top_k", 8)
        results = reranked[:top_k]

        print(f"to rerank: {top_m}   rerank scores: {len(logits)}   returning: {len(results)}")
        # ---------- END DEBUG ----------

        return results

