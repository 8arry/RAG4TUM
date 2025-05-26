#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_corpus_v2.py   ·   TUM program ETL (improved)
-----------------------------------------------------
• data/raw/*.json  ->  data/processed/tum_program_docs.jsonl
• Improvements:
    1. KeyData independent chunks
    2. Main text 256-token sliding window (stride 128)
    3. Chunk header keeps "INFO · Program Structure" title
    4. Metadata contains slug / degree
    5. Deduplication key (slug, section, chunk_idx)
"""

from __future__ import annotations
import argparse, json, re, uuid, glob, pathlib, os
from typing import List, Dict, Generator
import tiktoken
from urllib.parse import urlparse
from slugify import slugify

enc = tiktoken.get_encoding("cl100k_base")      # GPT-4 series tokenizer

# ─────────── utils ────────────
WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WS.sub(" ", text).strip()


def slug_and_degree(url: str | None, prog_name: str, key_data: Dict) -> tuple[str, str]:
    """
    If url exists → extract degree abbreviation from slug;
    If url missing → use program_name slug; guess degree from key_data.admission_category.
    """
    if url:
        path = urlparse(url).path.rstrip("/")
        slug_full = path.split("/")[-1]              # e.g. aerospace-master-of-science-msc
        parts = slug_full.split("-")
        degree = parts[-1].lower() if len(parts) >= 1 else ""
        prog_slug = "-".join(parts[:-4]) or "-".join(parts[:-1])
        return prog_slug, degree

    # ---- Fallback: derive from program_name ----
    prog_slug = slugify(prog_name, lowercase=True)
    # Try to extract degree from admission_category
    ac = key_data.get("admission_category", "").lower()
    if "master" in ac:
        degree = "msc"
    elif "bachelor" in ac:
        degree = "bsc"
    elif "doctor" in ac or "phd" in ac:
        degree = "phd"
    else:
        degree = ""
    return prog_slug, degree


def token_chunks(text: str, max_tks: int = 256, stride: int = 128):
    ids = enc.encode(text)
    for start in range(0, len(ids), stride):
        chunk_ids = ids[start:start + max_tks]
        if not chunk_ids:
            break
        yield enc.decode(chunk_ids)


def keydata_chunks(rec: Dict) -> Generator[Dict, None, None]:
    slug, deg = slug_and_degree(rec.get("url"), rec["program_name"], rec.get("key_data", {}))
    base_meta = {"program": rec["program_name"], "slug": slug, "degree": deg,
                 "category": "keydata"}
    for k, v in rec.get("key_data", {}).items():
        if not v:
            continue
        if isinstance(v, dict):
            txt = "; ".join(f"{kk}: {vv}" for kk, vv in v.items())
        elif isinstance(v, list):
            txt = ", ".join(map(str, v))
        else:
            txt = str(v)
        yield {
            "id": uuid.uuid4().hex,
            "text": f"{k.replace('_', ' ').title()}: {txt}",
            "metadata": {**base_meta, "section": k}
        }


def iter_chunks(rec: Dict) -> Generator[Dict, None, None]:
    slug, deg = slug_and_degree(rec.get("url"), rec["program_name"], rec.get("key_data", {}))

    def base_meta(cat, sec):
        return {"program": rec["program_name"], "slug": slug, "degree": deg,
                "category": cat, "section": sec}

    # ----- Description -----
    if rec.get("program_description"):
        yield {
            "id": uuid.uuid4().hex,
            "text": f"DESC · Overview\n\n{normalize(rec['program_description'])}",
            "metadata": base_meta("desc", "overview")
        }

    # ----- Long text blocks -----
    for block, cat in [("information_on_degree_program", "info"),
                       ("application_and_admission", "apply")]:
        for sec, payload in rec.get(block, {}).items():
            raw = normalize(payload.get("text", ""))
            links = payload.get("links", [])
            if not raw:
                continue
            pieces = list(token_chunks(raw))
            for i, piece in enumerate(pieces):
                yield {
                    "id": uuid.uuid4().hex,
                    "text": f"{cat.upper()} · {sec.replace('_',' ').title()}\n\n{piece}",
                    "metadata": {**base_meta(cat, sec),
                                 "chunk_index": i,
                                 "total_chunks": len(pieces),
                                 "links": links}
                }

    # ----- keydata -----
    yield from keydata_chunks(rec)


# ─────────── build corpus ────────────
def build_corpus(in_dir: str) -> List[Dict]:
    docs, seen = [], set()
    for fp in glob.glob(f"{in_dir.rstrip('/')}/**/*.json", recursive=True):
        rec = json.load(open(fp, encoding="utf-8"))
        for chunk in iter_chunks(rec):
            if len(chunk["text"]) < 30:
                continue
            sig = (chunk["metadata"]["slug"],
                   chunk["metadata"]["section"],
                   chunk["metadata"].get("chunk_index", 0))
            if sig in seen:
                continue
            seen.add(sig)
            docs.append(chunk)
    return docs


# ─────────── main ────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw", help="Raw JSON directory")
    ap.add_argument("--out_file", default="data/processed/tum_program_docs.jsonl",
                    help="Output JSONL file")
    args = ap.parse_args()

    pathlib.Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    docs = build_corpus(args.in_dir)

    with open(args.out_file, "w", encoding="utf-8") as f:
        for d in docs:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    print(f"✔  Saved {len(docs)} chunks → {args.out_file}")
