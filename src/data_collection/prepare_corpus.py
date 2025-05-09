#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_corpus.py  ·  TUM program data ETL
------------------------------------------
读取 data/raw/*.json  →
  · 清洗文本、去 \n\t
  · 拆分为若干 chunk（按逻辑小节 / 长度）
  · 去重、过滤短文本
  · 输出 JSONL（一行一文档）

Usage
-----
python prepare_corpus.py \
    --in_dir data/raw \
    --out_file data/processed/tum_program_docs.jsonl \
    --max_len 1500
"""
from __future__ import annotations
import argparse, json, re, uuid, glob, pathlib
from typing import List, Dict, Generator
import os

# Get the workspace root directory (2 levels up from this script)
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# ──────────────────────────────────────────────────────────────
# utils
# ──────────────────────────────────────────────────────────────
WS_PATTERN = re.compile(r"\s+")

def normalize_ws(text: str) -> str:
    return WS_PATTERN.sub(" ", text).strip()

def format_key_info(key_data: Dict) -> str:
    """格式化关键信息，保持重要信息的可见性"""
    info_parts = []
    
    # 特别处理重要字段
    if "main_locations" in key_data:
        locations = key_data["main_locations"]
        info_parts.append(f"Main Locations: {', '.join(locations)}")
    
    # 处理其他重要字段
    important_fields = [
        "type_of_study",
        "standard_duration_of_studies",
        "credits",
        "application_period",
        "admission_category",
        "start_of_degree_program",
        "required_language_proficiency"
    ]
    
    for field in important_fields:
        if field in key_data and key_data[field]:
            value = key_data[field]
            if isinstance(value, list):
                value = ", ".join(value)
            info_parts.append(f"{field.replace('_', ' ').title()}: {value}")
    
    # 处理其他字段
    for key, value in key_data.items():
        if key not in important_fields and key != "main_locations" and value:
            if isinstance(value, dict):
                # 处理嵌套字典（如 costs）
                for subkey, subvalue in value.items():
                    info_parts.append(f"{subkey.replace('_', ' ').title()}: {subvalue}")
            else:
                info_parts.append(f"{key.replace('_', ' ').title()}: {value}")
    
    return "\n".join(info_parts)

def iter_chunks(rec: Dict) -> Generator[Dict, None, None]:
    """Yield chunk dicts with text + metadata."""
    prog = rec["program_name"]
    key_data = rec.get("key_data", {})
    
    # 保持关键信息的结构化
    base_metadata = {
        "program": prog,
        "key_data": key_data,  # 保持原始结构
        "category": "desc",
        "section": "overview"
    }
    
    # 生成包含关键信息的文本
    key_info_text = format_key_info(key_data)

    # 1) description - 不需要分割
    if rec.get("program_description"):
        yield {
            "text": normalize_ws(f"{key_info_text}\n\n{rec['program_description']}"),
            "metadata": {
                **base_metadata,
                "category": "desc",
                "section": "overview",
                "links": [],
                "chunk_index": 0,
                "total_chunks": 1
            }
        }

    # 2) info / apply
    for block, cat in [
        ("information_on_degree_program", "info"),
        ("application_and_admission", "apply"),
    ]:
        for sec, payload in rec.get(block, {}).items():
            txt = normalize_ws(payload.get("text", ""))
            links = payload.get("links", [])
            if not txt:
                continue

            # 根据内容类型选择不同的 chunk 大小
            max_len = 1000 if sec == "program_profile" else 800
            # 将关键信息添加到文本开头
            full_text = f"{key_info_text}\n\n{txt}"
            chunks = split_long(full_text, max_len)
            
            for i, chunk in enumerate(chunks):
                yield {
                    "text": chunk,
                    "metadata": {
                        **base_metadata,
                        "category": cat,
                        "section": sec,
                        "links": links,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                }

def split_long(text: str, max_len: int = 1500) -> List[str]:
    """Split text into chunks while preserving paragraph boundaries."""
    if len(text) <= max_len:
        return [text]
    
    # 首先按段落分割
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # 如果段落本身超过最大长度，需要进一步分割
        if len(para) > max_len:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            # 在句子边界分割长段落
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_sentence = []
            current_sentence_len = 0
            
            for sentence in sentences:
                if current_sentence_len + len(sentence) > max_len:
                    if current_sentence:
                        chunks.append(" ".join(current_sentence))
                    current_sentence = [sentence]
                    current_sentence_len = len(sentence)
                else:
                    current_sentence.append(sentence)
                    current_sentence_len += len(sentence) + 1
            
            if current_sentence:
                chunks.append(" ".join(current_sentence))
        else:
            if current_len + len(para) > max_len:
                chunks.append(" ".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para) + 2
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# ──────────────────────────────────────────────────────────────
def build_corpus(in_dir: str, max_len: int) -> List[Dict]:
    docs, seen = [], set()
    for fp in glob.glob(f"{in_dir.rstrip('/')}/**/*.json", recursive=True):
        rec = json.load(open(fp, encoding="utf-8"))
        for chunk in iter_chunks(rec):
            for piece in split_long(chunk["text"], max_len):
                if len(piece) < 30:                 # 丢弃太短文本
                    continue
                sig = (chunk["metadata"]["program"], piece)
                if sig in seen:
                    continue
                seen.add(sig)
                docs.append({
                    "id": uuid.uuid4().hex,
                    "text": piece,
                    "metadata": chunk["metadata"]
                })
    return docs

# ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir",   default=os.path.join(WORKSPACE_ROOT, "data", "raw"), 
                    help="原始 JSON 目录")
    ap.add_argument("--out_file", default=os.path.join(WORKSPACE_ROOT, "data", "processed", "tum_program_docs.jsonl"),
                    help="输出 JSONL 文件")
    ap.add_argument("--max_len",  type=int, default=1500,
                    help="单 chunk 最大字符数（默认 1500）")
    args = ap.parse_args()

    pathlib.Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    docs = build_corpus(args.in_dir, args.max_len)

    with open(args.out_file, "w", encoding="utf-8") as f:
        for d in docs:
            json.dump(d, f, ensure_ascii=False)
            f.write("\n")

    print(f"✔  Saved {len(docs)} chunks → {args.out_file}")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
