#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved chunking strategy for better query-answer matching
"""

import json
import re
import uuid
from typing import List, Dict, Generator
import tiktoken
from urllib.parse import urlparse
from slugify import slugify

enc = tiktoken.get_encoding("cl100k_base")

def improved_token_chunks(text: str, max_tks: int = 256, stride: int = 64, 
                         context_overlap: int = 32):
    """
    Improved chunking with better context preservation
    - Smaller stride for better overlap
    - Add context sentences at boundaries
    """
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences into chunks based on token count
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = len(enc.encode(sentence))
        
        if current_tokens + sentence_tokens > max_tks and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
            current_chunk = overlap_sentences + [sentence]
            current_tokens = sum(len(enc.encode(s)) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def create_qa_oriented_chunks(rec: Dict) -> Generator[Dict, None, None]:
    """
    Create chunks specifically optimized for Q&A scenarios
    """
    slug, deg = slug_and_degree(rec.get("url"), rec["program_name"], rec.get("key_data", {}))
    
    def base_meta(cat, sec, chunk_type="content"):
        return {
            "program": rec["program_name"], 
            "slug": slug, 
            "degree": deg,
            "category": cat, 
            "section": sec,
            "chunk_type": chunk_type
        }

    # ----- Description -----
    if rec.get("program_description"):
        yield {
            "id": uuid.uuid4().hex,
            "text": f"DESC · Overview\n\n{rec['program_description']}",
            "metadata": base_meta("desc", "overview")
        }

    # ----- Process all text blocks with improved chunking -----
    for block, cat in [("information_on_degree_program", "info"),
                       ("application_and_admission", "apply")]:
        for sec, payload in rec.get(block, {}).items():
            raw = payload.get("text", "").strip()
            links = payload.get("links", [])
            if not raw:
                continue
                
            # Special handling for deadline/date information
            if cat == "apply" and ("deadline" in sec.lower() or "period" in sec.lower()):
                # Create a comprehensive deadline chunk
                deadline_info = f"DEADLINE INFO · {rec['program_name']}\n\n"
                deadline_info += f"Program: {rec['program_name']}\n"
                deadline_info += f"Section: {sec.replace('_', ' ').title()}\n\n"
                deadline_info += raw
                
                # Add key data dates if available
                key_data = rec.get("key_data", {})
                if "application_period" in key_data:
                    deadline_info += f"\n\nApplication Period: {key_data['application_period']}"
                
                yield {
                    "id": uuid.uuid4().hex,
                    "text": deadline_info,
                    "metadata": {**base_meta(cat, sec, "deadline"), "priority": "high", "links": links}
                }
            
            # Regular content chunks with better context
            chunks = improved_token_chunks(raw)
            for i, chunk in enumerate(chunks):
                # Add context header
                context_header = f"{cat.upper()} · {rec['program_name']} · {sec.replace('_', ' ').title()}\n\n"
                
                yield {
                    "id": uuid.uuid4().hex,
                    "text": context_header + chunk,
                    "metadata": {
                        **base_meta(cat, sec),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "links": links
                    }
                }

    # ----- keydata with improved processing -----
    key_data = rec.get("key_data", {})
    base_keydata_meta = {"program": rec["program_name"], "slug": slug, "degree": deg,
                         "category": "keydata"}
    
    for k, v in key_data.items():
        if not v:
            continue
            
        if isinstance(v, dict):
            txt = "; ".join(f"{kk}: {vv}" for kk, vv in v.items())
        elif isinstance(v, list):
            txt = ", ".join(map(str, v))
        else:
            txt = str(v)
            
        # Enhanced keydata chunk with program context
        keydata_text = f"KEYDATA · {rec['program_name']}\n\n{k.replace('_', ' ').title()}: {txt}"
        
        yield {
            "id": uuid.uuid4().hex,
            "text": keydata_text,
            "metadata": {**base_keydata_meta, "section": k}
        }

def slug_and_degree(url: str | None, prog_name: str, key_data: Dict) -> tuple[str, str]:
    """Extract slug and degree from URL or program name"""
    if url:
        path = urlparse(url).path.rstrip("/")
        slug_full = path.split("/")[-1]
        parts = slug_full.split("-")
        degree = parts[-1].lower() if len(parts) >= 1 else ""
        prog_slug = "-".join(parts[:-4]) or "-".join(parts[:-1])
        return prog_slug, degree

    prog_slug = slugify(prog_name, lowercase=True)
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