import re
from typing import Dict

DEGREE_MAP = {
    r"\bmaster\b|\bmsc\b": "msc",
    r"\bbachelor\b|\bbsc\b": "bsc",
    r"\bphd\b|\bdoctor":     "phd",
    r"\bma\b":               "ma",
    r"\bba\b":               "ba"
}

ACTION_CAT = {
    "apply":       "apply",
    "application": "apply",
    "deadline":    "apply",
    "documents":   "apply",
    "document":    "apply",
    "credit":      "keydata",
    "credits":     "keydata",
    "language":    "keydata",
    "languages":   "keydata",
    "structure":   "info",
    "curriculum":  "info",
    "modules":     "info",
}


def parse_query(q: str) -> Dict[str, str]:
    q_low = q.lower()

    # 1. degree
    degree = ""
    for pat, deg in DEGREE_MAP.items():
        if re.search(pat, q_low):
            degree = deg; break

    # 2. action → category
    category = ""
    for kw, cat in ACTION_CAT.items():
        if kw in q_low:
            category = cat; break

    # 3. slug：取最长专业名词（简单 heuristics，可换成 LLM NER）
    m = re.search(r'for ([A-Z][\w\s&-]+?) (?:master|bachelor|msc|bsc)', q, re.I)
    slug = re.sub(r'\W+', '-', m.group(1).strip().lower()) if m else ""

    return {"slug": slug, "degree": degree, "category": category}
