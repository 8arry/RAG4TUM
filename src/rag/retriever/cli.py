import argparse, json, re
from query_parser import parse_query
from hybrid_retriever import HybridRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--query", required=True)
ap.add_argument("--config", default="../config/retriever.yaml")
args = ap.parse_args()

retr = HybridRetriever(args.config)
filters = parse_query(args.query)
results = retr.retrieve(args.query, filters)

print(f"\nQuery: {args.query}  | Filters: {filters}\n" + "="*80)
for rk, (score, (doc, base)) in enumerate(results, 1):
    m = doc.metadata
    print(f"\n#{rk}  CE {score:.3f}  base {base:.3f}")
    print(f"{m.get('program')} · {m.get('category')} · {m.get('section')}")
    print(re.sub(r'\\s+', ' ', doc.page_content)[:320], "…")
    print("-"*80)
