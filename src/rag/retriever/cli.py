#!/usr/bin/env python
import argparse
import json
import re
import sys
import logging
from pathlib import Path
import yaml

from query_parser import parse_query
from hybrid_retriever import HybridRetriever

def setup_logging(verbose=False):
    """Configure logging based on verbosity level"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    """Main entry point for the retriever CLI"""
    ap = argparse.ArgumentParser(description="TUM RAG Retriever CLI")
    ap.add_argument("--query", required=True, help="Query string to search for")
    ap.add_argument("--config", default="src/rag/config/retriever.yaml", help="Path to retriever config")
    ap.add_argument("--output", "-o", help="Output results to file (JSON format)")
    ap.add_argument("--limit", "-n", type=int, help="Limit number of results")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ap.add_argument("--raw", action="store_true", help="Show raw results without formatting")
    ap.add_argument("--no-filter", action="store_true", help="Disable filtering (use raw retrieval results)")
    ap.add_argument("--exact-match", action="store_true", help="Only return exact program matches")
    args = ap.parse_args()

    logger = setup_logging(args.verbose)
    
    try:
        # Initialize retriever
        logger.debug(f"Initializing retriever with config: {args.config}")
        retr = HybridRetriever(args.config)
        
        # Parse query to extract filters
        filters = parse_query(args.query)
        logger.debug(f"Parsed query '{args.query}' to filters: {filters}")
        
        # If no-filter flag is set, clear the filters
        if args.no_filter:
            logger.info("Filters disabled by --no-filter flag")
            filters = {"slug": "", "degree": "", "category": ""}
        
        # Retrieve results
        results = retr.retrieve(args.query, filters)
        
        # Apply exact match filtering if requested
        if args.exact_match and filters.get("slug") and results:
            logger.info("Filtering for exact program matches only")
            exact_matches = []
            slug = filters["slug"].lower().replace('-', ' ')
            
            for score, (doc, base) in results:
                program = doc.metadata.get("program", "").lower()
                if slug == program or slug in program:
                    exact_matches.append((score, (doc, base)))
            
            if exact_matches:
                results = exact_matches
                logger.info(f"Found {len(results)} exact matches for '{slug}'")
            else:
                logger.warning(f"No exact matches found for '{slug}'. Showing all results.")
        
        # Limit results if requested
        if args.limit and 0 < args.limit < len(results):
            results = results[:args.limit]
        
        # Format and display/save results
        if args.output:
            # Save to file
            output_data = {
                "query": args.query,
                "filters": filters,
                "results": [
                    {
                        "score": float(score),
                        "base_score": float(base),
                        "metadata": doc.metadata,
                        "content": doc.page_content
                    }
                    for score, (doc, base) in results
                ]
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output}")
        
        # Print results to console
        extracted_info = f"Program: {filters.get('slug', 'None')} | Degree: {filters.get('degree', 'None')} | Category: {filters.get('category', 'None')}"
        print(f"\nQuery: {args.query}\nExtracted: {extracted_info}\n" + "="*80)
        
        if not results:
            print("\n⚠️  No results found for this query with the current filters.")
            print("Try one of the following:")
            print("  - Use more general terms in your query")
            print("  - Use the --no-filter flag to disable filtering")
            print("  - Check if the program name is correct")
            print("="*80)
            return 0
            
        if args.raw:
            print(json.dumps(results, indent=2))
        else:
            for rk, (score, (doc, base)) in enumerate(results, 1):
                m = doc.metadata
                program = m.get("program", "")
                category = m.get("category", "")
                section = m.get("section", "")
                
                # Highlight if program matches query
                if filters.get("slug") and filters["slug"].replace("-", " ") in program.lower():
                    program_display = f"✓ {program}"
                else:
                    program_display = f"× {program}"
                
                # Highlight if category matches query
                if filters.get("category") and (
                    filters["category"].lower() == category.lower() or 
                    filters["category"].lower() in section.lower()):
                    category_display = f"✓ {category}"
                else:
                    category_display = f"× {category}"
                
                print(f"\n#{rk}  Score: {score:.3f}  Base: {base:.3f}")
                print(f"Program: {program_display} | Category: {category_display} | Section: {section}")
                
                # Clean and truncate content for display
                content = re.sub(r'\s+', ' ', doc.page_content).strip()
                if len(content) > 320:
                    content = content[:320] + "…"
                print(content)
                print("-"*80)
                
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except yaml.YAMLError as e:
        logger.error(f"Error in YAML config: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
