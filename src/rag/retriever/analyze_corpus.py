#!/usr/bin/env python
"""
Script to analyze the corpus and find documents matching specific criteria.
Useful for debugging and improving the retriever.
"""

import argparse
import logging
import pathlib
import pickle
import re
import yaml
from typing import List, Dict, Any, Tuple

from hybrid_retriever import HybridRetriever

def setup_logging(verbose=False):
    """Configure logging based on verbosity level"""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def find_documents_by_program(retriever: HybridRetriever, program_name: str) -> List[Tuple[object, Dict[str, Any]]]:
    """Find all documents related to a specific program"""
    docs = []
    for doc in retriever.docs:
        program = doc.metadata.get("program", "").lower()
        if program_name.lower() in program:
            docs.append((doc, doc.metadata))
    return docs

def find_documents_by_criteria(retriever: HybridRetriever, 
                               program: str = None, 
                               category: str = None, 
                               section: str = None) -> List[Tuple[object, Dict[str, Any]]]:
    """Find all documents matching the provided criteria"""
    docs = []
    for doc in retriever.docs:
        meta = doc.metadata
        if program and program.lower() not in meta.get("program", "").lower():
            continue
        if category and category.lower() != meta.get("category", "").lower():
            continue
        if section and section.lower() not in meta.get("section", "").lower():
            continue
        docs.append((doc, meta))
    return docs

def print_document_summary(docs: List[Tuple[object, Dict[str, Any]]]):
    """Print a summary of the found documents"""
    print(f"\nFound {len(docs)} matching documents:")
    
    # Group by category
    by_category = {}
    for doc, meta in docs:
        cat = meta.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append((doc, meta))
    
    # Print summary by category
    for cat, cat_docs in by_category.items():
        print(f"\n== Category: {cat} ({len(cat_docs)} documents) ==")
        for i, (doc, meta) in enumerate(cat_docs, 1):
            program = meta.get("program", "unknown")
            section = meta.get("section", "unknown")
            # Show truncated content
            content = re.sub(r'\s+', ' ', doc.page_content).strip()
            if len(content) > 100:
                content = content[:97] + "..."
            print(f"{i}. {program} | {section}\n   {content}")
            print()

def main():
    """Main entry point for the corpus analyzer"""
    parser = argparse.ArgumentParser(description="Analyze corpus for specific documents")
    parser.add_argument("--config", default="../config/retriever.yaml", help="Path to retriever config")
    parser.add_argument("--program", help="Program name to search for")
    parser.add_argument("--category", help="Category to filter by (apply, info, keydata)")
    parser.add_argument("--section", help="Section to filter by")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", help="Save results to file")
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    if not args.program and not args.category and not args.section:
        print("Error: At least one filter (program, category, or section) must be specified.")
        return 1
    
    try:
        # Initialize retriever to get access to docs
        logger.info(f"Loading corpus from config: {args.config}")
        retriever = HybridRetriever(args.config)
        logger.info(f"Loaded {len(retriever.docs)} documents in corpus")
        
        # Find documents matching criteria
        docs = find_documents_by_criteria(
            retriever,
            program=args.program,
            category=args.category,
            section=args.section
        )
        
        # Print results
        print_document_summary(docs)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for i, (doc, meta) in enumerate(docs, 1):
                    f.write(f"Document {i}\n")
                    f.write(f"Program: {meta.get('program', 'unknown')}\n")
                    f.write(f"Category: {meta.get('category', 'unknown')}\n")
                    f.write(f"Section: {meta.get('section', 'unknown')}\n")
                    f.write("\nContent:\n")
                    f.write(doc.page_content)
                    f.write("\n\n" + "-"*80 + "\n\n")
            print(f"\nSaved {len(docs)} documents to {args.output}")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main()) 