#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to compare original vs improved retrieval performance
"""

import os
import sys
import logging
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.rag.retriever.improved_hybrid_retriever import ImprovedHybridRetriever
from src.rag.retriever.query_parser import parse_query
from src.rag.retriever.improved_query_parser import enhanced_parse_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_parsing():
    """Test query parsing improvements"""
    test_queries = [
        "Deadline for Information Engineering master application",
        "When can I apply for Computer Science bachelor?", 
        "What documents are needed for Data Engineering admission?",
        "How much does the Informatics program cost?",
        "What courses are in the Mathematics curriculum?"
    ]
    
    print("=" * 80)
    print("QUERY PARSING COMPARISON")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        
        # Original parsing
        original_result = parse_query(query)
        print(f"Original: {original_result}")
        
        # Enhanced parsing
        enhanced_result = enhanced_parse_query(query)
        print(f"Enhanced: Program={enhanced_result.program}, "
              f"Category={enhanced_result.category}, "
              f"Intent={enhanced_result.intent_type}, "
              f"Temporal={enhanced_result.temporal_keywords}")

def test_retrieval_comparison():
    """Compare retrieval results between original and improved systems"""
    
    test_queries = [
        "Deadline for Information Engineering master application",
        "Application documents for Computer Science",
        "Credits required for Data Engineering master"
    ]
    
    print("\n" + "=" * 80)
    print("RETRIEVAL COMPARISON")
    print("=" * 80)
    
    try:
        # Initialize retrievers
        original_retriever = HybridRetriever("src/rag/config/retriever.yaml")
        print("✓ Original retriever initialized")
        
        improved_retriever = ImprovedHybridRetriever("src/rag/config/improved_retriever.yaml") 
        print("✓ Improved retriever initialized")
        
    except Exception as e:
        print(f"❌ Error initializing retrievers: {e}")
        print("Make sure the vector indices are built and config paths are correct")
        return
    
    for query in test_queries:
        print(f"\n{'='*20} Query: '{query}' {'='*20}")
        
        # Original retrieval
        print(f"\n--- ORIGINAL RETRIEVAL ---")
        try:
            original_filters = parse_query(query)
            original_results = original_retriever.retrieve(query, original_filters)
            
            print(f"Found {len(original_results)} results")
            for i, (score, (doc, base_score)) in enumerate(original_results[:3], 1):
                meta = doc.metadata
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"{i}. Score: {score:.3f} | {meta.get('program', 'N/A')} | "
                      f"{meta.get('category', 'N/A')} | {meta.get('section', 'N/A')}")
                print(f"   Content: {content_preview}...")
                
        except Exception as e:
            print(f"❌ Original retrieval error: {e}")
        
        # Improved retrieval 
        print(f"\n--- IMPROVED RETRIEVAL ---")
        try:
            improved_results = improved_retriever.retrieve(query, enhanced_parsing=True)
            
            print(f"Found {len(improved_results)} results")
            for i, (score, (doc, base_score)) in enumerate(improved_results[:3], 1):
                meta = doc.metadata
                content_preview = doc.page_content[:100].replace('\n', ' ')
                print(f"{i}. Score: {score:.3f} | {meta.get('program', 'N/A')} | "
                      f"{meta.get('category', 'N/A')} | {meta.get('section', 'N/A')}")
                print(f"   Content: {content_preview}...")
                
        except Exception as e:
            print(f"❌ Improved retrieval error: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing RAG Improvements")
    print("This script compares original vs improved query parsing and retrieval")
    
    # Test query parsing
    test_query_parsing()
    
    # Test retrieval (requires built indices)
    print(f"\nChecking if vector indices exist...")
    index_path = "data/embeddings/index.faiss"
    bm25_path = "data/embeddings/bm25.pkl"
    
    if os.path.exists(index_path) and os.path.exists(bm25_path):
        print("✓ Vector indices found, testing retrieval...")
        test_retrieval_comparison()
    else:
        print("❌ Vector indices not found. Please run:")
        print("  python src/data_collection/prepare_corpus.py")
        print("  python src/data_collection/vectorize.py") 
        print("  python src/rag/build_bm25.py")

if __name__ == "__main__":
    main() 