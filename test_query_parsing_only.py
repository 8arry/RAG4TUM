#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for query parsing improvements without API calls
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag.retriever.query_parser import parse_query
from src.rag.retriever.improved_query_parser import enhanced_parse_query

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

if __name__ == "__main__":
    print("🧪 Testing Query Parsing Improvements")
    print("This test doesn't require OpenAI API or vector indices")
    test_query_parsing() 