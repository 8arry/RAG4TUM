#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_retriever.py  ·  测试检索器
------------------------------------------
测试检索器的搜索效果
"""
import os
from retriever import Retriever


WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def print_results(results):
    """打印搜索结果"""
    print("\n搜索结果:")
    print("-" * 80)
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"程序: {result['metadata']['program']}")
        print(f"相似度分数: {result['similarity']:.4f}")
        print(f"内容: {result['text'][:200]}...")  # 只显示前200个字符
        print("-" * 80)

def main():
    # 初始化检索器
    index_path = os.path.join(WORKSPACE_ROOT, "data", "embeddings", "tum_programs.index")
    docs_path = os.path.join(WORKSPACE_ROOT, "data", "embeddings", "tum_programs_metadata.json")
    retriever = Retriever(index_path=index_path, docs_path=docs_path)
    
    # 测试查询
    test_queries = [
        "How to apply for Information Engineering Master program?"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        print("=" * 80)
        
        # 执行搜索
        results = retriever.search(query)
        
        # 打印结果
        print_results(results)

if __name__ == "__main__":
    main() 