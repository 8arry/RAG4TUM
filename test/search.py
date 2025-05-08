#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search.py  ·  TUM program search
------------------------------------------
使用 FAISS 索引进行相似度搜索

Usage
-----
python search.py \
    --query "你的搜索查询" \
    --index_dir data/embeddings \
    --top_k 5
"""
import os
import json
import argparse
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

class ProgramSearcher:
    def __init__(self, index_dir: str, model_name: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
        self.dimension = 1536
        
        # 加载索引和元数据
        self.index = faiss.read_index(os.path.join(index_dir, "tum_programs.index"))
        with open(os.path.join(index_dir, "tum_programs_metadata.json"), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取查询文本的向量表示"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索最相似的文档"""
        # 获取查询的向量表示
        query_vector = self.get_embedding(query)
        
        # 执行搜索
        distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
        
        # 获取结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS 返回 -1 表示没有找到结果
                doc = self.metadata[idx]
                results.append({
                    'rank': i + 1,
                    'score': float(distances[0][i]),
                    'title': doc.get('title', '无标题'),
                    'text': doc.get('text', '')[:200] + '...',
                    'url': doc.get('url', ''),  # 添加URL
                    'program_type': doc.get('program_type', ''),  # 添加项目类型
                    'credits': doc.get('credits', '')  # 添加学分信息
                })
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="搜索查询")
    parser.add_argument("--index_dir", 
                       default=os.path.join(WORKSPACE_ROOT, "data", "embeddings"),
                       help="索引目录")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    args = parser.parse_args()

    # 初始化搜索器
    searcher = ProgramSearcher(args.index_dir)
    
    # 执行搜索
    print(f"\n搜索查询: {args.query}")
    print("-" * 80)
    
    results = searcher.search(args.query, args.top_k)
    
    # 显示结果
    for result in results:
        print(f"\n排名 {result['rank']} (相似度分数: {result['score']:.4f})")
        print(f"标题: {result['title']}")
        print(f"内容: {result['text']}")
        print("-" * 80)

if __name__ == "__main__":
    main() 