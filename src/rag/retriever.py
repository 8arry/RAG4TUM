#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retriever.py  ·  RAG 检索器
------------------------------------------
加载向量索引，执行相似度搜索，返回相关文档
"""
from __future__ import annotations
import os, json, numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict, Tuple
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

class Retriever:
    def __init__(self, index_path: str, docs_path: str):
        """初始化检索器
        
        Args:
            index_path: FAISS 索引文件路径
            docs_path: 文档 JSONL 文件路径
        """
        # 加载 FAISS 索引
        self.index = faiss.read_index(index_path)
        
        # 加载文档
        self.docs = []
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=api_key)
    
    def _enhance_query(self, query: str) -> str:
        """优化查询文本
        
        1. 添加上下文信息
        2. 规范化查询格式
        3. 突出关键信息
        """
        # 提取关键信息
        program_name = None
        degree_type = None
        query_type = None
        
        # 识别程序名称
        query_lower = query.lower()
        if "information engineering" in query_lower:
            program_name = "Information Engineering"
        elif "electrical engineering" in query_lower:
            program_name = "Electrical Engineering"
        # ... 可以添加更多程序名称识别
        
        # 识别学位类型
        if "master" in query_lower or "msc" in query_lower:
            degree_type = "Master"
        elif "bachelor" in query_lower or "bsc" in query_lower:
            degree_type = "Bachelor"
            
        # 识别查询类型
        query_type_mapping = {
            "credit": "credits",
            "location": "location",
            "campus": "location",
            "duration": "duration",
            "semester": "duration",
            "language": "language",
            "requirement": "requirements",
            "application": "application",
            "period": "application",
            "cost": "costs",
            "fee": "costs",
            "admission": "admission",
            "category": "admission",
            "start": "start",
            "begin": "start",
            "structure": "structure",
            "description": "description"
        }
        
        for keyword, qtype in query_type_mapping.items():
            if keyword in query_lower:
                query_type = qtype
                break
            
        # 构建增强查询
        enhanced_parts = []
        
        # 1. 基础查询
        enhanced_parts.append(query)
        
        # 2. 程序信息
        if program_name:
            enhanced_parts.append(f"Program: {program_name}")
        
        # 3. 学位类型
        if degree_type:
            enhanced_parts.append(f"Degree: {degree_type}")
        
        # 4. 查询类型
        if query_type:
            enhanced_parts.append(f"Looking for: {query_type}")
        
        # 5. 添加上下文
        enhanced_parts.append("TUM program information")
        
        # 组合增强查询
        enhanced = " | ".join(enhanced_parts)
        
        return enhanced
    
    def get_embedding(self, text: str) -> np.ndarray:
        """获取文本的向量表示"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """执行相似度搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表，每个文档包含文本、元数据和相似度分数
        """
        # 1. 优化查询
        enhanced_query = self._enhance_query(query)
        query_vector = self.get_embedding(enhanced_query)
        
        # 2. 执行搜索
        distances, indices = self.index.search(
            query_vector.reshape(1, -1), 
            top_k * 3  # 获取更多结果用于过滤
        )
        
        # 3. 处理结果
        results = []
        seen_programs = set()
        
        # 提取查询信息
        query_lower = query.lower()
        query_parts = enhanced_query.split(" | ")
        program_name = next((p.split(": ")[1] for p in query_parts if p.startswith("Program:")), None)
        degree_type = next((p.split(": ")[1] for p in query_parts if p.startswith("Degree:")), None)
        query_type = next((p.split(": ")[1] for p in query_parts if p.startswith("Looking for:")), None)
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # FAISS 填充值
                continue
                
            doc = self.docs[idx]
            doc_program = doc["metadata"]["program"]
            
            # 跳过重复的程序
            if doc_program in seen_programs:
                continue
            
            # 计算基础相似度分数
            similarity = 1.0 / (1.0 + dist)
            
            # 1. 程序名称匹配
            if program_name and program_name.lower() in doc_program.lower():
                similarity *= 1.5
            
            # 2. 学位类型匹配
            if degree_type:
                if degree_type == "Master" and "master" in doc_program.lower():
                    similarity *= 1.3
                elif degree_type == "Bachelor" and "bachelor" in doc_program.lower():
                    similarity *= 1.3
            
            # 3. 查询类型匹配
            if query_type:
                # 根据查询类型检查相关内容
                if query_type == "credits" and "credits" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "location" and "Main Locations" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "duration" and "Duration" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "language" and "Language" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "application" and "Application" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "costs" and "Costs" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "admission" and "Admission" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "start" and "Start" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "structure" and "Structure" in doc["text"]:
                    similarity *= 1.2
                elif query_type == "description" and "Description" in doc["text"]:
                    similarity *= 1.2
            
            # 4. 内容相关性
            doc_text = doc["text"].lower()
            query_words = set(query_lower.split())
            matching_words = sum(1 for word in query_words if word in doc_text)
            if matching_words > 0:
                similarity *= (1 + 0.1 * matching_words)
            
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "similarity": similarity
            })
            
            seen_programs.add(doc_program)
            
            # 达到所需结果数量后停止
            if len(results) >= top_k:
                break
        
        # 4. 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results[:top_k] 