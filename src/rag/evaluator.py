#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluator.py  ·  RAG 评估框架
------------------------------------------
系统性地评估检索性能：
1. 支持多种评估指标
2. 批量测试不同查询类型
3. 生成评估报告
"""
from __future__ import annotations
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from retriever import Retriever

@dataclass
class QueryResult:
    """查询结果"""
    query: str
    expected_program: str
    results: List[Dict]
    relevance_scores: List[float]
    found_expected: bool
    rank_of_expected: int

class Evaluator:
    def __init__(self, retriever: Retriever):
        """初始化评估器
        
        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever
        self.metrics = {
            "precision@k": self._precision_at_k,
            "recall@k": self._recall_at_k,
            "mrr": self._mean_reciprocal_rank,
            "ndcg@k": self._ndcg_at_k
        }
    
    def evaluate_queries(self, test_queries: List[Dict]) -> Dict[str, float]:
        """评估一组查询
        
        Args:
            test_queries: 测试查询列表，每个查询包含：
                - query: 查询文本
                - expected_program: 期望返回的程序名称
                - expected_info: 期望包含的信息类型（可选）
        
        Returns:
            各项评估指标的得分
        """
        results = []
        for query_info in test_queries:
            query = query_info["query"]
            expected_program = query_info["expected_program"]
            
            # 执行查询
            search_results = self.retriever.search(query)
            
            # 计算相关性分数
            relevance_scores = []
            found_expected = False
            rank_of_expected = -1
            
            for i, result in enumerate(search_results):
                program_name = result["metadata"]["program"]
                # 计算相关性分数（基于程序名称匹配和内容相关性）
                relevance = 1.0 if program_name == expected_program else 0.0
                if "expected_info" in query_info:
                    # 如果指定了期望信息，检查是否包含
                    if query_info["expected_info"] in result["text"]:
                        relevance += 0.5
                relevance_scores.append(relevance)
                
                if program_name == expected_program:
                    found_expected = True
                    rank_of_expected = i + 1
            
            results.append(QueryResult(
                query=query,
                expected_program=expected_program,
                results=search_results,
                relevance_scores=relevance_scores,
                found_expected=found_expected,
                rank_of_expected=rank_of_expected
            ))
        
        # 计算各项指标
        scores = {}
        for metric_name, metric_func in self.metrics.items():
            scores[metric_name] = metric_func(results)
        
        return scores
    
    def _precision_at_k(self, results: List[QueryResult], k: int = 5) -> float:
        """计算 Precision@K"""
        precisions = []
        for result in results:
            if result.rank_of_expected > 0 and result.rank_of_expected <= k:
                precisions.append(1.0)
            else:
                precisions.append(0.0)
        return np.mean(precisions)
    
    def _recall_at_k(self, results: List[QueryResult], k: int = 5) -> float:
        """计算 Recall@K"""
        recalls = []
        for result in results:
            if result.rank_of_expected > 0 and result.rank_of_expected <= k:
                recalls.append(1.0)
            else:
                recalls.append(0.0)
        return np.mean(recalls)
    
    def _mean_reciprocal_rank(self, results: List[QueryResult]) -> float:
        """计算 Mean Reciprocal Rank (MRR)"""
        ranks = []
        for result in results:
            if result.rank_of_expected > 0:
                ranks.append(1.0 / result.rank_of_expected)
            else:
                ranks.append(0.0)
        return np.mean(ranks)
    
    def _ndcg_at_k(self, results: List[QueryResult], k: int = 5) -> float:
        """计算 Normalized Discounted Cumulative Gain@K"""
        ndcgs = []
        for result in results:
            # 计算 DCG
            dcg = 0.0
            for i, score in enumerate(result.relevance_scores[:k]):
                dcg += score / np.log2(i + 2)  # i+2 因为 log2(1) = 0
            
            # 计算 IDCG（理想情况下的 DCG）
            ideal_scores = sorted(result.relevance_scores, reverse=True)[:k]
            idcg = 0.0
            for i, score in enumerate(ideal_scores):
                idcg += score / np.log2(i + 2)
            
            # 计算 NDCG
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs)
    
    def generate_report(self, scores: Dict[str, float], output_path: str):
        """生成评估报告
        
        Args:
            scores: 评估指标得分
            output_path: 输出文件路径
        """
        report = {
            "metrics": scores,
            "summary": {
                "overall_score": np.mean(list(scores.values())),
                "best_metric": max(scores.items(), key=lambda x: x[1])[0],
                "worst_metric": min(scores.items(), key=lambda x: x[1])[0]
            }
        }
        
        # 保存报告
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report 