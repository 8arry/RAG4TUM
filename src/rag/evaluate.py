#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py  ·  RAG 评估脚本
------------------------------------------
运行评估并生成报告
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from retriever import Retriever
from evaluator import Evaluator

def main():
    # 获取工作空间根目录
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    
    # 初始化检索器
    index_path = os.path.join(workspace_root, "data", "embeddings", "tum_programs.index")
    docs_path = os.path.join(workspace_root, "data", "processed", "tum_program_docs.jsonl")
    retriever = Retriever(index_path, docs_path)
    
    # 初始化评估器
    evaluator = Evaluator(retriever)
    
    # 加载测试查询
    test_queries_path = os.path.join(os.path.dirname(__file__), "test_queries.json")
    with open(test_queries_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    
    # 运行评估
    scores = evaluator.evaluate_queries(test_data["queries"])
    
    # 生成报告
    report_path = os.path.join(workspace_root, "data", "evaluation", "retrieval_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = evaluator.generate_report(scores, report_path)
    
    # 打印评估结果
    print("\n评估结果:")
    print("=" * 50)
    for metric, score in report["metrics"].items():
        print(f"{metric}: {score:.4f}")
    print("\n总结:")
    print(f"整体得分: {report['summary']['overall_score']:.4f}")
    print(f"最佳指标: {report['summary']['best_metric']}")
    print(f"最差指标: {report['summary']['worst_metric']}")
    print(f"\n详细报告已保存至: {report_path}")

if __name__ == "__main__":
    main() 