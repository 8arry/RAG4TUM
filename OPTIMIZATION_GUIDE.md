# RAG4TUM Optimization Guide

## Problem Analysis

The original RAG system has several issues with query-answer matching:

### 1. **Chunking Issues**
- **Large stride**: Original 128 token stride causes important information to be split
- **Lack of context**: Insufficient context connections between chunks
- **Generic chunking**: No specialized optimization for specific query types (like deadline queries)

### 2. **Query Understanding Issues** 
- **Inaccurate program name recognition**: Difficulty identifying compound program names (like "Information Engineering")
- **Weak intent understanding**: Unable to accurately identify specific intents like deadline, application, etc.
- **Lack of semantic expansion**: Limited query vocabulary, cannot associate synonyms

### 3. **Retrieval Scoring Issues**
- **Improper scoring weights**: Unreasonable allocation of semantic similarity and exact match weights
- **Lack of intent awareness**: Reranking doesn't consider query intent
- **Insufficient special query handling**: High-priority queries like deadlines lack special treatment

## Optimization Solutions

### 1. **Improved Chunking Strategy** (`src/data_collection/improved_chunking.py`)

```python
def improved_token_chunks(text: str, max_tks: int = 256, stride: int = 64):
    """
    - Reduce stride from 128 to 64, increase overlap
    - Split by sentence boundaries, maintain semantic integrity
    - Add context overlap to ensure information continuity
    """

def create_qa_oriented_chunks(rec: Dict):
    """
    - Create specialized high-priority chunks for deadline queries
    - Add rich context header information
    - Include program name and section info in each chunk
    """
```

**Improvements**:
- ✅ Reduce important information loss
- ✅ Improve retrievability of deadline and other key information
- ✅ Enhance chunk self-containment

### 2. **Enhanced Query Parsing** (`src/rag/retriever/improved_query_parser.py`)

```python
def enhanced_parse_query(query: str) -> QueryIntent:
    """
    - Intent-based semantic analysis
    - Temporal keyword detection
    - Entity extraction and fuzzy program name matching
    """
```

**New Features**:
- 🎯 **Intent Classification**: deadline_intent, application_intent, document_intent, etc.
- 🔍 **Semantic Understanding**: Recognize "When can I apply" type natural language queries
- 📝 **Entity Extraction**: Automatically identify program names, degree types, temporal words

### 3. **Smart Retriever** (`src/rag/retriever/improved_hybrid_retriever.py`)

```python
def _apply_enhanced_filters(self, merged_results, filters, intent):
    """
    - Intent-based dynamic scoring
    - Specialized deadline query weighting
    - Program-category exact match rewards
    """

def _enhanced_rerank(self, query, filtered_results, filters, intent):
    """
    - Combine semantic scores and rule scores
    - Intent-aware reranking
    - Minimum score threshold filtering
    """
```

**Core Improvements**:
- 🚀 **Query Expansion**: Automatically add relevant synonyms
- 🎯 **Intent-Aware Scoring**: Deadline queries get 3x weighting
- ⚡ **Hybrid Scoring**: Combine reranker scores and rule scores

### 4. **Optimized Configuration** (`src/rag/config/improved_retriever.yaml`)

```yaml
# Retrieval parameter optimization
n_dense: 60                    # Increase candidate count
n_sparse: 60                   # Increase BM25 candidates
top_m: 20                      # More reranking candidates
top_k: 10                      # Return more results

# New semantic parameters
deadline_priority_boost: 3.0   # Special weighting for deadline queries
exact_match_boost: 2.0         # Exact match rewards
query_expansion: true          # Enable query expansion
```

## Usage Instructions

### 1. Regenerate corpus with improved chunking

```bash
# Use improved chunking strategy to reprocess data
python src/data_collection/prepare_corpus.py --use_improved_chunking

# Rebuild vector indices
python src/data_collection/vectorize.py
python src/rag/build_bm25.py
```

### 2. Test improvement effects

```bash
# Run comparison tests
python test_improved_retrieval.py

# Test specific queries
python src/rag/retriever/cli.py --config src/rag/config/improved_retriever.yaml \
    --query "Deadline for Information Engineering master application" --verbose
```

### 3. Use improved retriever

```python
from src.rag.retriever.improved_hybrid_retriever import ImprovedHybridRetriever

# Initialize improved retriever
retriever = ImprovedHybridRetriever("src/rag/config/improved_retriever.yaml")

# Retrieve with enhanced parsing
results = retriever.retrieve(
    "Deadline for Information Engineering master application", 
    enhanced_parsing=True
)
```

## Expected Improvement Effects

### For deadline-type queries:
- ✅ **More precise**: Deadline information ranks in top 3
- ✅ **More comprehensive**: Includes application period and related information
- ✅ **More direct**: Reduces interference from irrelevant program information

### For program-specific queries:
- ✅ **More accurate program matching**: "Information Engineering" vs "Bioprocess Engineering"
- ✅ **Clearer category distinction**: apply vs info vs keydata
- ✅ **More relevant content**: Avoid dilution from generic information

### Overall metric expectations:
- 📈 **Relevance**: 30-50% improvement in top-3 precision
- 📈 **Accuracy**: 50% reduction in program mismatches
- 📈 **Response Quality**: More direct answers to user questions

## Debugging and Tuning

### 1. Enable detailed logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Adjust parameters
```yaml
# If too few results, increase candidates
n_dense: 80
n_sparse: 80

# If precision insufficient, increase boost parameters
deadline_priority_boost: 4.0
exact_match_boost: 2.5

# If need more diversity, lower threshold
min_rerank_score: -3.0
```

### 3. Analyze query parsing
```python
from src.rag.retriever.improved_query_parser import enhanced_parse_query

intent = enhanced_parse_query("your query")
print(f"Program: {intent.program}")
print(f"Category: {intent.category}") 
print(f"Intent: {intent.intent_type}")
print(f"Temporal: {intent.temporal_keywords}")
```

## Important Notes

1. **Rebuild indices**: Need to rebuild vector indices after using improved chunking
2. **Configuration files**: Ensure correct configuration file paths
3. **Dependencies**: May need to install additional NLP dependencies
4. **Memory usage**: Increased candidate count will increase memory usage

This optimization suite should significantly improve the "large gap between search results and query" issue you mentioned. It's recommended to test in a test environment first and then deploy to production after verifying effectiveness. 