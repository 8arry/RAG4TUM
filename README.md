# RAG4TUM - Retrieval Augmented Generation for TUM Applications

A specialized information retrieval system for Technical University of Munich (TUM) program information. This system combines dense retrieval (embeddings), sparse retrieval (BM25), and neural reranking to provide accurate answers about academic programs.

## Features

- **Hybrid Retrieval**: Combines dense vector search and BM25 keyword search for optimal results
- **Smart Query Parsing**: Extracts program names, degree types, and information categories
- **Neural Reranking**: Uses a cross-encoder model to improve result quality
- **Flexible Filtering**: Filter results by program, degree type, and information category

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RAG4TUM.git
cd RAG4TUM

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for OpenAI API (required for embeddings)
# Replace with your actual API key
export OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
RAG4TUM/
├── data/                  # Data storage
│   ├── embeddings/        # Vector and BM25 indices
│   │   ├── index.faiss    # FAISS vector index
│   │   ├── index.pkl      # FAISS metadata
│   │   └── bm25.pkl       # BM25 index and documents
│   ├── processed/         # Processed document chunks
│   └── raw/               # Source data files
├── src/
│   ├── data_collection/   # Scripts to collect and process data
│   │   ├── scraper.py     # Web scraping for TUM data
│   │   ├── prepare_corpus.py  # Document processing and chunking
│   │   └── vectorize.py   # Generate embeddings and indices
│   └── rag/               # RAG system components
│       ├── config/        # Configuration files
│       │   └── retriever.yaml  # Main system configuration
│       └── retriever/     # Retrieval components
│           ├── hybrid_retriever.py  # Main retriever implementation
│           ├── query_parser.py      # Query parsing utilities
│           ├── cli.py               # Command-line interface
│           └── analyze_corpus.py    # Corpus analysis tools
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Key Files and Their Roles

### Core Retrieval System

#### `src/rag/retriever/hybrid_retriever.py`
**Main retrieval engine** - This is the heart of the system that:
- Combines dense (FAISS) and sparse (BM25) retrieval methods
- Implements intelligent filtering based on program names and categories
- Uses neural reranking to improve result quality
- Handles query enhancement and result boosting

#### `src/rag/retriever/query_parser.py`
**Query understanding module** - Extracts structured information from natural language queries:
- Identifies program names (e.g., "Information Engineering" → "information-engineering")
- Detects degree types (Bachelor, Master, PhD)
- Categorizes query intent (apply, keydata, info)
- Uses regex patterns and keyword matching

#### `src/rag/retriever/cli.py`
**Command-line interface** - User-friendly way to interact with the system:
- Provides various options (--verbose, --limit, --no-filter, etc.)
- Formats and displays results with visual indicators
- Supports output to JSON files for further processing

### Data Processing Pipeline

#### `src/data_collection/scraper.py`
**Data collection** - Gathers information from TUM websites:
- Scrapes program pages and documentation
- Handles different page structures and formats
- Saves raw data in structured JSON format

#### `src/data_collection/prepare_corpus.py`
**Document processing** - Converts raw data into searchable chunks:
- Segments long documents into manageable pieces
- Adds metadata (program, degree, category, section)
- Prepares data for embedding generation

#### `src/data_collection/vectorize.py`
**Index generation** - Creates the search indices:
- Generates OpenAI embeddings for semantic search
- Builds FAISS vector index for fast similarity search
- Prepares metadata for document retrieval

#### `src/rag/build_bm25.py`
**BM25 index builder** - Creates keyword search index:
- Tokenizes document content
- Builds BM25 statistical model
- Saves index for sparse retrieval

### Configuration and Analysis

#### `src/rag/config/retriever.yaml`
**System configuration** - Controls all system parameters:
- Model names and paths
- Retrieval parameters (number of results, thresholds)
- Index locations and settings

#### `src/rag/retriever/analyze_corpus.py`
**Debugging and analysis tool** - Helps understand corpus content:
- Examines document distribution by program and category
- Identifies potential data quality issues
- Assists in debugging retrieval problems

## Usage

### Command Line Interface

The system provides a simple CLI to query program information:

```bash
# Basic query
python src/rag/retriever/cli.py --query "What are the requirements for Computer Science master program?"

# Save results to a file
python src/rag/retriever/cli.py --query "Credits for Informatics bachelor" --output results.json

# Limit number of results
python src/rag/retriever/cli.py --query "How to apply for Mathematics master" --limit 5

# Debug mode with detailed logging
python src/rag/retriever/cli.py --query "Deadline for Data Engineering master application" --verbose

# Disable filtering for broader results
python src/rag/retriever/cli.py --query "machine learning courses" --no-filter

# Only exact program matches
python src/rag/retriever/cli.py --query "Information Engineering credits" --exact-match
```

### Python API

You can also use the system programmatically:

```python
from src.rag.retriever.hybrid_retriever import HybridRetriever
from src.rag.retriever.query_parser import parse_query

# Initialize retriever
retriever = HybridRetriever("src/rag/config/retriever.yaml")

# Parse query
query = "What courses are in the Computer Science master program?"
filters = parse_query(query)

# Retrieve results
results = retriever.retrieve(query, filters)

# Process results
for score, (doc, base_score) in results:
    print(f"Score: {score}, Base: {base_score}")
    print(f"Program: {doc.metadata.get('program')}")
    print(doc.page_content[:100] + "...")
    print("-" * 80)
```

## Getting Started for New Contributors

### 1. Understanding the Data Flow
1. **Raw Data** (`data/raw/`) → **Processing** (`prepare_corpus.py`) → **Processed Chunks** (`data/processed/`)
2. **Processed Chunks** → **Vectorization** (`vectorize.py`) → **Search Indices** (`data/embeddings/`)
3. **User Query** → **Query Parser** → **Hybrid Retriever** → **Results**

### 2. Making Your First Query
```bash
# Test the system with a simple query
python src/rag/retriever/cli.py --query "Information Engineering credits" --verbose
```

### 3. Adding New Features
- **Query parsing**: Modify `query_parser.py` to handle new query types
- **Retrieval logic**: Extend `hybrid_retriever.py` for new filtering or ranking strategies
- **Data processing**: Update `prepare_corpus.py` for new data sources or formats

### 4. Debugging and Analysis
```bash
# Analyze corpus content
python src/rag/retriever/analyze_corpus.py

# Test with different parameters
python src/rag/retriever/cli.py --query "your query" --no-filter --verbose
```

## Configuration

The system uses YAML files for configuration:

```yaml
# src/rag/config/retriever.yaml
index_dir: ../../data/embeddings      # Index directory
embed_model: text-embedding-3-small   # OpenAI embedding model
reranker_model: BAAI/bge-reranker-base # HuggingFace reranker
n_dense: 40      # Number of dense retrieval results
n_sparse: 40     # Number of BM25 retrieval results
top_m: 12        # Number of results to rerank
top_k: 8         # Number of final results to return
```

## Data Preparation

1. **Collect** program information from TUM website:
   ```bash
   python src/data_collection/scraper.py
   ```

2. **Process** documents into searchable chunks:
   ```bash
   python src/data_collection/prepare_corpus.py
   ```

3. **Generate** embeddings and indices:
   ```bash
   python src/data_collection/vectorize.py
   python src/rag/build_bm25.py
   ```

## Troubleshooting

### Common Issues
- **"No results found"**: Try using `--no-filter` flag or more general terms
- **API errors**: Check your `OPENAI_API_KEY` environment variable
- **File not found**: Ensure you're running commands from the project root directory

### Performance Tips
- Use `--limit` to reduce response time for testing
- Enable `--verbose` for debugging retrieval issues
- Use `analyze_corpus.py` to understand data distribution

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with various queries
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
