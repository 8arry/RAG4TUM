#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Hybrid Retriever with enhanced query understanding and chunking
"""

import pickle, re, pathlib, yaml, logging, os
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from improved_query_parser import enhanced_parse_query, convert_to_legacy_format
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedHybridRetriever:
    def __init__(self, cfg_path: str = "../config/improved_retriever.yaml"):
        try:
            # Load environment variables
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            
            root_dir = pathlib.Path(__file__).parent.parent.parent.parent
            cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
            index_dir = root_dir / "data" / "embeddings"
            cfg["index_dir"] = str(index_dir)
            self.cfg = cfg
            
            # Initialize embeddings and vector store
            self.emb = GoogleGenerativeAIEmbeddings(model=cfg["embed_model"], google_api_key=api_key)
            self.vdb = FAISS.load_local(cfg["index_dir"], self.emb,
                                        allow_dangerous_deserialization=True)
            
            # Load BM25 index
            try:
                with open(pathlib.Path(cfg["index_dir"]) / "bm25.pkl", "rb") as f:
                    pack = pickle.load(f)
                self.bm25: BM25Okapi = pack["bm25"]
                self.docs = pack["docs"]
            except FileNotFoundError:
                logger.error(f"BM25 index file not found at {cfg['index_dir']}/bm25.pkl")
                raise
            
            # Initialize reranker
            self.rerank = CrossEncoder(cfg["reranker_model"])
            
        except Exception as e:
            logger.error(f"Error initializing ImprovedHybridRetriever: {str(e)}")
            raise

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _expand_query(self, query: str, intent) -> str:
        """Expand query with synonyms and related terms based on intent"""
        if not self.cfg.get("query_expansion", False):
            return query
            
        expanded_terms = []
        
        # Add deadline-specific terms for deadline queries
        if "deadline" in intent.temporal_keywords:
            expanded_terms.extend(["application period", "submission deadline", "due date"])
        
        # Add application-specific terms  
        if intent.category == "apply":
            expanded_terms.extend(["admission", "enrollment", "registration"])
            
        # Add program-specific expansions
        if intent.program:
            # Convert slug back to readable form
            program_readable = intent.program.replace("-", " ").title()
            expanded_terms.append(program_readable)
        
        if expanded_terms:
            expanded_query = f"{query} {' '.join(expanded_terms)}"
            logger.info(f"Expanded query: '{query}' -> '{expanded_query}'")
            return expanded_query
        
        return query

    def _bm25_search(self, query: str, k: int) -> List[Tuple[object, float]]:
        tok = self._tokenize(query)
        scores = self.bm25.get_scores(tok)
        idxs = scores.argsort()[-k:][::-1]
        mx = scores.max() or 1
        return [(self.docs[i], scores[i] / mx) for i in idxs]

    def retrieve(self, query: str, enhanced_parsing: bool = True) -> List[Tuple[float, Tuple[object, float]]]:
        """
        Enhanced retrieval with improved query understanding
        """
        try:
            # Enhanced query parsing
            if enhanced_parsing:
                intent = enhanced_parse_query(query)
                filters = convert_to_legacy_format(intent)
                
                # Query expansion
                expanded_query = self._expand_query(query, intent)
            else:
                from query_parser import parse_query
                filters = parse_query(query)
                expanded_query = query
                intent = None
            
            logger.info(f"Processing query: '{query}' with filters: {filters}")
            
            # Dense retrieval with expanded query
            dense = self.vdb.similarity_search_with_score(
                expanded_query, k=self.cfg["n_dense"]
            )
            dense = [(d, 1 - s) for d, s in dense]
            
            # Sparse retrieval
            sparse = self._bm25_search(expanded_query, self.cfg["n_sparse"])
            
            # Merge results
            merged: Dict[str, Tuple[object, float]] = {}
            for doc, score in dense + sparse:
                key = doc.metadata.get("id") or id(doc)
                if key not in merged or score > merged[key][1]:
                    merged[key] = (doc, score)
            
            logger.info(f"Dense: {len(dense)}, Sparse: {len(sparse)}, Merged: {len(merged)}")
            
            # Enhanced filtering with intent awareness
            filtered_boosted = self._apply_enhanced_filters(merged, filters, intent)
            
            if not filtered_boosted:
                logger.warning("No results after filtering")
                return []
            
            # Enhanced reranking with intent priority
            return self._enhanced_rerank(query, filtered_boosted, filters, intent)
            
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            return []

    def _apply_enhanced_filters(self, merged_results: Dict[str, Tuple[object, float]], 
                               filters: Dict[str, str], intent=None) -> List[Tuple[object, float]]:
        """Enhanced filtering with intent-based boosting"""
        
        def calculate_relevance_score(doc, base_score):
            meta = doc.metadata
            relevance_multiplier = 1.0
            
            # Program matching
            if filters.get("slug"):
                target_program = filters["slug"].lower().replace("-", " ")
                doc_program = meta.get("program", "").lower()
                
                if target_program == doc_program or target_program in doc_program:
                    relevance_multiplier *= self.cfg.get("exact_match_boost", 2.0)
                elif any(word in doc_program for word in target_program.split()):
                    relevance_multiplier *= 1.5
            
            # Category matching with intent awareness
            if filters.get("category"):
                target_category = filters["category"].lower()
                doc_category = meta.get("category", "").lower()
                doc_section = meta.get("section", "").lower()
                
                # Exact category match
                if doc_category == target_category:
                    relevance_multiplier *= self.cfg.get("exact_match_boost", 2.0)
                
                # Special handling for deadline queries
                if (intent and intent.temporal_keywords and 
                    ("deadline" in doc_section or "period" in doc_section)):
                    relevance_multiplier *= self.cfg.get("deadline_priority_boost", 3.0)
                    logger.info(f"Applied deadline boost to: {doc_section}")
                
                # Priority chunk type boosting
                if (self.cfg.get("prefer_specialized_chunks", True) and 
                    meta.get("chunk_type") in ["deadline", "application"]):
                    relevance_multiplier *= 2.0
                    logger.info(f"Applied specialized chunk boost: {meta.get('chunk_type')}")
            
            # Semantic similarity boost
            if base_score > 0.8:  # High semantic similarity
                relevance_multiplier *= self.cfg.get("semantic_boost", 1.5)
            
            return base_score * relevance_multiplier
        
        # Apply enhanced scoring
        results_with_scores = []
        for doc, base_score in merged_results.values():
            enhanced_score = calculate_relevance_score(doc, base_score)
            
            # Apply minimum threshold
            if enhanced_score > 0.1:  # Minimum relevance threshold
                results_with_scores.append((doc, enhanced_score))
        
        # Sort by enhanced scores
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Enhanced filtering: {len(results_with_scores)} results after boosting")
        return results_with_scores

    def _enhanced_rerank(self, query: str, filtered_results: List[Tuple[object, float]], 
                        filters: Dict[str, str], intent=None) -> List[Tuple[float, Tuple[object, float]]]:
        """Enhanced reranking with intent awareness"""
        try:
            top_m = self.cfg.get("top_m", 20)
            candidates = filtered_results[:top_m]
            
            if not candidates:
                return []
            
            # Prepare reranking pairs
            pairs = [[query, doc.page_content] for doc, _ in candidates]
            
            # Get reranking scores
            batch_size = self.cfg.get("reranker_batch_size", 16)
            logits = self.rerank.predict(pairs, batch_size=batch_size, convert_to_numpy=True)
            
            # Combine reranking scores with enhanced scores
            final_results = []
            for i, (logit, (doc, enhanced_score)) in enumerate(zip(logits, candidates)):
                # Combine reranker score with our enhanced score
                final_score = float(logit) + (enhanced_score * 0.1)  # Weight our scoring
                
                # Apply minimum score threshold
                min_score = self.cfg.get("min_rerank_score", -5.0)
                if final_score >= min_score:
                    final_results.append((final_score, (doc, enhanced_score)))
            
            # Sort by final combined score
            final_results.sort(reverse=True)
            
            # Return top-k results
            top_k = self.cfg.get("top_k", 10)
            results = final_results[:top_k]
            
            logger.info(f"Reranked {len(candidates)} candidates, returning {len(results)} results")
            
            # Log top result for debugging
            if results:
                top_doc = results[0][1][0]
                logger.info(f"Top result: {top_doc.metadata.get('program')} | "
                           f"{top_doc.metadata.get('category')} | "
                           f"{top_doc.metadata.get('section')}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in enhanced reranking: {str(e)}")
            # Fallback to original scores
            sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
            top_k = self.cfg.get("top_k", 10)
            return [(item[1], item) for item in sorted_results[:top_k]] 