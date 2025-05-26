import pickle, re, pathlib, yaml, logging
from typing import List, Tuple, Dict, Optional
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, cfg_path: str = "../config/retriever.yaml"):
        try:
            # Get project root directory (parent of src)
            root_dir = pathlib.Path(__file__).parent.parent.parent.parent
            # Load configuration
            cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
            # Convert relative paths to absolute paths (relative to project root)
            index_dir = root_dir / "data" / "embeddings"
            cfg["index_dir"] = str(index_dir)
            self.cfg = cfg
            
            # dense vectordb
            self.emb = OpenAIEmbeddings(model=cfg["embed_model"])
            self.vdb = FAISS.load_local(cfg["index_dir"], self.emb,
                                        allow_dangerous_deserialization=True)
            # bm25
            try:
                with open(pathlib.Path(cfg["index_dir"]) / "bm25.pkl", "rb") as f:
                    pack = pickle.load(f)
                self.bm25: BM25Okapi = pack["bm25"]
                self.docs             = pack["docs"]
            except FileNotFoundError:
                logger.error(f"BM25 index file not found at {cfg['index_dir']}/bm25.pkl")
                raise
            # reranker
            self.rerank = CrossEncoder(cfg["reranker_model"])
        except Exception as e:
            logger.error(f"Error initializing HybridRetriever: {str(e)}")
            raise

    # -------- helpers --------
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _bm25_search(self, query: str, k: int) -> List[Tuple[object, float]]:
        tok = self._tokenize(query)
        scores = self.bm25.get_scores(tok)
        idxs = scores.argsort()[-k:][::-1]
        mx = scores.max() or 1
        return [(self.docs[i], scores[i] / mx) for i in idxs]

    # -------- main entry --------
    def retrieve(self, query: str, filters: Dict[str, str]) -> List[Tuple[float, Tuple[object, float]]]:
        """
        Hybrid retrieval: Dense + BM25 → Filter → Cross-Encoder reranking
        
        Args:
            query: User query text
            filters: Dictionary of filters extracted from query
            
        Returns:
            List[Tuple[ce_score, (Document, base_score)]]: Reranked documents with scores
        """
        try:
            # Check if this is an application-related query with a specific program
            is_apply_query = ("apply" in query.lower() or 
                              "application" in query.lower() or 
                              "admission" in query.lower() or
                              "how to" in query.lower())
            
            program_name = filters.get("slug", "").replace("-", " ")
            
            # Special case search for program application documents
            if is_apply_query and program_name:
                logger.info(f"Special case: Application query detected for program: {program_name}")
                # First try to find direct application documents for the program
                application_docs = self._find_program_application_docs(program_name)
                
                if application_docs:
                    logger.info(f"Found {len(application_docs)} application documents for {program_name}")
                    # Map to format [(score, (doc, base_score)), ...]
                    results = [(1.0, (doc, 1.0)) for doc in application_docs]
                    # Limit to top_k
                    top_k = self.cfg.get("top_k", 8)
                    return results[:top_k]
                else:
                    logger.info(f"No specific application documents found for {program_name}, continuing with regular search")
            
            # Add program name to query if extracted but not in query
            enhanced_query = query
            if filters.get("slug") and filters["slug"].replace("-", " ") not in query.lower():
                enhanced_query = f"{query} {filters['slug'].replace('-', ' ')}"
                logger.info(f"Enhanced query: '{enhanced_query}'")
            
            # 1) Dense retrieval
            dense = self.vdb.similarity_search_with_score(enhanced_query, k=self.cfg["n_dense"])
            dense = [(d, 1 - s) for d, s in dense]           # Distance → Similarity 0-1

            # 2) Sparse retrieval
            sparse = self._bm25_search(enhanced_query, self.cfg["n_sparse"])

            # 3) Merge taking maximum score
            merged: Dict[str, Tuple[object, float]] = {}
            for doc, score in dense + sparse:
                key = doc.metadata.get("id") or id(doc)      # Prefer id; otherwise object address
                if key not in merged or score > merged[key][1]:
                    merged[key] = (doc, score)

            # ---------- DEBUG ----------
            logger.info(f"—— DEBUG ——\ndense raw: {len(dense)}  sparse raw: {len(sparse)}")
            logger.info(f"merged raw: {len(merged)}")

            # 4) Filter and boost exact matches
            filtered_boosted = self._apply_filters_with_boosting(merged, filters)
            if not filtered_boosted:
                logger.warning("After filter: 0 - no candidate left")
                return []

            # Log top results before reranking
            if filtered_boosted:
                top_doc = filtered_boosted[0][0]
                logger.info(f"After filter: {len(filtered_boosted)}, first: {top_doc.metadata.get('program', 'unknown')} | {top_doc.metadata.get('category', '-')} | {top_doc.metadata.get('section', '-')}")

            # 5) Cross-Encoder reranking with exact match prioritization
            return self._rerank_with_exact_priority(query, filtered_boosted, filters)
            
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}")
            return []

    def _apply_filters_with_boosting(self, merged_results: Dict[str, Tuple[object, float]], 
                      filters: Dict[str, str]) -> List[Tuple[object, float]]:
        """Apply filters to the merged results and boost exact matches"""
        
        def ok(meta, boost_factor=1.0):
            # If no metadata, skip filter check
            if not meta:
                return False, 1.0
                
            # Get the program/slug from metadata for matching
            meta_slug = meta.get("slug", "").lower()
            program_name = meta.get("program", "").lower() 
            meta_category = meta.get("category", "").lower()
            meta_section = meta.get("section", "").lower()
            
            # Track match quality to calculate boost
            program_match_quality = 0.0
            category_match_quality = 0.0
            
            # Check program/slug match
            if filters.get("slug"):
                slug_query = filters["slug"].lower().replace('-', ' ')
                
                # Exact matches get highest program match quality
                if slug_query == meta_slug or slug_query == program_name:
                    program_match_quality = 1.0
                # Partial but good matches get medium program match quality
                elif slug_query in meta_slug or slug_query in program_name:
                    program_match_quality = 0.8
                # Word-level matches get lower program match quality
                else:
                    terms = slug_query.split()
                    if any(term in program_name for term in terms):
                        # More matching terms = higher quality
                        matching_terms = sum(1 for term in terms if term in program_name)
                        match_ratio = matching_terms / len(terms)
                        program_match_quality = match_ratio * 0.6
            
            # Check for category match - give this high importance
            if filters.get("category"):
                wanted = {c.strip().lower() for c in filters["category"].split(",")}
                
                # Direct category match
                if meta_category and meta_category in wanted:
                    category_match_quality = 1.0
                # Section match (slightly lower quality)
                elif meta_section and any(cat in meta_section for cat in wanted):
                    category_match_quality = 0.8
                # Contains a keyword from category
                elif any(cat in meta_category for cat in wanted) or any(cat in meta_section for cat in wanted):
                    category_match_quality = 0.6
                # Look for category keywords in content (lower priority)
                elif doc.page_content and any(cat in doc.page_content.lower() for cat in wanted):
                    category_match_quality = 0.4
            else:
                # If no category filter, don't penalize
                category_match_quality = 0.5
                
            # Calculate final match quality - prioritize exact program+category matches
            # Program match should always be more important
            match_quality = 0.0
            
            if program_match_quality > 0:
                # For matching programs, significantly boost documents with matching categories
                if category_match_quality > 0.7:
                    # When we have a specific category and a matching program, this should be top priority
                    match_quality = program_match_quality * 1.5 + category_match_quality
                else:
                    match_quality = program_match_quality + category_match_quality * 0.3
            else:
                # For non-matching programs, still include but with lower quality
                if category_match_quality > 0.8:
                    # High-quality category match for non-matching program (generic info)
                    match_quality = category_match_quality * 0.5
                else:
                    # Low quality match all around
                    match_quality = 0.1
            
            # Decide if document passes filters and calculate boost
            if program_match_quality > 0 or (filters.get("category") and category_match_quality > 0.5):
                # Calculate a boost factor based on match quality (1.0-3.0)
                boost = 1.0 + match_quality * 2.0
                return True, boost
            
            # If we have filters but nothing matched well, reject
            if filters.get("slug") or filters.get("category"):
                return False, 1.0
                
            # Default pass for documents when no filters active
            return True, 1.0

        # Apply filters and collect results with boost factors
        results_with_boost = []
        for doc, score in merged_results.values():
            passed, boost = ok(doc.metadata)
            if passed:
                results_with_boost.append((doc, score * boost))
        
        # If filtering removed everything but we had filters, try with relaxed filters
        if not results_with_boost and filters.get("slug"):
            logger.warning("All results filtered out. Trying with relaxed filters...")
            for doc, score in merged_results.values():
                meta = doc.metadata
                if not meta:
                    continue
                
                # Very relaxed matching for program name
                slug_query = filters["slug"].lower().replace('-', ' ')
                program_name = meta.get("program", "").lower()
                
                # Check for any significant word overlap
                for word in slug_query.split():
                    if len(word) > 3 and word in program_name:
                        # Add but with lower score
                        results_with_boost.append((doc, score * 0.8))
                        break
        
        # Sort by boosted score
        results_with_boost.sort(key=lambda x: x[1], reverse=True)
        
        if results_with_boost:
            top_doc = results_with_boost[0][0]
            logger.info(f"Found {len(results_with_boost)} results, top: {top_doc.metadata.get('program', 'unknown')} | {top_doc.metadata.get('category', '-')} | {top_doc.metadata.get('section', '-')}")
        
        return results_with_boost
        
    def _rerank_with_exact_priority(self, query: str, filtered_results: List[Tuple[object, float]], 
                                   filters: Dict[str, str]) -> List[Tuple[float, Tuple[object, float]]]:
        """Rerank filtered results using cross-encoder but prioritize exact program+category matches"""
        try:
            # First, separate documents by match quality
            exact_program_category_matches = []
            exact_program_matches = []
            other_matches = []
            
            target_program = filters.get("slug", "").lower().replace("-", " ")
            target_category = filters.get("category", "").lower()
            
            # Check if we have program and category filters
            has_program_filter = bool(target_program)
            has_category_filter = bool(target_category)
            
            if has_program_filter or has_category_filter:
                for doc, score in filtered_results:
                    meta = doc.metadata
                    program = meta.get("program", "").lower()
                    category = meta.get("category", "").lower()
                    section = meta.get("section", "").lower()
                    
                    # Check for exact program + category match
                    is_program_match = has_program_filter and target_program in program
                    is_category_match = has_category_filter and (
                        category == target_category or 
                        target_category in section or
                        (category == "apply" and "application" in section)
                    )
                    
                    if is_program_match and is_category_match:
                        exact_program_category_matches.append((doc, score))
                    elif is_program_match:
                        exact_program_matches.append((doc, score))
                    else:
                        other_matches.append((doc, score))
            else:
                # If no filters, all are other matches
                other_matches = filtered_results
            
            # Log how many documents in each match tier
            logger.info(f"Exact program+category matches: {len(exact_program_category_matches)}, "
                      f"Program-only matches: {len(exact_program_matches)}, "
                      f"Other matches: {len(other_matches)}")
                      
            # Decide which group to rerank
            top_m = self.cfg.get("top_m", 12)
            results_to_rerank = []
            
            # If we have exact program+category matches, prioritize those
            if exact_program_category_matches:
                results_to_rerank = exact_program_category_matches[:top_m]
                logger.info(f"Using {len(results_to_rerank)} exact program+category matches for reranking")
            # Otherwise use program matches, supplemented with other if needed
            elif exact_program_matches:
                results_to_rerank = exact_program_matches[:top_m]
                logger.info(f"Using {len(results_to_rerank)} program matches for reranking")
            # Otherwise use other matches
            else:
                results_to_rerank = other_matches[:top_m]
                logger.info(f"Using {len(results_to_rerank)} other matches for reranking")
                
            # Perform reranking on the selected subset
            if results_to_rerank:
                pairs = [[query, doc.page_content] for doc, _ in results_to_rerank]
                logits = self.rerank.predict(pairs, batch_size=8, convert_to_numpy=True)
                
                # Convert to list of (score, (doc, base_score)) for final sorting
                reranked = []
                for i, (logit, (doc, base)) in enumerate(zip(logits, results_to_rerank)):
                    reranked.append((float(logit), (doc, base)))
                
                # Sort by reranking score
                reranked.sort(reverse=True)
                
                # Limit to top-k
                top_k = self.cfg.get("top_k", 8)
                results = reranked[:top_k]
                
                logger.info(f"Reranked {len(results_to_rerank)} documents, returning top {len(results)}")
                
                # Add in extra high-priority documents if needed
                if len(exact_program_category_matches) > 0 and not any(doc.metadata.get("program", "").lower() == target_program and 
                                              doc.metadata.get("category", "").lower() == "apply" 
                                              for _, (doc, _) in results[:3]):
                    # Find a good apply document to insert at position 1
                    for doc, base in exact_program_category_matches:
                        if doc.metadata.get("category", "").lower() == "apply":
                            # Insert at position 1 with artificially high score
                            logger.info(f"Inserting application document at position 1: {doc.metadata.get('section', '')}")
                            results.insert(0, (1.0, (doc, base)))
                            results = results[:top_k]  # Keep to top_k
                            break
                
                return results
            else:
                logger.warning("No documents to rerank")
                return []
                
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            # Fall back to base scores if reranking fails
            sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)
            top_k = self.cfg.get("top_k", 8)
            return [(item[1], item) for item in sorted_results[:top_k]]

    def _find_program_application_docs(self, program_name: str) -> List[object]:
        """Directly find application documents for a specific program by metadata"""
        application_docs = []
        application_section_terms = [
            "application_process", 
            "admission_process", 
            "application_deadlines", 
            "documents_required",
            "documents_required_for_online_application",
            "documents_required_for_enrollment",
            "application_period",
            "application"
        ]
        
        # Ensure program name is lowercase for matching
        program_name = program_name.lower()
        
        for doc in self.docs:
            meta = doc.metadata
            
            # Check for program match
            doc_program = meta.get("program", "").lower()
            # Check for either exact match or if the program name is contained in the document program
            if not (program_name == doc_program or program_name in doc_program):
                continue
                
            # Check for application-related section
            section = meta.get("section", "").lower()
            if any(term in section for term in application_section_terms):
                application_docs.append(doc)
                logger.info(f"Found {doc_program} application document: {section}")
                
            # Also check category
            category = meta.get("category", "").lower()
            if category == "apply" and section:
                application_docs.append(doc)
                logger.info(f"Found {doc_program} application document (category apply): {section}")
        
        # Sort by section relevance - application process first, then documents, then deadlines
        application_docs.sort(key=lambda doc: (
            "application_process" in doc.metadata.get("section", "").lower(),
            "admission" in doc.metadata.get("section", "").lower(),
            "deadline" in doc.metadata.get("section", "").lower(),
            "document" in doc.metadata.get("section", "").lower()
        ), reverse=True)
        
        # Remove duplicates while preserving order
        unique_docs = []
        seen_sections = set()
        for doc in application_docs:
            section = doc.metadata.get("section", "")
            if section not in seen_sections:
                unique_docs.append(doc)
                seen_sections.add(section)
                
        return unique_docs

