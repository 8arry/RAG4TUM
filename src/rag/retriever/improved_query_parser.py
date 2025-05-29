import re
import logging
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Structured representation of query intent"""
    program: str = ""
    degree: str = ""
    category: str = ""
    intent_type: str = ""  # specific, general, comparison
    entities: List[str] = None
    temporal_keywords: List[str] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = []
        if self.temporal_keywords is None:
            self.temporal_keywords = []

# Enhanced program detection patterns
PROGRAM_PATTERNS = {
    r"\b(information|informatik)\s+(engineering|ingenieurwesen)\b": "information-engineering",
    r"\b(computer|computing)\s+(science|informatics|informatik)\b": "computer-science", 
    r"\b(data|daten)\s+(engineering|science|wissenschaft)\b": "data-engineering",
    r"\b(electrical|elektro)\s+(engineering|technik)\b": "electrical-engineering",
    r"\b(mechanical|maschinen)\s+(engineering|bau)\b": "mechanical-engineering",
    r"\b(bioprocess|bio)\s+(engineering|technik)\b": "bioprocess-engineering",
    r"\binformatics?\b": "informatics",
    r"\bmathematics?\b": "mathematics",
    r"\bphysics?\b": "physics"
}

# Intent-based category mapping with semantic understanding
SEMANTIC_CATEGORIES = {
    "deadline_intent": {
        "keywords": ["deadline", "due", "when", "date", "period", "timeline", "schedule", "time"],
        "category": "apply",
        "priority": "high"
    },
    "application_intent": {
        "keywords": ["apply", "application", "submit", "admission", "how to", "process", "procedure"],
        "category": "apply", 
        "priority": "high"
    },
    "document_intent": {
        "keywords": ["documents", "papers", "certificates", "transcripts", "requirements", "needed"],
        "category": "apply",
        "priority": "medium"
    },
    "program_info_intent": {
        "keywords": ["about", "overview", "description", "what is", "curriculum", "courses", "modules"],
        "category": "info",
        "priority": "medium"
    },
    "cost_duration_intent": {
        "keywords": ["cost", "fee", "price", "duration", "credits", "ects", "semesters", "years"],
        "category": "keydata",
        "priority": "medium"
    }
}

# Temporal keyword detection for deadline queries
TEMPORAL_KEYWORDS = [
    "when", "deadline", "due", "date", "period", "timeline", "schedule", 
    "winter", "summer", "semester", "application period", "until"
]

def enhanced_parse_query(query: str) -> QueryIntent:
    """
    Enhanced query parsing with better semantic understanding
    """
    query_lower = query.lower().strip()
    intent = QueryIntent()
    
    # 1. Extract program with better pattern matching
    intent.program = _extract_program_enhanced(query)
    
    # 2. Extract degree information  
    intent.degree = _extract_degree(query_lower)
    
    # 3. Detect intent type and category using semantic analysis
    intent_result = _detect_semantic_intent(query_lower)
    intent.category = intent_result["category"]
    intent.intent_type = intent_result["type"]
    
    # 4. Extract entities and temporal keywords
    intent.entities = _extract_entities(query)
    intent.temporal_keywords = _extract_temporal_keywords(query_lower)
    
    # 5. Apply intent-based boosting rules
    if intent.temporal_keywords and "deadline" in query_lower:
        intent.category = "apply"  # Force apply category for deadline queries
        intent.intent_type = "specific"
        
    logger.info(f"Enhanced parsing: '{query}' -> Program: {intent.program}, "
                f"Category: {intent.category}, Intent: {intent.intent_type}")
    
    return intent

def _extract_program_enhanced(query: str) -> str:
    """Enhanced program extraction with fuzzy matching"""
    query_clean = re.sub(r'\b(master|bachelor|msc|bsc|phd|degree|program)\b', '', query.lower())
    
    # Check exact patterns first
    for pattern, slug in PROGRAM_PATTERNS.items():
        if re.search(pattern, query_clean, re.I):
            logger.debug(f"Matched program pattern '{pattern}' -> '{slug}'")
            return slug
    
    # Fuzzy matching for common terms
    program_keywords = {
        "information": "information-engineering",
        "computer": "computer-science", 
        "data": "data-engineering",
        "electrical": "electrical-engineering"
    }
    
    for keyword, slug in program_keywords.items():
        if keyword in query_clean:
            # Additional validation - check if it makes sense in context
            if "engineering" in query_clean or "science" in query_clean:
                return slug
    
    return ""

def _extract_degree(query_lower: str) -> str:
    """Extract degree type"""
    degree_patterns = {
        r"\bmaster\b|\bmsc\b|\bm\.sc\.?\b": "msc",
        r"\bbachelor\b|\bbsc\b|\bb\.sc\.?\b": "bsc", 
        r"\bphd\b|\bdoctor|\bdoctoral\b": "phd"
    }
    
    for pattern, degree in degree_patterns.items():
        if re.search(pattern, query_lower):
            return degree
    return ""

def _detect_semantic_intent(query_lower: str) -> Dict[str, str]:
    """Detect semantic intent using keyword analysis"""
    intent_scores = {}
    
    for intent_name, intent_data in SEMANTIC_CATEGORIES.items():
        score = 0
        for keyword in intent_data["keywords"]:
            if keyword in query_lower:
                # Boost score based on keyword importance
                if keyword in ["deadline", "when", "apply", "how to"]:
                    score += 3
                elif keyword in ["documents", "requirements", "cost"]:
                    score += 2
                else:
                    score += 1
        
        if score > 0:
            intent_scores[intent_name] = {
                "score": score,
                "category": intent_data["category"],
                "priority": intent_data["priority"]
            }
    
    if not intent_scores:
        return {"category": "", "type": "general"}
    
    # Select highest scoring intent
    best_intent = max(intent_scores.items(), key=lambda x: x[1]["score"])
    intent_type = "specific" if best_intent[1]["score"] >= 3 else "general"
    
    return {
        "category": best_intent[1]["category"],
        "type": intent_type
    }

def _extract_entities(query: str) -> List[str]:
    """Extract named entities (programs, locations, etc.)"""
    entities = []
    
    # Capitalized words (potential program names)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    entities.extend(capitalized)
    
    # Technical terms
    technical_terms = re.findall(r'\b(engineering|science|informatics|mathematics|physics|chemistry)\b', query.lower())
    entities.extend(technical_terms)
    
    return list(set(entities))

def _extract_temporal_keywords(query_lower: str) -> List[str]:
    """Extract temporal/deadline related keywords"""
    found_temporal = []
    for keyword in TEMPORAL_KEYWORDS:
        if keyword in query_lower:
            found_temporal.append(keyword)
    return found_temporal

def convert_to_legacy_format(intent: QueryIntent) -> Dict[str, str]:
    """Convert enhanced intent back to legacy format for compatibility"""
    return {
        "slug": intent.program,
        "degree": intent.degree, 
        "category": intent.category
    } 