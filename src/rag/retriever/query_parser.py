import re
import logging
from typing import Dict, List, Set, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Program degrees
DEGREE_MAP = {
    r"\bmaster\b|\bmsc\b|\bma\b|\bm\.sc\.?\b|\bm\.a\.?\b": "msc",
    r"\bbachelor\b|\bbsc\b|\bba\b|\bb\.sc\.?\b|\bb\.a\.?\b": "bsc",
    r"\bphd\b|\bdoctor|\bdoctoral\b|\bdr\.?\b": "phd",
    r"\bmaster of arts\b|\bm\.a\.?\b": "ma", 
    r"\bbachelor of arts\b|\bb\.a\.?\b": "ba"
}

# Program categories based on action/intent keywords
ACTION_CAT = {
    "apply": [
        # Application process
        "apply", "application", "submit", "admission", "admissions", "enroll", "enrollment",
        "register", "registration", "matriculation", "tum online", "tumonline",
        
        # Application documents
        "documents", "document", "certificate", "certificates", "transcript", "transcripts",
        
        # Application requirements
        "requirements", "qualification", "qualifications", "eligibility", "prerequisites",
        "required", "require", "needed", "need",
        
        # Application timeline
        "deadline", "deadlines", "dates", "timeline", "when", "date", "schedule",
        
        # Application status
        "status", "decision", "acceptance", "admit", "admitted", "reject", "rejected",
        "confirm", "confirmation",
        
        # Application fee
        "fee", "fees", "payment", "pay", "cost", "costs", "charges"
    ],
    "keydata": [
        "credit", "credits", "language", "languages", "fees", "cost", "duration", "ects",
        "workload", "hours", "semesters", "years", "timeframe", "tuition", "scholarship",
        "scholarships", "stipend", "funding", "financial", "statistics", "key facts", 
        "key data", "key figures", "at a glance", "overview", "summary"
    ],
    "info": [
        "structure", "curriculum", "modules", "courses", "syllabus", "content", "overview",
        "description", "professors", "lecturers", "faculty", "subjects", "topics", "specialization",
        "about", "introduction", "details", "outline", "program structure", "study plan",
        "curriculum", "syllabus", "coursework", "research", "thesis", "project", "internship",
        "practical", "laboratory", "lab", "workshop", "seminar", "lecture", "class", "teaching",
        "learning", "study", "studies", "education", "academic", "knowledge", "skills", "competencies"
    ]
}

# Flatten category lookup for faster access
CATEGORY_LOOKUP = {}
for category, terms in ACTION_CAT.items():
    for term in terms:
        CATEGORY_LOOKUP[term] = category

def parse_query(q: str) -> Dict[str, str]:
    """
    Parse a natural language query to extract structured filters.
    
    Args:
        q: User query string
        
    Returns:
        Dictionary with the following keys:
            - slug: Program name/identifier
            - degree: Degree type (msc, bsc, phd, etc.)
            - category: Information category (apply, keydata, info)
    """
    try:
        q_low = q.lower()
        
        # 1. Extract degree information
        degree = ""
        for pat, deg in DEGREE_MAP.items():
            if re.search(pat, q_low):
                degree = deg
                break

        # 2. Extract category based on action/intent keywords
        # Special case: "How to apply" or similar application-focused queries
        if re.search(r'\b(how|where|when)\s+to\s+(apply|register|submit|enroll)', q_low) or \
           re.search(r'\b(application|admission)s?\s+(process|procedure|steps|requirements)', q_low) or \
           re.search(r'\b(what|which)\s+(documents|papers|certificates).+\b(need|require)', q_low):
            category = "apply"
            logger.debug(f"Detected application intent from query pattern: '{q}'")
        else:
            # Standard category detection
            category_matches = []
            for word in re.findall(r'\b\w+\b', q_low):
                if word in CATEGORY_LOOKUP:
                    category_matches.append(CATEGORY_LOOKUP[word])
                    
            # Use the most frequent category, or empty if none found
            category = _most_frequent(category_matches) if category_matches else ""

        # 3. Extract program name/slug using different patterns
        slug = _extract_program_slug(q)

        result = {"slug": slug, "degree": degree, "category": category}
        logger.debug(f"Parsed query '{q}' to filters: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error parsing query '{q}': {str(e)}")
        return {"slug": "", "degree": "", "category": ""}

def _extract_program_slug(query: str) -> str:
    """
    Extract program name from query using multiple patterns.
    
    Args:
        query: User query string
        
    Returns:
        Normalized program slug or empty string if not found
    """
    # Special case patterns for common programs
    special_programs = {
        r"\binformation\s+engineering\b": "information-engineering",
        r"\bcomputer\s+science\b": "computer-science",
        r"\bdata\s+engineering\b": "data-engineering",
        r"\binformatics\b": "informatics",
        r"\bmathematics\b": "mathematics",
        r"\belectrical\s+engineering\b": "electrical-engineering"
    }
    
    # First check for exact program names
    query_lower = query.lower()
    for pattern, slug in special_programs.items():
        if re.search(pattern, query_lower, re.I):
            logger.debug(f"Matched special program pattern '{pattern}' -> '{slug}'")
            return slug
    
    # Regular patterns
    patterns = [
        # Pattern for "for X master/bachelor"
        r'for\s+([A-Z][\w\s&-]+?)\s+(?:master|bachelor|msc|bsc|phd|ma|ba)',
        # Pattern for "X program/degree/master"
        r'([A-Z][\w\s&-]+?)\s+(?:program|degree|master|bachelor|course|study)',
        # Pattern for "studying/taking X"
        r'(?:studying|taking|about)\s+([A-Z][\w\s&-]+?)(?:\s+|$)',
        # Pattern for "X at TUM"
        r'([A-Z][\w\s&-]+?)\s+(?:at|in|from)\s+(?:TUM|Technical University)',
        # Fallback pattern - try to find any capitalized phrase
        r'\b([A-Z][\w\s&-]{2,}?)\b'
    ]
    
    for pattern in patterns:
        m = re.search(pattern, query, re.I)
        if m:
            # Normalize slug: lowercase, replace non-word chars with hyphens
            return re.sub(r'\W+', '-', m.group(1).strip().lower())
    
    # Last resort - look for technical terms
    technical_keywords = ["engineering", "science", "informatics", "mathematics", "physics", "chemistry"]
    for kw in technical_keywords:
        if kw in query_lower:
            # Look for words around the keyword
            context = re.search(r'(\w+\s+)?' + kw + r'(\s+\w+)?', query_lower)
            if context:
                match = context.group(0).strip()
                return re.sub(r'\W+', '-', match)
    
    return ""

def _most_frequent(items: List[str]) -> str:
    """Return the most frequently occurring item in a list, or empty string for empty list"""
    if not items:
        return ""
        
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
        
    return max(counts.items(), key=lambda x: x[1])[0]
