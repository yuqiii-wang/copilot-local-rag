"""
feature_weights.py

Centralized configuration for feature weighting logic in the Knowledge Graph.
This file defines weights for:
- HTML structure (titles, headers, etc.)
- QA Status (accepted, pending, rejected)
- N-gram length boosting
- Code dependency priors
- Recency boosting
- Model training loss weights
"""

from typing import Dict, Any, Union

# HTML Structure Weights
# Used in feature_generator.py to boost tokens found in specific HTML tags.
# Higher weight for titles and headers emphasizes document structure.
HTML_TAG_WEIGHTS = {
    'h1': 5.0,
    'h2': 4.0,
    'h3': 3.0,
    'title': 10.0,
    'strong': 3.0,
    'b': 3.0,
    'em': 2.0,
    'i': 2.0,
    'th': 2.0
}

# QA / Conversation Status Weights
# Used in feature_generator.py and train_model.py when injecting priors.
STATUS_WEIGHTS = {
    'accepted': 10.0,        # Trusted human validation
    'added_confluence': 5.0, # Semi-trusted documentation update
    'pending': 1.0,          # Default / unknown state
    'rejected': 0.2,         # Explicitly down-weighted
    'human_msg': 10.0,       # Human messages in conversation logs (stronger signal)
    'ai_msg': 1.0            # AI responses in conversation logs (weaker signal)
}

# QA Component Boosts
# Multipliers applied on top of status weights for specific field types in QA/Records.
QA_COMPONENT_BOOSTS = {
    'comment': 3.0,      # Comments often explain *why* something is relevant
    'keywords': 20.0,    # Explicit keywords tagged by users - STRONG PRIORITY
    'question': 1.0      # The raw question text
}

# Position Decay Rules for QA References
# Used to weight reference documents in a list (1st doc is most relevant).
QA_POSITION_DECAY = {
    'start': 1.0,
    'step': 0.2,
    'floor': 0.2
}

# Code Graph Priors
# Weights for dependency propagation in train_model.py.
CODE_GRAPH_WEIGHTS = {
    'hard_link': 2.0,    # Direct dependency import (e.g. Java 'import', C++ '#include')
    'decay_factor': 0.5  # Decay for indirect/soft links
}

# Structural Pattern Weights
# Boost for shared structural patterns (like matching ISINs, UUIDs, Error Codes).
# Used in train_model.py.
PATTERN_WEIGHT = 5.0

# Filename Weights
# Boost for tokens found in the filename (very important).
FILENAME_WEIGHT = 20.0

# Recency Weighting
# Max percentage boost for most recent files within the dataset.
# Used in feature_generator.py.
RECENCY_MAX_BOOST = 0.05  # 5% boost for newest files

# Training / Loss Weights
# Used in the GNN training loop in train_model.py.
TRAINING_WEIGHTS = {
    'non_zero_loss_multiplier': 20.0, # Penalize missing real connections 20x more than 0s
    'missed_doc_penalty': 100.0,      # Extra penalty weight if a known document link is predicted as ~0
    'inference_threshold': 0.5        # Minimum score to be considered a valid link in CSV output
}

# Expansion Weights
# Used in keyword_expander.py to determine expansion weight based on edge score.
EXPANSION_WEIGHTS = {
    'base': 0.3,
    'thresholds': [
        (8.0, 2.5), # Score threshold, Target Weight (or base for dynamic calc)
        (5.0, 1.8),
        (3.0, 1.0),
        (1.0, 0.8)
    ],
    'high_score_multiplier': 0.25 # For scores >= 8.0, weight = score * this
}

# Inference / Query Weights
# Used in knowledge_graph_service.py during query processing.
INFERENCE_WEIGHTS = {
    'topic_feature_scale': 10.0,      # Scaling factor for NMF topic features
    'identifier_boost': 3.0,          # Boost for identifiers (digits/uppercase > 4)
    'result_min_score': 0.01,         # Absolute minimum score to return
    'result_ratio_threshold': 0.1,    # Only return results with score >= top_score * this
    'word_count_power': 3.0,          # Power for word count boosting (count ^ 3)
    'shared_match_boost_log': True,   # Whether to apply log1p(shared_counts) boost
    'apply_idf_boost': True,          # Apply global Inverse Document Frequency boost
    'idf_scale': 1.0                  # Scaling factor for IDF (matched_weight *= idf * scale)
}

# Config / Priors
PRIOR_INJECTION_WEIGHT = 10.0   # Weight for injected QA priors in train_model.py

# Latent Edge Construction Weights
# Used in keyword_expander.py
LATENT_WEIGHTS = {
    'prefix_match': 10.0,    # Strong boost for prefix matches (sec -> security)
    'consonant_match': 5.0, # Strong boost for consonant matches
    'base_latent': 0.1       # Base weight for co-occurrence without pattern match
}

# --- Logic / Helper Functions ---

def get_expansion_weight(score: float) -> float:
    """
    Calculates the weight for a keyword expansion based on the edge score.
    """
    w = EXPANSION_WEIGHTS['base']
    
    # Check thresholds in descending order
    # Currently hardcoded in the loop, but logic reflects:
    # >= 8.0: Dynamic
    # >= 5.0: 1.8
    # >= 3.0: 1.0
    # >= 1.0: 0.8
    
    # Use the defined constants
    thresholds = sorted(EXPANSION_WEIGHTS['thresholds'], key=lambda x: x[0], reverse=True)
    
    for thresh, weight_val in thresholds:
        if score >= thresh:
            if thresh >= 8.0:
                # Dynamic calculation for very high scores
                return max(weight_val, score * EXPANSION_WEIGHTS['high_score_multiplier'])
            return weight_val
            
    return w

def get_html_weight(tag: str) -> float:
    """Returns the weight for a specific HTML tag, default 1.0."""
    return HTML_TAG_WEIGHTS.get(tag, 1.0)

def calculate_html_weight(tag_info: Union[Dict[str, Any], str], token_count: int = 0) -> float:
    """
    Calculates weight for an HTML segment based on tag, attributes, and content length.
    Handles extended parsing rules:
    - Non-standard colors
    - Short table entries
    - Anchors with links
    - Custom ri:: tags
    - Attachments
    """
    if isinstance(tag_info, str):
        return get_html_weight(tag_info)
        
    tag_name = tag_info.get('name', '').lower()
    attrs = tag_info.get('attrs', {})
    
    # Base weight
    weight = HTML_TAG_WEIGHTS.get(tag_name, 1.0)
    
    # 1. Anchors with href (Valid Links)
    if tag_name == 'a' and 'href' in attrs:
        weight = max(weight, 3.0)

    # 2. Table entries (cols/rows) with short length (likely headers or key-values)
    if tag_name in ['td', 'th', 'tr']:
        if 0 < token_count <= 5: 
             weight = max(weight, 4.0)

    # 3. ri:: tags (Custom Important Tags)
    if tag_name.startswith('ri:') or tag_name.startswith('ri::'):
        weight = max(weight, 5.0)

    # 4. Attachments (Tag or Attribute)
    if tag_name == 'attachment' or 'attachment' in str(attrs.get('class', '')).lower():
         weight = max(weight, 5.0)

    # 5. Important Attributes or Colors
    # Check for style overrides or specific keywords
    style = str(attrs.get('style', '')).lower()
    attr_str = str(attrs).lower()
    
    if 'color' in style: # simplified check for non-standard color presence
        weight = max(weight, 3.0)
        
    if 'important' in attr_str or 'priority' in attr_str:
        weight = max(weight, 4.0)
            
    return weight

def get_ngram_boost(token_text: str) -> float:
    """
    Calculates importance boost for n-grams based on length.
    Logic: Exponential boost to strongly favor longest matches.
    """
    length = len(token_text.split())
    if length <= 1:
        return 1.0
    # Exponential boost: 3.5^length.
    # L=1: 1.0
    # L=2: 12.25 (vs 5.6 previously)
    # L=3: 42.8 (vs 15.6 previously)
    return 3.5 ** float(length)

def get_qa_sequence_boost(ngram_len: int, base_weight: float = 1.0) -> float:
    """
    Calculates power boost for sequences during QA prior injection.
    Used in train_model.py to heavily weight long matching phrases from questions.
    Logic: base_weight * (5.0 ^ ngram_len)
    """
    return base_weight * (5.0 ** float(ngram_len))

def calculate_qa_position_weight(index: int, base_multiplier: float) -> float:
    """
    Calculates weight for a reference document based on its position in the list.
    weight = base_multiplier * max(floor, start - index * step)
    """
    pos_mult = max(
        QA_POSITION_DECAY['floor'], 
        QA_POSITION_DECAY['start'] - index * QA_POSITION_DECAY['step']
    )
    return base_multiplier * pos_mult
