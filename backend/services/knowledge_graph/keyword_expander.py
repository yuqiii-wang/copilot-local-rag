import os
import json
import re
import pickle
from collections import defaultdict
from typing import List, Dict, Set
from services.knowledge_graph.tokenization import clean_and_tokenize
from services.knowledge_graph import feature_weights

class KeywordExpander:
    def __init__(self):
        # map: token -> {related_token: score}
        self.keyword_edges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.consonant_cache = {}

    def get_consonants(self, text: str) -> str:
        if text in self.consonant_cache:
            return self.consonant_cache[text]
        # Remove vowels
        res = re.sub(r'[aeiouAEIOU]', '', text)
        self.consonant_cache[text] = res
        return res

    def normalize_variant(self, text: str) -> str:
        """
        Simple heuristic normalization for matching variants (plurals, gerunds, etc.)
        """
        if len(text) < 3: return text
        t = text.lower()
        
        # Strip common suffixes (heuristic)
        if t.endswith('ies'): return t[:-3] + 'y'
        if t.endswith('ing'): return t[:-3]
        if t.endswith('tion'): return t[:-4]
        if t.endswith('ment'): return t[:-4]
        if t.endswith('es') and len(t) > 4: return t[:-2]
        if t.endswith('s') and not t.endswith('ss') and len(t) > 3: return t[:-1]
        
        return t

    def build(self, feature_generator, qa_records: List[Dict]):
        """
        Builds the keyword expansion graph based on QA records and document vocabulary.
        """
        print("Building Keyword Expansion Graph...")
        
        # Get vocab set from vectorizer
        try:
            vocab_set = set(feature_generator.vectorizer.get_feature_names_out())
        except:
             # Fallback if fitted vectorizer not available directly (should not happen in train_model)
             print("Warning: feature_generator vectorizer not ready.")
             return

        # Pre-process doc tokens for lookup to avoid re-tokenizing repeatedly
        # doc_name (basename) -> set of tokens (intersection with vocab)
        doc_token_map = {}
        
        # We need doc contents. verify feature_generator has them
        if not hasattr(feature_generator, 'doc_contents') or not feature_generator.doc_contents:
             print("Warning: No doc contents in feature_generator. Skipping latent edge construction.")
             return

        print(f"Indexing {len(feature_generator.doc_names)} documents for keyword linkage...")
        for idx, content in enumerate(feature_generator.doc_contents):
            doc_rel_path = feature_generator.doc_names[idx]
            doc_name = os.path.basename(doc_rel_path)
            
            # Use basic split similar to feature_generator's tokenizer logic
            # clean_and_tokenize might have been applied before?
            # In FG, doc_contents are usually raw text? No, FG load_data stores content.
            # load_data calls extract_*_tokens which returns LIST of strings?
            # Actually FG load_data: doc_contents.append(" ".join(tokens))
            # So doc_contents are space-separated strings of tokens.
            
            tokens = set(content.split())
            # Filter to only vocab words to avoid noise
            valid_tokens = tokens.intersection(vocab_set)
            doc_token_map[doc_name] = valid_tokens

        count_pattern = 0
        count_latent = 0

        print(f"Processing {len(qa_records)} QA records for pattern mining...")

        for record in qa_records:
            # Handle both structure types (direct dict or wrapped in 'query')
            if 'query' in record:
                q_data = record['query']
            else:
                q_data = record

            question = q_data.get('question', '')
            if not question: continue
            
            # Tokenize question
            q_tokens = clean_and_tokenize(question)
            
            ref_docs = q_data.get('ref_docs', [])
            target_docs_tokens = []
            
            for ref in ref_docs:
                src = ref.get('source', '')
                if not src: continue
                # We need to match filenames. FG stores relative paths usually, or absolute.
                # FG.doc_names stores what load_data found. extract_tokens stores basename usually? 
                # Nope, usually full paths. But keyword matching often easier on basenames.
                fname = os.path.basename(src)
                
                if fname in doc_token_map:
                    target_docs_tokens.append(doc_token_map[fname])
                else:
                    # Try fuzzy match or finding the file in map keys
                    # Sometimes encoding 'file:///' messes up
                    for k, v in doc_token_map.items():
                        if fname in k or k in fname:
                            target_docs_tokens.append(v)
                            break
            
            if not target_docs_tokens:
                continue

            # Consolidate target tokens and their frequency across referenced docs for this query
            combined_target_tokens = defaultdict(int)
            for d_toks in target_docs_tokens:
                for t in d_toks:
                    combined_target_tokens[t] += 1
            
            # Analyze each Question Token
            for qt in q_tokens:
                # We want to link qt to words in the document
                
                is_oov = qt not in vocab_set
                qt_cons = self.get_consonants(qt)
                qt_len = len(qt)
                
                match_found = False
                
                # Check against all words in the target documents
                for target_token, doc_freq in combined_target_tokens.items():
                    if target_token == qt: continue
                    
                    score = 0
                    tt_len = len(target_token)
                    
                    # Compute normalized forms for variant matching
                    qt_norm = self.normalize_variant(qt)
                    t_norm = self.normalize_variant(target_token)

                    # 1. Prefix Match (qt="sec", target="security")
                    if qt_len >= 3 and target_token.startswith(qt):
                        score += feature_weights.LATENT_WEIGHTS['prefix_match']
                    
                    # 2. Reverse Prefix Match (qt="lending", target="lend")
                    elif tt_len >= 3 and qt.startswith(target_token):
                        score += 3.0

                    # 3. Normalized Variant Match (secs -> securities, construct -> construction)
                    elif len(qt_norm) >= 3 and (t_norm.startswith(qt_norm) or qt_norm.startswith(t_norm)):
                        score += 2.5

                    # 4. Consonant Match (lnd -> land, lend)
                    # Only if lengths are somewhat comparable to avoid 's' matching 'supercalif...' falsely?
                    # Consonant skeletons of 'sec' -> 'sc'. 'security' -> 'scrty'. Match? No.
                    # 'lnd' -> 'lnd'. 'land' -> 'lnd'. Match!
                    elif qt_len >= 3 and self.get_consonants(target_token) == qt_cons:
                        score += feature_weights.LATENT_WEIGHTS['consonant_match']
                        
                    if score > 0:
                        # Enhance by freq: if it appears in multiple linked docs, boost it
                        # The user says "enhanced by multiplication if in records json one questions to multiple documents"
                        # We use doc_freq * score
                        final_score = score * doc_freq
                        self.keyword_edges[qt][target_token] += final_score
                        match_found = True
                        count_pattern += 1
                
                # 3. Latent Co-occurrence (No letter Match)
                # "if there is totally no found by letter patterns, apply weights equally... enhanced by multiplication"
                if not match_found and is_oov:
                     # Distribute weight to meaningful target tokens
                     for target_token, doc_freq in combined_target_tokens.items():
                         if len(target_token) < 3: continue
                         
                         # Avoid stop words or noise if possible (vocab intersection helps)
                         # Weight: Base (small) * (doc_freq ^ 2)
                         # Squaring doc_freq gives the "multiplication" effect for agreement
                         w = 0.1 * (doc_freq ** 2)
                         self.keyword_edges[qt][target_token] += w
                         count_latent += 1

        print(f"Keyword Expansion Built: {len(self.keyword_edges)} source tokens.")
        print(f"  - Pattern matches (Prefix/Consonant): {count_pattern}")
        print(f"  - Latent associations (Co-occurrence): {count_latent}")

    def save(self, path):
        # Flatten for pickling
        data = {k: dict(v) for k, v in self.keyword_edges.items()}
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Keyword Expansion model saved to {path}")
        except Exception as e:
            print(f"Error saving Keyword Expansion model: {e}")

    def load(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.keyword_edges = defaultdict(lambda: defaultdict(float), data)
                print(f"Keyword Expansion model loaded from {path}")
            except Exception as e:
                print(f"Error loading Keyword Expansion model: {e}")
        else:
            print(f"No Keyword Expansion model found at {path}")

    def expand_query(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Returns a dictionary of {expanded_token: weight} for the given query tokens.
        """
        expansion = defaultdict(float)
        
        for q in query_tokens:
            if q in self.keyword_edges:
                edges = self.keyword_edges[q]
                
                # Filter and Top-K
                # If we have many latent matches, pick top ones
                sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
                
                # Take top 5 expansions per token
                for target, score in sorted_edges[:5]:
                    # Normalize weight
                    # If score is from pattern (3.0+), keep high.
                    # If latent (0.1...), keep low unless high agreement.
                    
                    if score >= 3.0: 
                        w = 1.2 # Strong boost
                    elif score >= 1.0: 
                        w = 0.8
                    else: 
                        w = 0.3 # Weak association
                        
                    # Accumulate (in case multiple query tokens map to same target)
                    expansion[target] += w
        
        return dict(expansion)

    def generate_expanded_ngrams(self, query_tokens: List[str], token_base_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Generates n-grams (1-5) taking into account token expansions in sequence.
        Returns {ngram_string: max_weight}.
        weight = 1.0 for original sequence, 
                 product of expansion weights for expanded sequences.
        Args:
           query_tokens: List of tokens in order.
           token_base_weights: Optional map of token -> base_weight (e.g. from IQF).
                               if provided, lattice options will be scaled by this.
        """
        import itertools
        from typing import List, Tuple
        
        # 1. Build Lattice: List of (token, weight) options for each position
        lattice: List[List[Tuple[str, float]]] = []
        
        for q in query_tokens:
            
            # Apply base weight from external logic (IQF) if present
            base_w = 1.0
            if token_base_weights and q in token_base_weights:
                base_w = token_base_weights[q]
            
            options = [(q, 1.0 * base_w)] # Original token always present, scaled
            
            if q in self.keyword_edges:
                edges = self.keyword_edges[q]
                sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
                
                # Take top 3 expansions per token for the lattice to explode too much
                for target, score in sorted_edges[:3]:
                    # Determine weight using centralized logic
                    w = feature_weights.get_expansion_weight(score)
                    
                    # Apply base scale to the EXPANSION too (so 'sec' IQF boosts 'security')
                    options.append((target, w * base_w))
            
            lattice.append(options)
            
        result_ngrams = defaultdict(float)
        
        # 2. Generate N-grams from Lattice
        L = len(lattice)
        max_n = 5
        
        for n in range(1, max_n + 1):
            for i in range(L - n + 1):
                # Window of layers: lattice[i] ... lattice[i+n-1]
                window_layers = lattice[i : i + n]
                
                # Cartesian Product
                # itertools.product(*[[(t1,w1), (t2,w2)], [(t3,w3)...]])
                for path in itertools.product(*window_layers):
                    # path is tuple of (token, weight)
                    tokens = [p[0] for p in path]
                    weights = [p[1] for p in path]
                    
                    ngram_str = " ".join(tokens)
                    
                    # Combined weight: geometric mean? product? min?
                    # Product makes sense: 0.8 * 0.8 = 0.64 confidence
                    combined_weight = 1.0
                    for w in weights:
                        combined_weight *= w
                        
                    # Keep max weight for this ngram if generated multiple ways
                    if combined_weight > result_ngrams[ngram_str]:
                        result_ngrams[ngram_str] = combined_weight
                        
        return dict(result_ngrams)
