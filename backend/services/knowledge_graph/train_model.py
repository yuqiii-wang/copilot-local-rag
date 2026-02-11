import os
import sys
import pickle
import copy

# Add current working directory to path so we can run this as a script from backend/
sys.path.append(os.getcwd())

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import csv
from sklearn.preprocessing import normalize
from services.knowledge_graph.feature_generator import FeatureGenerator
from services.knowledge_graph.graph_model import KeywordReconstructionHGNN
from services.knowledge_graph.tokenization import clean_and_tokenize
from services.knowledge_graph.keyword_expander import KeywordExpander
from services.knowledge_graph.code_indexer import (
    JavaCodeIndexer,
    CppCodeIndexer,
    BashCodeIndexer,
    SqlCodeIndexer
)

from services.knowledge_graph import feature_weights
from config import config

# Configuration
MAX_FEATURES = None # Extract every word
HIDDEN_DIM = 128
LEARNING_RATE = 0.001 # Reduced to improve stability
EPOCHS = 200 

if config.DEBUG:
    DATASET_DIR = "services/knowledge_graph/dummy_dataset"
else:
    DATASET_DIR = "services/knowledge_graph/real_dataset"

MODEL_SAVE_PATH = "services/knowledge_graph/query_model.pth"
FEATURE_GEN_PATH = "services/knowledge_graph/feature_gen.pkl"
KEYWORD_EXPANDER_PATH = "services/knowledge_graph/keyword_expander.pkl"
RESULTS_CSV_PATH = "services/knowledge_graph/hypergraph_results.csv"

def train():
    # 0. Run Indexers FIRST to extract literals
    print("Running Code Indexers to capture literals for Vocabulary...")
    
    # Map extensions to indexers
    extension_indexer_map = {
        '.java': JavaCodeIndexer,
        '.cpp': CppCodeIndexer, '.hpp': CppCodeIndexer, '.h': CppCodeIndexer, '.c': CppCodeIndexer, '.cc': CppCodeIndexer,
        '.sh': BashCodeIndexer, '.bash': BashCodeIndexer,
        '.sql': SqlCodeIndexer
    }

    active_indexer_classes = set()
    print(f"Scanning {DATASET_DIR} for code files...")
    if os.path.exists(DATASET_DIR):
        for root, _, files in os.walk(DATASET_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in extension_indexer_map:
                    active_indexer_classes.add(extension_indexer_map[ext])
    else:
        print(f"Warning: Dataset directory {DATASET_DIR} does not exist.")

    executed_indexers = []
    all_extracted_literals = []

    for IndexerClass in active_indexer_classes:
        print(f"Running {IndexerClass.__name__} on {DATASET_DIR}")
        indexer = IndexerClass(DATASET_DIR)
        indexer.index_project()
        executed_indexers.append(indexer)
        
        # Collect literals for Vocab Augmentation
        for fname, literals in indexer.file_literals.items():
             for lit in literals:
                 # Tokenize strictly for vocab
                 tokens = clean_and_tokenize(lit)
                 if tokens:
                     all_extracted_literals.extend(tokens)
                     # Add separator to prevent false n-grams between independent literals
                     all_extracted_literals.append("___SEP___")

    print(f"Collected {len(all_extracted_literals)} tokens from hardcoded strings for Vocabulary.")

    # 1. Initialize and Fit Feature Generator (Extract every word)
    print("Initializing Feature Generator...")
    fg = FeatureGenerator(max_features=MAX_FEATURES)
    fg.load_data(DATASET_DIR)
    
    fg.fit()
    fg.save(FEATURE_GEN_PATH)

    num_docs = len(fg.doc_names)
    vocab_size = len(fg.vectorizer.get_feature_names_out())
    print(f"Total documents: {num_docs}")
    print(f"Vocabulary size: {vocab_size}")

    # 2. Construct Graph
    # Nodes: Documents
    # Hyperedges: Keywords
    # Incidence Matrix (H): [Nodes, Hyperedges] -> [Docs, Keywords]
    # In PyG HypergraphConv: hyperedge_index = [Nodes, Hyperedges]
    # feature_generator produces [Docs, Keywords] sparse matrix
    
    coo = fg.features.tocoo()
    # Row: Docs (Nodes), Col: Keywords (Hyperedges)
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    hyperedge_index = torch.stack([row, col], dim=0)

    # 2b. Add Topic Hyperedges (NMF)
    total_hyperedges = vocab_size
    if hasattr(fg, 'doc_topic_matrix') and fg.doc_topic_matrix is not None:
        print("Constructing Topic Hyperedges...")
        doc_topic = fg.doc_topic_matrix
        n_topics = doc_topic.shape[1]
        
        # Link documents to their top topics (using simple threshold)
        rows, cols = np.where(doc_topic > 0.01) # Low threshold to capture weak links too
        
        # Shift topic indices by vocab_size so they form new hyperedges
        # Hyperedge IDs: 0..vocab-1 (Keywords), vocab..vocab+topics-1 (Topics)
        topic_indices = cols + vocab_size
        
        t_row = torch.from_numpy(rows.astype(np.int64))
        t_col = torch.from_numpy(topic_indices.astype(np.int64))
        
        topic_hyperedge_index = torch.stack([t_row, t_col], dim=0)
        
        # Merge
        hyperedge_index = torch.cat([hyperedge_index, topic_hyperedge_index], dim=1)
        print(f"Added {len(t_row)} topic edges.")
        
        total_hyperedges = vocab_size + n_topics

    # 3. Prepare Node Features (X) with Priors
    # Base Features: TF-IDF Matrix [Docs, Keywords]
    X_check = fg.features.toarray()
    X = torch.FloatTensor(X_check)

    # N-Gram Boosting: Scale features by n-gram length to emphasize wider coverage
    print("Boosting features based on N-Gram length...")
    vocab_list = fg.vectorizer.get_feature_names_out()
    # Boost: length^2.5 to strongly favor longest matches as requested
    ngram_weights = [feature_weights.get_ngram_boost(w) for w in vocab_list] 
    ngram_weights_tensor = torch.FloatTensor(ngram_weights)
    X = X * ngram_weights_tensor

    # Load QA Dataset first (QueryData schema)
    print("Loading QA (QueryData schema)")
    # Note: Logic for loading from a single QA file has been removed. 
    # We now strictly use the conversation records directory.
    qa_map = {}  # question -> list of dicts {fname, status, comment
    # Also ingest conversation records (if present) to augment QA priors.
    # Human messages are treated as strong signals (status='accepted'),
    # while assistant responses are included with lower weight (status='pending').
    
    # Correctly resolve path to data/records relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    conv_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'records'))
    
    try:
        if os.path.isdir(conv_dir):
            files = [f for f in os.listdir(conv_dir) if f.endswith('.json')]
            for conv_file in files:
                print(f"Loading records from {conv_file}...")
                with open(os.path.join(conv_dir, conv_file), 'r', encoding='utf-8') as cf:
                    conv_list = json.load(cf)
                for rec in conv_list:
                    q = rec.get('query', {}) or {}
                    ref_docs = q.get('ref_docs', []) or []
                    conversations = q.get('conversations', []) or []
                    # Add main question if present (handles records with empty conversations)
                    main_question = (q.get('question') or "").strip()
                    rec_status = q.get('status') or 'accepted'
                    if main_question:
                         for ref in ref_docs:
                            src = ref.get('source', '')
                            fname_key = os.path.basename(src)
                            score = ref.get('score')
                            if score is None: score = 0.0
                            qa_map.setdefault(main_question, []).append({'fname': fname_key, 'status': rec_status, 'comment': '', 'pos_mult': 1.0, 'score': score})

                    for conv in conversations:
                        human_text = (conv.get('human') or "").strip()
                        ai_text = (conv.get('ai_assistant') or "").strip()
                        for ref in ref_docs:
                            src = ref.get('source', '')
                            fname_key = os.path.basename(src)
                            score = ref.get('score')
                            if score is None:
                                score = 0.0
                                
                            # Avoid duplicates if conversation text matches the main question perfectly
                            if human_text and human_text != main_question:
                                qa_map.setdefault(human_text, []).append({'fname': fname_key, 'status': 'accepted', 'comment': '', 'pos_mult': 1.0, 'score': score})
                            if ai_text:
                                qa_map.setdefault(ai_text, []).append({'fname': fname_key, 'status': 'pending', 'comment': '', 'pos_mult': 1.0, 'score': score})
            print("Augmented QA mapping with conversation records.")
    except Exception as e:
        print(f"Error loading conversation records: {e}")

    # === KEYWORD EXPANDER BUILD ===
    print("Building Keyword Expander from QA records...")
    # Convert qa_map back to record list structure for the expander
    # Only include records from JSON/Conversations (before code literals are added)
    expander_records = []
    for q_text, targets in qa_map.items():
         expander_records.append({
             'question': q_text,
             'ref_docs': [{'source': t['fname']} for t in targets]
         })
    
    keyword_expander = KeywordExpander()
    # We pass the fit feature_generator which contains doc_names and doc_contents
    keyword_expander.build(fg, expander_records)
    keyword_expander.save(KEYWORD_EXPANDER_PATH)
    # === END KEYWORD EXPANDER ===

    # 3b. Augment QA mapping with Indexer extracted literals (as pending queries)
    print("Augmenting QA mapping with extracted literals from code indexers...")
    for indexer in executed_indexers:
        count_lits = 0
        for fname, literals in indexer.file_literals.items():
            for lit in literals:
                clean_lit = lit.strip()
                if len(clean_lit) > 2:
                    # treat literal as a short query mapping to this file
                    if clean_lit not in qa_map:
                        qa_map[clean_lit] = []
                    # avoid duplicate entries for same file
                    if not any(entry['fname'] == fname for entry in qa_map[clean_lit]):
                        qa_map[clean_lit].append({'fname': fname, 'status': 'pending', 'comment': '', 'pos_mult': 1.0})
                        count_lits += 1
        print(f"Added {count_lits} literal->file mappings from {indexer.__class__.__name__}")


    print("Injecting Priors from QA Dataset (and extracted literals)...")
    prior_weight = feature_weights.PRIOR_INJECTION_WEIGHT # Used to be hardcoded 10.0
    
    vocab_map = fg.vectorizer.vocabulary_ # Word -> Index
    
    # helper map for lookups since fg.file_map now uses full paths
    basename_to_idx = {os.path.basename(fpath): idx for fpath, idx in fg.file_map.items()}

    priors_injected_count = 0

    # Status weight multipliers
    status_weights = feature_weights.STATUS_WEIGHTS

    # Pre-calculate frequency of each query keyword in the QA dataset (weighted by status importance)
    query_keyword_counts = {}
    for query, targets in qa_map.items():
        if targets:
            max_w = max(status_weights.get(t.get('status','pending'), 1.0) for t in targets)
        else:
            max_w = 1.0
        for token in clean_and_tokenize(query):
            query_keyword_counts[token] = query_keyword_counts.get(token, 0) + max_w

    # Calculate and Save Inverse Question Frequency (IQF) stats for Service
    print("Calculating Inverse Question Frequency (IQF)...")
    qa_idf_stats = {}
    total_questions = len(qa_map)
    for token, freq in query_keyword_counts.items():
        # log(N / df). Using slightly smoothed simple version.
        # Ensure minimum frequency 1.0
        df = max(1.0, freq)
        iqf = np.log(total_questions / df + 1.0) + 1.0 # +1 to ensure base weight at least 1
        qa_idf_stats[token] = iqf
    
    qa_idf_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'question_idf.pkl')
    with open(qa_idf_path, 'wb') as f:
        pickle.dump(qa_idf_stats, f)
    print(f"Saved IQF stats for {len(qa_idf_stats)} tokens to {qa_idf_path}")

    for query, targets in qa_map.items():
        # Tokenize query to match vocab tokens using the SAME logic as ingestion
        query_tokens = clean_and_tokenize(query)
        
        # EXPAND QUERY TOKENS
        # Use keyword expander to find latent tokens AND sequence combinations
        # e.g. "order sec" -> "order security"
        tokens_to_inject = keyword_expander.generate_expanded_ngrams(query_tokens)
        
        # 1. Calculate Per-Query Total Weight for Normalization
        # This ensures that terms in a query compete for the total prior weight.
        # Unique terms (high IQF) get a larger share of the signal.
        token_stats = []
        total_query_weight = 0.0
        
        for token, modification_weight in tokens_to_inject.items():
            if token not in vocab_map: continue
            
            parts = token.split()
            if parts:
               iqfs = [qa_idf_stats.get(p, 1.0) for p in parts]
               iqf_val = sum(iqfs) / len(iqfs)
            else:
               iqf_val = 1.0
               
            ngram_len = len(parts)
            seq_boost = feature_weights.get_qa_sequence_boost(ngram_len, 1.0)
            
            # AMPLIFIED IDF: Cube the IQF to make unique tokens dominate the weight distribution
            # "sec" (IQF~2 -> 8) vs "US123" (IQF~5 -> 125). Ratio 1:15 instead of 1:2.5
            # This prevents common words in specific queries from "stealing" the prior signal.
            raw_w = (iqf_val ** 3.0) * seq_boost * modification_weight
            token_stats.append({
                'token': token,
                'raw_w': raw_w
            })
            total_query_weight += raw_w
            
        if total_query_weight < 1e-6: total_query_weight = 1.0
        
        # 2. Inject Normalized Priors
        for stat in token_stats:
            token = stat['token']
            kw_idx = vocab_map[token]
            
            # Relative Importance: How much does this token contribute to THIS query?
            # [0.0 - 1.0]
            relative_ratio = stat['raw_w'] / total_query_weight
            
            # Apply Prior Weight: The total energy injected for this query is prior_weight
            current_weight = prior_weight * relative_ratio
             
            for entry in targets:
                fname_key = entry.get('fname')
                status = entry.get('status','pending')
                pos_mult = entry.get('pos_mult', 1.0)
                
                # BUG FIX: Score Normalization using centralized weights
                score_mult = feature_weights.get_user_score_weight(entry.get('score'))
                
                weight_multiplier = status_weights.get(status, 1.0) * pos_mult
                
                if fname_key in basename_to_idx:
                    doc_idx = basename_to_idx[fname_key]
                    # Boost Feature: Keyword Presence in Doc
                    X[doc_idx, kw_idx] += current_weight * weight_multiplier * score_mult
                    priors_injected_count += 1
                    
                    # Also inject comment tokens if present (strong boost, includes position)
                    # Note: Since we loop through query tokens, summing relative_ratio = 1.0
                    # Effectively we inject the comment exactly 'Once' (weighted by prior_weight sum)
                    comment = entry.get('comment','') or ''
                    if comment and fname_key in basename_to_idx:
                        for ctoken in clean_and_tokenize(comment):
                            if ctoken in vocab_map:
                                cidx = vocab_map[ctoken]
                                comment_boost = feature_weights.QA_COMPONENT_BOOSTS['comment']
                                # Use current_weight ensures comment injection is also "distributed" logic 
                                # essentially scaling comment importance by query importance is fine
                                X[basename_to_idx[fname_key], cidx] += current_weight * weight_multiplier * comment_boost * score_mult
                                priors_injected_count += 1

                    # Inject explicit keywords with a large boost
                    kw_list = entry.get('keywords', []) or []
                    keyword_boost = feature_weights.QA_COMPONENT_BOOSTS['keywords']
                    if kw_list and fname_key in basename_to_idx:
                        for kw in kw_list:
                            # allow multi-word keywords as-is
                            if kw in vocab_map:
                                kidx = vocab_map[kw]
                                X[basename_to_idx[fname_key], kidx] += keyword_boost * current_weight * weight_multiplier * score_mult
                                priors_injected_count += 1
                            else:
                                # Try tokenized variants
                                for kw_token in clean_and_tokenize(kw):
                                    if kw_token in vocab_map:
                                        tidx = vocab_map[kw_token]
                                        X[basename_to_idx[fname_key], tidx] += (keyword_boost/5.0) * current_weight * weight_multiplier * score_mult
                                        priors_injected_count += 1

    print(f"Injected {priors_injected_count} QA priors into feature matrix.")

    # 3b. Inject Code Priors (Strings and Dependencies) - SECOND PASS for Graph Propagation
    print("Injecting Code Graph Priors (Dependency Propagation)...")
    code_prior_count = 0
    
    for indexer in executed_indexers:
        # We reuse the already executed indexers
        priors = indexer.get_code_priors(
            hard_weight=feature_weights.CODE_GRAPH_WEIGHTS['hard_link'], 
            decay=feature_weights.CODE_GRAPH_WEIGHTS['decay_factor']
        )
        
        for token, file_weights in priors.items():
            if token in vocab_map:
                kw_idx = vocab_map[token]
                for fname, weight in file_weights.items():
                    fname_key = os.path.basename(fname)
                    
                    if fname_key in basename_to_idx: 
                        doc_idx = basename_to_idx[fname_key]
                        X[doc_idx, kw_idx] += weight
                        code_prior_count += 1
                     
    print(f"Injected {code_prior_count} Code graph priors into feature matrix.")

    # 3c. Inject Pattern Metadata (Structural Priors)
    if hasattr(fg, 'doc_patterns'):
        print("Injecting Metadata Patterns into feature matrix (Structural Boosting)...")
        pattern_inject_count = 0
        vocab_map = fg.vectorizer.vocabulary_
        
        for i, patterns in enumerate(fg.doc_patterns):
            for pat in patterns:
                if pat in vocab_map:
                    idx = vocab_map[pat]
                    # Specific Boost for Structural Patterns
                    # This ensures documents with similar structures (e.g. both have ISINs) are linked
                    X[i, idx] += feature_weights.PATTERN_WEIGHT # Significant boost
                    pattern_inject_count += 1
        print(f"Injected {pattern_inject_count} pattern signals.")

    # 3c-2. Inject Filename Prefix/Partial Matching
    print("Injecting Filename Prefix Matches...")
    filename_match_count = 0
    
    fn_boost = feature_weights.FILENAME_WEIGHT
    
    # Pre-compute prefixes for speed
    prefix_to_docs = {}
    for fname, doc_idx in basename_to_idx.items():
        clean_name = fname.lower().replace('.', ' ').replace('_', ' ').replace('-', ' ')
        parts = clean_name.split()
        for p in parts:
             if len(p) < 3: continue
             for l in range(3, len(p) + 1):
                 prefix = p[:l]
                 if prefix not in prefix_to_docs:
                     prefix_to_docs[prefix] = []
                 if doc_idx not in prefix_to_docs[prefix]:
                     prefix_to_docs[prefix].append(doc_idx)
    
    for token, kw_idx in vocab_map.items():
        t_low = token.lower()
        if t_low in prefix_to_docs:
             docs = prefix_to_docs[t_low]
             for d_idx in docs:
                 X[d_idx, kw_idx] += fn_boost
                 filename_match_count += 1
                 
    print(f"Injected {filename_match_count} filename prefix matches.")

    # 3d. Augment X with Topic Weights (if available) - for Reconstruction of Topics
    if hasattr(fg, 'doc_topic_matrix') and fg.doc_topic_matrix is not None:
        print(f"Augmenting Node Features with {fg.doc_topic_matrix.shape[1]} NMF topics...")
        topic_features = torch.FloatTensor(fg.doc_topic_matrix)
        # Scale topics to match TF-IDF scale (approx) before log
        topic_features = topic_features * feature_weights.INFERENCE_WEIGHTS['topic_feature_scale']
        X = torch.cat([X, topic_features], dim=1)
    
    # Log-transform features to handle skewness and large priors (100.0 -> ~4.6)
    print("Log-transforming features for better numerical stability...")
    X = torch.log1p(X)

    # Save the fully engineered features including priors for inference
    feat_save_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'node_features.pt')
    print(f"Saving enhanced node features to {feat_save_path}")
    torch.save(X, feat_save_path)

    # 4. Initialize Model
    # Input: [Docs, Features] (Content + Topics)
    # Output: [Docs, Features] (Reconstructed/Predicted Links)
    
    input_dim = X.shape[1]
    print(f"Model Input Dimension: {input_dim}")
    
    # KeywordReconstructionHGNN args: in_channels, hidden_channels, num_keywords
    # Since we are predicting keywords for doc nodes, num_keywords is the output dim
    model = KeywordReconstructionHGNN(
        in_channels=input_dim,
        hidden_channels=HIDDEN_DIM,
        num_keywords=input_dim
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # AdamW for better regularization
    
    # Scheduler: CosineAnnealingLR for smoother convergence over fixed epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # criterion = nn.SmoothL1Loss(beta=1.0) 

    # 5. Training Loop
    print(f"Starting training (Reconstruction Task) for {EPOCHS} epochs...")
    model.train()
    
    # Pre-calculate Loss Weights based on Target Importance (X)
    # This ensures that stronger signals (high Inverse Frequency, N-Grams, Priors) 
    # have higher probability of being learned and locating the document.
    # Weight = 1.0 + (X * Multiplier). 
    # Example: Weak signal (0.1) -> Weight ~3. Strong signal (4.0) -> Weight ~81.
    loss_multiplier = feature_weights.TRAINING_WEIGHTS['non_zero_loss_multiplier']
    base_loss_weights = 1.0 + (X * loss_multiplier)
    
    miss_penalty = feature_weights.TRAINING_WEIGHTS.get('missed_doc_penalty', 50.0)

    loss_fn = nn.MSELoss(reduction='none')

    start_check_epoch = int(EPOCHS * 0.75)
    best_loss = float('inf')
    best_model_state = None
    print(f"Will track lowest loss model starting from epoch {start_check_epoch} (Last 25% of training)...")

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X, hyperedge_index)
        
        # Dynamic Loss Weighting:
        # Check if we are totally missing a signal (Target > 0 but Output ~ 0)
        # We define "totally missed" as output < 0.2 * Target (when Target > 0)
        # This prevents the model from ignoring difficult documents.
        with torch.no_grad():
             # Create binary mask for missed signals (False Negatives)
             missed_mask = (X > 0.0) & (outputs < (X * 0.2))
             # Add penalty weight to the base weights for these specific elements
             current_weights = base_loss_weights + (missed_mask.float() * miss_penalty)

        # Weighted Reconstruction Loss
        element_loss = loss_fn(outputs, X)
        loss = (element_loss * current_weights).mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track best model in last 25%
        if epoch >= start_check_epoch:
             current_loss = loss.item()
             if current_loss < best_loss:
                 best_loss = current_loss
                 best_model_state = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

    # 6. Save Model
    if best_model_state is not None:
        print(f"Restoring best model state from last 25% of epochs (Loss: {best_loss:.6f})...")
        model.load_state_dict(best_model_state)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 7. Generate Output Mapping (CSV)
    print("Generating knowledge graph mapping...")
    model.eval()
    with torch.no_grad():
        gnn_scores = model(X, hyperedge_index)
        # Add Input Features to Output Scores (Residual/Skip Connection for Inference)
        # This boosts the score of documents that actually contain the term (from TF-IDF or Priors)
        # helping distinguish them from neighbors that only get "leaked" signal.
        final_scores = (gnn_scores + X).numpy() # [Docs, Keywords]
        
    # We want "Keyword -> Related Documents".
    
    vocab_list = fg.vectorizer.get_feature_names_out()
    
    with open(RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Keyword', 'Document ID', 'Score', 'Document Type'])
        
        # Determine Doc Types for Visualization
        doc_types = {}
        for name in fg.doc_names:
            if name.endswith(('.java', '.cpp', '.cc', '.h', '.hpp', '.c', '.sh', '.bash', '.sql')): 
                d_type = 'code'
            elif 'jira' in name.lower() or 'bug' in name.lower() or 'trd' in name.lower():
                d_type = 'jira'
            else:
                d_type = 'confluence'
            doc_types[name] = d_type

        count = 0
        threshold = feature_weights.TRAINING_WEIGHTS['inference_threshold'] # Loosened threshold to allow more entities (TF-IDF often < 1.0)
        
        for k_idx, keyword in enumerate(vocab_list):
            # Get docs for this keyword
            doc_scores = final_scores[:, k_idx]
            
            # Find docs with score > threshold
            top_indices = np.where(doc_scores > threshold)[0]
            
            for d_idx in top_indices:
                score = doc_scores[d_idx]
                doc_name = fg.reverse_file_map[d_idx]
                d_type = doc_types[doc_name]
                
                writer.writerow([keyword, doc_name, f"{score:.4f}", d_type])
                count += 1
        
        # Add Co-occurrence Keyword-Keyword Edges (context-based)
        print("Calculating Keyword Co-occurrence from Document Context...")
        # Normalize columns to unit length -> Dot product becomes Cosine Similarity
        # This ensures the score is [0,1] and independent of document length/count
        X_norm = normalize(fg.features, norm='l2', axis=0)

        # Calculate Co-occurrence: (Vocab x Docs) @ (Docs x Vocab) -> (Vocab x Vocab)
        # Element (i, j) is Cosine Similarity between keyword i and keyword j
        word_co_occurrence = X_norm.T @ X_norm
        word_co_occurrence.setdiag(0)
        word_co_occurrence.eliminate_zeros()
        
        # Hard limit on edges to prevent CSV explosion
        # (Small datasets with few docs cause many 1.0 co-occurrences)
        MAX_COOCCURRENCE_EDGES = 1000
        
        coo_kw = word_co_occurrence.tocoo()
        row, col, data = coo_kw.row, coo_kw.col, coo_kw.data
        
        if len(data) > MAX_COOCCURRENCE_EDGES:
            print(f"Found {len(data)} co-occurrence edges. Keeping top {MAX_COOCCURRENCE_EDGES}...")
            # argsort is ascending, so take last N
            top_indices = np.argsort(data)[-MAX_COOCCURRENCE_EDGES:]
            # Reverse for descending order in CSV (optional)
            top_indices = top_indices[::-1]
            
            row = row[top_indices]
            col = col[top_indices]
            data = data[top_indices]

        vocab_names = fg.vectorizer.get_feature_names_out()
        co_occur_count = 0
        
        for i, j, v in zip(row, col, data):
             src_kw = vocab_names[i]
             tgt_kw = vocab_names[j]
             writer.writerow([src_kw, tgt_kw, f"{v:.4f}", 'keyword_cooccurrence'])
             co_occur_count += 1
             count += 1
        
        print(f"Appended {co_occur_count} context-based keyword co-occurrence edges.")

        # Add Latent Keyword-Keyword Edges
        print("Appending Keyword-Keyword latent edges to CSV...")
        if keyword_expander and hasattr(keyword_expander, 'keyword_edges'):
            latent_count = 0
            for src, tgts in keyword_expander.keyword_edges.items():
                for tgt, sc in tgts.items():
                    if sc > 0.5:
                        # Format: Keyword=src, Document ID=tgt (pseudo-doc), Type=keyword
                        writer.writerow([src, tgt, f"{sc:.4f}", 'keyword'])
                        latent_count += 1
                        count += 1
            print(f"Appended {latent_count} latent keyword edges.")
                
    print(f"Saved {count} connections to {RESULTS_CSV_PATH}")

if __name__ == "__main__":
    train()
