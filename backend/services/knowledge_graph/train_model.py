import os
import sys

# Add current working directory to path so we can run this as a script from backend/
sys.path.append(os.getcwd())

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import csv
from services.knowledge_graph.feature_generator import FeatureGenerator
from services.knowledge_graph.graph_model import KeywordReconstructionHGNN
from services.knowledge_graph.tokenization import clean_and_tokenize
from services.knowledge_graph.code_indexer import (
    JavaCodeIndexer,
    CppCodeIndexer,
    BashCodeIndexer,
    SqlCodeIndexer
)

# Configuration
MAX_FEATURES = None # Extract every word
HIDDEN_DIM = 128
LEARNING_RATE = 0.001 # Reduced to improve stability
EPOCHS = 500 
DATASET_DIR = "services/knowledge_graph/dummy_dataset"
QA_FILE = os.path.join(DATASET_DIR, "qa_dataset.json")
MODEL_SAVE_PATH = "services/knowledge_graph/query_model.pth"
FEATURE_GEN_PATH = "services/knowledge_graph/feature_gen.pkl"
RESULTS_CSV_PATH = "services/knowledge_graph/hypergraph_results.csv"

def generate_ngrams(tokens, n_range=(1, 5)):
    """Generate n-grams from a list of tokens."""
    ngrams = []
    for n in range(n_range[0], n_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            ngrams.append(ngram)
    return ngrams

def train():
    # 0. Run Indexers FIRST to extract literals
    print("Running Code Indexers to capture literals for Vocabulary...")
    
    indexer_configs = [
        (JavaCodeIndexer, os.path.join("services", "knowledge_graph", "dummy_dataset", "dummy_trading_sys_java")),
        (CppCodeIndexer, os.path.join("services", "knowledge_graph", "dummy_dataset", "dummy_traing_sys_cpp")),
        (BashCodeIndexer, os.path.join("services", "knowledge_graph", "dummy_dataset")),
        (SqlCodeIndexer, os.path.join("services", "knowledge_graph", "dummy_dataset", "dummy_trading_system_sql"))
    ]

    executed_indexers = []
    all_extracted_literals = []

    for IndexerClass, root_path in indexer_configs:
        if not os.path.exists(root_path):
            print(f"Skipping {IndexerClass.__name__}: path {root_path} not found")
            continue
            
        print(f"Running {IndexerClass.__name__} on {root_path}")
        indexer = IndexerClass(root_path)
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

    # 3. Prepare Node Features (X) with Priors
    # Base Features: TF-IDF Matrix [Docs, Keywords]
    X_check = fg.features.toarray()
    X = torch.FloatTensor(X_check)

    # N-Gram Boosting: Scale features by n-gram length to emphasize wider coverage
    print("Boosting features based on N-Gram length...")
    vocab_list = fg.vectorizer.get_feature_names_out()
    # Boost: length^2.5 to strongly favor longest matches as requested
    ngram_weights = [len(w.split()) ** 2.5 for w in vocab_list] 
    ngram_weights_tensor = torch.FloatTensor(ngram_weights)
    X = X * ngram_weights_tensor

    # Load QA Dataset first (QueryData schema)
    print("Loading QA Dataset (QueryData schema)...")
    qa_map = {}  # question -> list of dicts {fname, status, comment}
    try:
        with open(QA_FILE, 'r') as f:
            qa_list = json.load(f)
        for rec in qa_list:
            q = rec.get('query', {}) or {}
            question = (q.get('question') or "").strip()
            status = (q.get('status') or "pending").lower()
            ref_docs = q.get('ref_docs', [])
            for idx, ref in enumerate(ref_docs):
                # Position multiplier: first doc strongest, then step down; floor at 0.2
                pos_mult = max(0.2, 1.0 - idx * 0.2)
                src = ref.get('source', '')
                fname = os.path.basename(src)
                comment = ref.get('comment', '') or ''
                keywords = ref.get('keywords', []) or []
                if not question:
                    continue
                qa_map.setdefault(question, []).append({
                    'fname': fname,
                    'status': status,
                    'comment': comment,
                    'keywords': keywords,
                    'pos_mult': pos_mult
                })
        print(f"Loaded {len(qa_map)} QA queries from {QA_FILE}.")
    except Exception as e:
        print(f"Error loading QA dataset: {e}")
        qa_map = {}

    # Also ingest conversation records (if present) to augment QA priors.
    # Human messages are treated as strong signals (status='accepted'),
    # while assistant responses are included with lower weight (status='pending').
    conv_dir = os.path.join(os.getcwd(), 'backend', 'data', 'records')
    try:
        if os.path.isdir(conv_dir):
            files = [f for f in os.listdir(conv_dir) if f.endswith('.json')]
            for conv_file in files:
                with open(os.path.join(conv_dir, conv_file), 'r', encoding='utf-8') as cf:
                    conv_list = json.load(cf)
                for rec in conv_list:
                    q = rec.get('query', {}) or {}
                    ref_docs = q.get('ref_docs', []) or []
                    conversations = q.get('conversations', []) or []
                    for conv in conversations:
                        human_text = (conv.get('human') or "").strip()
                        ai_text = (conv.get('ai_assistant') or "").strip()
                        for ref in ref_docs:
                            src = ref.get('source', '')
                            fname_key = os.path.basename(src)
                            if human_text:
                                qa_map.setdefault(human_text, []).append({'fname': fname_key, 'status': 'accepted', 'comment': '', 'pos_mult': 1.0})
                            if ai_text:
                                qa_map.setdefault(ai_text, []).append({'fname': fname_key, 'status': 'pending', 'comment': '', 'pos_mult': 1.0})
            print("Augmented QA mapping with conversation records.")
    except Exception as e:
        print(f"Error loading conversation records: {e}")

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
    prior_weight = 1.0 # Reduced from 100.0 to 1.0 to match TF-IDF scale
    
    vocab_map = fg.vectorizer.vocabulary_ # Word -> Index
    
    # helper map for lookups since fg.file_map now uses full paths
    basename_to_idx = {os.path.basename(fpath): idx for fpath, idx in fg.file_map.items()}

    priors_injected_count = 0

    # Status weight multipliers
    status_weights = {'accepted': 10.0, 'added_confluence': 5.0, 'pending': 1.0, 'rejected': 0.2}

    # Pre-calculate frequency of each query keyword in the QA dataset (weighted by status importance)
    query_keyword_counts = {}
    for query, targets in qa_map.items():
        if targets:
            max_w = max(status_weights.get(t.get('status','pending'), 1.0) for t in targets)
        else:
            max_w = 1.0
        for token in clean_and_tokenize(query):
            query_keyword_counts[token] = query_keyword_counts.get(token, 0) + max_w

    for query, targets in qa_map.items():
        # Tokenize query to match vocab tokens using the SAME logic as ingestion
        query_tokens = clean_and_tokenize(query)
        
        # PRIOR INJECTION UPDATE: Use N-Grams to match sequence order
        query_ngrams = generate_ngrams(query_tokens, n_range=(1, 7))
        
        for token in query_ngrams:
            if token in vocab_map:
                kw_idx = vocab_map[token]
                
                # Dynamic weighting based on Inverse Frequency in QA Dataset
                freq = query_keyword_counts.get(token, 1) 
                
                # Boost based on sequence length (the "sql"/"seq" of keywords)
                ngram_len = len(token.split())
                
                # Power boost for sequences
                current_weight = prior_weight * (5.0 ** ngram_len)
                
                for entry in targets:
                    fname_key = entry.get('fname')
                    status = entry.get('status','pending')
                    pos_mult = entry.get('pos_mult', 1.0)
                    weight_multiplier = status_weights.get(status, 1.0) * pos_mult
                    if fname_key in basename_to_idx:
                        doc_idx = basename_to_idx[fname_key]
                        # Boost Feature: Keyword Presence in Doc (status-weighted + position)
                        X[doc_idx, kw_idx] += current_weight * weight_multiplier
                        priors_injected_count += 1

                    # Also inject comment tokens if present (strong boost, includes position)
                    comment = entry.get('comment','') or ''
                    if comment and fname_key in basename_to_idx:
                        for ctoken in clean_and_tokenize(comment):
                            if ctoken in vocab_map:
                                cidx = vocab_map[ctoken]
                                comment_boost = 5.0
                                X[basename_to_idx[fname_key], cidx] += current_weight * weight_multiplier * comment_boost
                                priors_injected_count += 1

                    # Inject explicit keywords with a large boost
                    kw_list = entry.get('keywords', []) or []
                    keyword_boost = 50.0
                    if kw_list and fname_key in basename_to_idx:
                        for kw in kw_list:
                            # allow multi-word keywords as-is
                            if kw in vocab_map:
                                kidx = vocab_map[kw]
                                X[basename_to_idx[fname_key], kidx] += keyword_boost * weight_multiplier
                                priors_injected_count += 1
                            else:
                                # Try tokenized variants
                                for token in clean_and_tokenize(kw):
                                    if token in vocab_map:
                                        tidx = vocab_map[token]
                                        X[basename_to_idx[fname_key], tidx] += (keyword_boost/5.0) * weight_multiplier
                                        priors_injected_count += 1

    print(f"Injected {priors_injected_count} QA priors into feature matrix.")

    # 3b. Inject Code Priors (Strings and Dependencies) - SECOND PASS for Graph Propagation
    print("Injecting Code Graph Priors (Dependency Propagation)...")
    code_prior_count = 0
    
    for indexer in executed_indexers:
        # We reuse the already executed indexers
        priors = indexer.get_code_priors(hard_weight=2.0, decay=0.5)
        
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
                    X[i, idx] += 5.0 # Significant boost
                    pattern_inject_count += 1
        print(f"Injected {pattern_inject_count} pattern signals.")

    # Log-transform features to handle skewness and large priors (100.0 -> ~4.6)
    print("Log-transforming features for better numerical stability...")
    X = torch.log1p(X)

    # 4. Initialize Model
    # Input: [Docs, Keywords]
    # Output: [Docs, Keywords] (Reconstructed/Predicted Links)
    
    print(f"Model Input Dimension: {vocab_size}")
    
    # KeywordReconstructionHGNN args: in_channels, hidden_channels, num_keywords
    # Since we are predicting keywords for doc nodes, num_keywords is the output dim
    model = KeywordReconstructionHGNN(
        in_channels=vocab_size,
        hidden_channels=HIDDEN_DIM,
        num_keywords=vocab_size 
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # AdamW for better regularization
    
    # Scheduler: CosineAnnealingLR for smoother convergence over fixed epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # criterion = nn.SmoothL1Loss(beta=1.0) 

    # 5. Training Loop
    print(f"Starting training (Reconstruction Task) for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X, hyperedge_index)
        
        # Weighted Reconstruction Loss
        # We want to penalize missing non-zero values (the signal) much more than getting 0s wrong.
        # Since the matrix is highly sparse (~99% zeros), standard loss is dominated by 0->0 predictions.
        
        loss_fn = nn.MSELoss(reduction='none')
        element_loss = loss_fn(outputs, X)
        
        # Create a weight matrix: Boost importance of non-zero targets
        # Non-zero entries get 20x weight
        weights = torch.ones_like(X)
        weights[X > 0] = 20.0 
        
        loss = (element_loss * weights).mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

    # 6. Save Model
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
        threshold = 0.5 # Loosened threshold to allow more entities (TF-IDF often < 1.0)
        
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
                
    print(f"Saved {count} connections to {RESULTS_CSV_PATH}")

if __name__ == "__main__":
    train()
