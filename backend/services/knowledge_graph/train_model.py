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
from services.knowledge_graph.java_code_indexer import JavaCodeIndexer

# Configuration
MAX_FEATURES = None # Extract every word
HIDDEN_DIM = 128
LEARNING_RATE = 0.01
EPOCHS = 300 
DATASET_DIR = "services/knowledge_graph/dummy_dataset"
QA_FILE = os.path.join(DATASET_DIR, "qa_dataset.json")
MODEL_SAVE_PATH = "services/knowledge_graph/query_model.pth"
FEATURE_GEN_PATH = "services/knowledge_graph/feature_gen.pkl"
RESULTS_CSV_PATH = "services/knowledge_graph/hypergraph_results.csv"

def train():
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
    
    print("Injecting Priors from QA Dataset...")
    with open(QA_FILE, 'r') as f:
        qa_data = json.load(f)
    
    prior_weight = 100.0 # "Very high weight" as requested
    
    vocab_map = fg.vectorizer.vocabulary_ # Word -> Index
    
    priors_injected_count = 0
    for query, target_files in qa_data.items():
        # Tokenize query to match vocab tokens using the SAME logic as ingestion
        query_tokens = clean_and_tokenize(query)
        
        for token in query_tokens:
            if token in vocab_map:
                kw_idx = vocab_map[token]
                for fname in target_files:
                    # Fix: Dataset paths vs FeatureGenerator keys match
                    # FeatureGenerator stores just the filename (basename)
                    fname_key = os.path.basename(fname)
                    
                    if fname_key in fg.file_map:
                        doc_idx = fg.file_map[fname_key]
                        # Boost Feature: Keyword Presence in Doc
                        X[doc_idx, kw_idx] += prior_weight
                        priors_injected_count += 1
                        
    print(f"Injected {priors_injected_count} QA priors into feature matrix.")

    # 3b. Inject Code Priors (Strings and Dependencies)
    print("Injecting Code Priors from Java analysis...")
    # Assume source code is in 'services/knowledge_graph/dummy_dataset/dummy_trading_sys_java/src'
    # In a real app, this path would be dynamic
    code_root = os.path.join("services", "knowledge_graph", "dummy_dataset", "dummy_trading_sys_java", "src")
    indexer = JavaCodeIndexer(code_root)
    indexer.index_project()
    code_priors = indexer.get_code_priors(hard_weight=50.0, decay=0.5)
    
    code_prior_count = 0
    for token, file_weights in code_priors.items():
        if token in vocab_map:
            kw_idx = vocab_map[token]
            for fname, weight in file_weights.items():
                 if fname in fg.file_map: # Check if file exists in our feature set
                     doc_idx = fg.file_map[fname]
                     X[doc_idx, kw_idx] += weight
                     code_prior_count += 1
                     
    print(f"Injected {code_prior_count} Code priors into feature matrix.")

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
    
    # Scheduler: Cosine Annealing for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-4)
    
    criterion = nn.SmoothL1Loss(beta=1.0) # SmoothL1Loss is less sensitive to outliers (like our heavy priors) than MSE

    # 5. Training Loop
    print(f"Starting training (Reconstruction Task) for {EPOCHS} epochs...")
    model.train()
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # Forward
        outputs = model(X, hyperedge_index)
        
        # Loss
        loss = criterion(outputs, X)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    # 6. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 7. Generate Output Mapping (CSV)
    print("Generating knowledge graph mapping...")
    model.eval()
    with torch.no_grad():
        final_scores = model(X, hyperedge_index).numpy() # [Docs, Keywords]
        
    # We want "Keyword -> Related Documents".
    
    vocab_list = fg.vectorizer.get_feature_names_out()
    
    with open(RESULTS_CSV_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Keyword', 'Document ID', 'Score', 'Document Type'])
        
        # Determine Doc Types for Visualization
        doc_types = {}
        for name in fg.doc_names:
            if name.endswith('.java'): 
                d_type = 'code'
            elif 'jira' in name.lower() or 'bug' in name.lower() or 'trd' in name.lower():
                d_type = 'jira'
            else:
                d_type = 'confluence'
            doc_types[name] = d_type

        count = 0
        threshold = 0.1 # Loosened threshold to allow more entities (TF-IDF often < 1.0)
        
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
