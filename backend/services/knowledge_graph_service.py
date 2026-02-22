import torch
import sys
import os
from pathlib import Path
from typing import List, Dict
from services.knowledge_graph.graph_model import KeywordReconstructionHGNN
from services.knowledge_graph.feature_generator import generate_features
from services.knowledge_graph.tokenization import clean_and_tokenize
from services.knowledge_graph.tokenization_utils import generate_ngrams
from services.knowledge_graph.keyword_expander import KeywordExpander
from services.knowledge_graph import feature_weights
from config import config
import numpy as np

# Add knowledge graph directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
kg_dir = os.path.join(current_dir, "knowledge_graph")

class KnowledgeGraphService:
    def __init__(self):
        self.model = None
        self.data = None
        if config.DEBUG:
            self.dataset_path = os.path.join(kg_dir, "dummy_dataset")
        else:
            self.dataset_path = os.path.join(kg_dir, "real_dataset")
        # Use valid trained model path
        self.model_path = os.path.join(kg_dir, "query_model.pth")
        self.expander_path = os.path.join(kg_dir, "keyword_expander.pkl")
        self.unique_ngrams_bridge_path = os.path.join(kg_dir, "unique_ngrams_bridge.pkl")
        self.expander = None
        self.unique_ngrams_bridge = []
        # Initialization is now async via load_async()
    
    async def load_async(self):
        """Asynchronous initialization wrapper"""
        if self.model is not None:
             return
        
        print("Scheduling Knowledge Graph initialization...")
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._initialize)

    def reload(self):
        """Reloads the model and data from disk. Useful after training."""
        print("Reloading Knowledge Graph model from disk...")
        self._initialize()

    def _initialize(self):
        print("Initializing Knowledge Graph Service...")
        
        fg_path = os.path.join(kg_dir, "feature_gen.pkl")
        
        # Check if model or feature generator exists, if not train
        if not os.path.exists(fg_path) or not os.path.exists(self.model_path):
            print("Model or feature generator missing. Triggering training...")
            try:
                from services.knowledge_graph.train_model import train
                train()
                print("Training completed.")
            except Exception as e:
                print(f"Error running training: {e}")
        
        try:
            # We must use the same generator to ensure 'vocab' indices align with model weights
            from services.knowledge_graph.feature_generator import FeatureGenerator
            fg = FeatureGenerator()
            fg.load(fg_path)
            # Reconstruct the 'data' dict expected by the service
            coo = fg.features.tocoo()
            row = torch.from_numpy(coo.row.astype('int64'))
            col = torch.from_numpy(coo.col.astype('int64'))
            hyperedge_index = torch.stack([row, col], dim=0)
            
            # Replicate Training Preprocessing: N-Gram Boosting
            # Must match train_model.py logic (using centralized feature_weights)
            
            # Try to load pre-calculated enhanced features (with priors) from training
            features_path = os.path.join(kg_dir, "node_features.pt")
            if os.path.exists(features_path):
                print(f"Loading enhanced node features from {features_path}...")
                features_tensor = torch.load(features_path)
                # Ensure it matches current dimensions (safety check)
                if features_tensor.shape[0] != fg.features.shape[0]:
                     print("Warning: Loaded features doc count mismatch. Falling back to reconstruction.")
                     use_cached_features = False
                else:
                     use_cached_features = True
            else:
                 use_cached_features = False

            if not use_cached_features:
                print("Enhanced features not found or invalid. Falling back to reconstruction (Missing Priors!).")
                features_tensor = torch.from_numpy(fg.features.toarray()).float()
                
                vocab_list = fg.vectorizer.get_feature_names_out()
                ngram_weights = [feature_weights.get_ngram_boost(w) for w in vocab_list] 
                ngram_weights_tensor = torch.FloatTensor(ngram_weights)
                
                features_tensor = features_tensor * ngram_weights_tensor
                
                # Augment with Topic Features if available (must match train_model.py)
                if hasattr(fg, 'doc_topic_matrix') and fg.doc_topic_matrix is not None:
                    topic_features = torch.FloatTensor(fg.doc_topic_matrix)
                    topic_features = topic_features * feature_weights.INFERENCE_WEIGHTS['topic_feature_scale'] # Match training scale
                    features_tensor = torch.cat([features_tensor, topic_features], dim=1)
                    print(f"Augmented features with {fg.doc_topic_matrix.shape[1]} topics.")

                # Log transform features to match training distribution
                features_tensor = torch.log1p(features_tensor)

            features_array = features_tensor.cpu().numpy()

            self.data = {
                'vocab': fg.vectorizer.get_feature_names_out(),
                'node_features': features_array, # Features for nodes (docs)
                'hyperedge_index': hyperedge_index,
                'docs': [{'id': name, 'path': name, 'raw_content': "...", 'type': 'file'} for name in fg.doc_names]
            }
            # Note: raw_content is missing in pickle, simplified for this patch
            
            self.vocab = self.data['vocab']
            self.num_features = self.data['node_features'].shape[1]
            # Crucial: The model output dimension (num_keywords) covers {Vocab + Topics}
            # So we set it to the full feature dimension
            self.num_keywords = self.num_features 
            
            # Load Global IDF stats from Vectorizer if available
            # This allows us to boost unique query terms (high IDF) during retrieval
            if hasattr(fg.vectorizer, 'idf_'):
                 self.idf = fg.vectorizer.idf_
                 print(f"Loaded IDF stats for {len(self.idf)} terms.")
            else:
                 self.idf = None
                 print("IDF stats not available in FeatureGenerator.")

            print(f"Loaded features from {fg_path}. Input dim: {self.num_features}")
            
            # Load Inverse Question Frequency (IQF) stats if available
            iqf_path = os.path.join(kg_dir, "question_idf.pkl")
            if os.path.exists(iqf_path):
                try:
                    import pickle
                    with open(iqf_path, 'rb') as f:
                        self.question_idf = pickle.load(f)
                    print(f"Loaded IQF stats for {len(self.question_idf)} question tokens.")
                except Exception as e:
                    print(f"Error loading IQF stats: {e}")
                    self.question_idf = {}
            else:
                self.question_idf = {}

            # Load Keyword Expander
            self.expander = KeywordExpander()
            self.expander.load(self.expander_path)
            
            # Load Unique Ngrams Bridge
            if os.path.exists(self.unique_ngrams_bridge_path):
                try:
                    with open(self.unique_ngrams_bridge_path, 'rb') as f:
                        self.unique_ngrams_bridge = pickle.load(f)
                    print(f"Loaded unique ngrams bridge with {len(self.unique_ngrams_bridge)} entries")
                except Exception as e:
                    print(f"Error loading unique ngrams bridge: {e}")
                    self.unique_ngrams_bridge = []
            else:
                self.unique_ngrams_bridge = []

        except Exception as e:
            print(f"Failed to load trained feature generator: {e}. Falling back to regeneration (Model mismatch risk!).")
            self.data = generate_features(self.dataset_path, max_features=None)
            if not self.data: return
            self.vocab = self.data['vocab']
            self.num_features = self.data['node_features'].shape[1]
            self.num_keywords = len(self.vocab)
        
        # 2. Load Model Architecture
        # Hidden dim must match training (128)
        self.model = KeywordReconstructionHGNN(
            in_channels=self.num_features, 
            hidden_channels=128, 
            num_keywords=self.num_keywords
        )
        
        # 3. Load State Dict
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        else:
            print("Model weights file not found. Service will run with untrained weights (useless).")

    def query_graph(self, user_query: str, skip: int = 0, limit: int = 5) -> List[Dict]:
        """
        1. Tokenize query
        2. Find matching selective keywords from vocab
        3. Run model to get P(Keyword | Document)
        4. Aggregate scores for matched keywords -> Documents
        """
        if not self.model or not self.data:
            return []

        # 1. Tokenize
        query_tokens = clean_and_tokenize(user_query)
        print(f"Query tokens: {query_tokens}")
        
        # KEYWORD EXPANSION + N-GRAMS
        # Generate n-grams including latents (e.g. "order sec" -> "order security")
        # We pass None for token_base_weights to get pure expansion/structural weights.
        # We will apply IQF weighting specifically in the matching loop to match Training logic.
        if self.expander:
             query_ngrams = self.expander.generate_expanded_ngrams(query_tokens, token_base_weights=None)
        else:
             # Fallback
             query_ngrams = {t: 1.0 for t in query_tokens}
             for n in range(2, 6):
                for i in range(len(query_tokens) - n + 1):
                    ngram = " ".join(query_tokens[i:i+n])
                    query_ngrams[ngram] = 1.0

        # 2. Match with Vocab
        matched_indices = []
        matched_words = []
        matched_weights = []

        for i, word in enumerate(self.vocab):
            # Check if vocab word is in our query n-grams
            if word in query_ngrams:
                matched_indices.append(i)
                matched_words.append(word)
                
                # Expansion/Structural Weight from Expander
                mod_weight = query_ngrams[word]
                
                # Calculate IQF Component (Cubic Amplification)
                # Matches logic in train_model.py
                parts = word.split()
                if parts and self.question_idf:
                     iqfs = [self.question_idf.get(p, 1.0) for p in parts]
                     iqf_val = sum(iqfs) / len(iqfs)
                else:
                     iqf_val = 1.0
                
                # Metric: Cubic IQF Boost
                # Strongly favors unique terms (e.g. "US..." vs "sec")
                iqf_boost = iqf_val ** 3.0
                
                # Sequence Length Boost (consistent with training)
                ngram_len = len(parts)
                seq_boost = feature_weights.get_qa_sequence_boost(ngram_len, 1.0)
                
                # Final Inference Weight
                final_weight = mod_weight * iqf_boost * seq_boost
                
                matched_weights.append(final_weight)
        
        if not matched_indices:
            print("No matching selective keywords found in query.")
            return []
            
        print(f"Matched Keywords: {matched_words}")

        # 3. Run Inference (Forward Pass)
        # We run on ALL documents to see which ones reconstruct these keywords best
        node_features = torch.FloatTensor(self.data['node_features'])
        hyperedge_index = torch.LongTensor(self.data['hyperedge_index'])
        
        with torch.no_grad():
            logits = self.model(node_features, hyperedge_index)
            # Use ReLU instead of Sigmoid. Model is trained on log1p(features), so 0 matches 0.
            # Sigmoid(0) would be 0.5, which creates noise.
            reconstructed_probs = torch.relu(logits) # Shape: [num_docs, num_keywords]
            
            # COMBINE Reconstructed (Graph) Signal with Direct (feature) Signal
            # This ensures documents with EXACT matches get highest scores
            # while still allowing graph propagation.
            probs = reconstructed_probs + node_features

        # 4. Score Documents
        # Sum probabilities of the matched keywords for each document
        # We select the columns corresponding to matched keywords
        relevant_probs = probs[:, matched_indices] # Shape: [num_docs, num_matched]
        
        # Apply weights to relevant probabilities
        weights_tensor = torch.FloatTensor(matched_weights)
        if relevant_probs.is_cuda:
            weights_tensor = weights_tensor.to(relevant_probs.device)
        
        # Weighted sum of probabilities
        # Shape: [num_docs, num_matched] * [num_matched] -> [num_docs, num_matched] -> sum -> [num_docs]
        weighted_probs = relevant_probs * weights_tensor
        doc_scores = weighted_probs.sum(dim=1)
        
        # Apply unique ngrams bridge boost
        if self.unique_ngrams_bridge:
            # Tokenize query and generate ngrams
            query_token_list = clean_and_tokenize(user_query)
            query_ngrams_list = generate_ngrams(query_token_list, n_min=1, n_max=5)
            # Create a set for quick lookups
            query_ngram_set = set(query_ngrams_list)
            # Create a boost tensor
            bridge_boost = torch.zeros_like(doc_scores)
            # Check each unique ngram in the bridge
            for entry in self.unique_ngrams_bridge:
                token = entry['token']
                if token in query_ngram_set:
                    # Boost all documents that have this unique ngram
                    for doc_idx in entry['doc_indices']:
                        # Use predefined score if available (Bypass Bridge), else calculation
                        if 'score' in entry:
                             boost_amount = float(entry['score'])
                        else:
                             # Boost by ngram length (longer = more boost) - Legacy fallback
                             boost_amount = 100.0 * (entry['length'] ** 2.0)
                        
                        bridge_boost[doc_idx] += boost_amount
            # Add bridge boost to doc_scores
            doc_scores += bridge_boost
            print(f"Unique ngram bridge boost applied to {len(torch.nonzero(bridge_boost))} docs")
        
        # Additional boosting for overlap count
        if len(matched_indices) > 1:
             shared_counts = (relevant_probs > 0.1).float().sum(dim=1)
             # Log scale boost to prevent explosion, but reward multiple hits
             if feature_weights.INFERENCE_WEIGHTS['shared_match_boost_log']:
                 doc_scores = doc_scores * (1.0 + torch.log1p(shared_counts))
        
        # Normalize scores so sum is 100
        if doc_scores.numel() > 0:
            total_sum = doc_scores.sum()
            if total_sum > 1e-6:
               doc_scores = (doc_scores / total_sum) * 100.0
            else:
               # Avoid division by zero if all scores are effectively zero
               doc_scores = torch.zeros_like(doc_scores)

        # Get top results
        results = []
        docs = self.data['docs']
        
        # Create (index, score) pairs and sort
        scores_list = doc_scores.tolist()
        scored_docs = list(enumerate(scores_list))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top results for query: '{user_query}'")
        for idx, score in scored_docs[:20]:
            if score > 0.01:
                print(f" - {Path(docs[idx]['id']).name}: {score:.4f}")
        
        # Dynamic threshold: Filter out docs with score lower than 10% of top score
        top_score = scored_docs[0][1] if scored_docs else 0
        min_score = max(feature_weights.INFERENCE_WEIGHTS['result_min_score'], top_score * feature_weights.INFERENCE_WEIGHTS['result_ratio_threshold'])

        for idx, score in scored_docs:
            if score >= min_score:
                doc = docs[idx]
                results.append({
                    "id": Path(doc['path']).as_uri(), # Consistent URI format
                    "title": f"[{doc.get('type', 'file').upper()}] {Path(doc['id']).name}",
                    "link": doc['path'], # Serving local path for demo
                    "score": score,
                    "snippet": doc['raw_content'][:200] + "...",
                    "matched_keywords": matched_words
                })
        
        return results[skip : skip + limit]

kg_service = KnowledgeGraphService()
