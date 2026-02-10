import torch
import sys
import os
from pathlib import Path
from typing import List, Dict
from services.knowledge_graph.graph_model import KeywordReconstructionHGNN
from services.knowledge_graph.feature_generator import generate_features
from services.knowledge_graph.tokenization import clean_and_tokenize
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
        self.dataset_path = os.path.join(kg_dir, "dummy_dataset")
        # Use valid trained model path
        self.model_path = os.path.join(kg_dir, "query_model.pth")
        # Initialization is now async via load_async()
    
    async def load_async(self):
        """Asynchronous initialization wrapper"""
        if self.model is not None:
             return
        
        print("Scheduling Knowledge Graph initialization...")
        import asyncio
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._initialize)

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
            features_tensor = torch.from_numpy(fg.features.toarray()).float()
            
            vocab_list = fg.vectorizer.get_feature_names_out()
            ngram_weights = [feature_weights.get_ngram_boost(w) for w in vocab_list] 
            ngram_weights_tensor = torch.FloatTensor(ngram_weights)
            
            features_tensor = features_tensor * ngram_weights_tensor
            
            # Augment with Topic Features if available (must match train_model.py)
            if hasattr(fg, 'doc_topic_matrix') and fg.doc_topic_matrix is not None:
                topic_features = torch.FloatTensor(fg.doc_topic_matrix)
                topic_features = topic_features * 10.0 # Match training scale
                features_tensor = torch.cat([features_tensor, topic_features], dim=1)
                print(f"Augmented features with {fg.doc_topic_matrix.shape[1]} topics.")

            # Log transform features to match training distribution
            features_array = torch.log1p(features_tensor).numpy()

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
            print(f"Loaded features from {fg_path}. Input dim: {self.num_features}")

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
        
        # Generate n-grams (1-5) to match sequence in vocabulary
        query_ngrams = set(query_tokens) # Unigrams
        for n in range(2, 6):
            for i in range(len(query_tokens) - n + 1):
                ngram = " ".join(query_tokens[i:i+n])
                query_ngrams.add(ngram)
        
        # 2. Match with Vocab
        matched_indices = []
        matched_words = []
        matched_weights = []

        for i, word in enumerate(self.vocab):
            # Check if vocab word is in our query n-grams
            if word in query_ngrams:
                matched_indices.append(i)
                matched_words.append(word)
                # Weight by word count squared to favor longer precise matches
                # e.g. 'succeed at least once' (4) -> 64 (cubed) to ensure dominance
                #      'should' (1) -> 1
                word_count = len(word.split())
                matched_weights.append(word_count ** 3)
        
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
        
        # Additional boosting for overlap count
        if len(matched_indices) > 1:
             shared_counts = (relevant_probs > 0.1).float().sum(dim=1)
             # Log scale boost to prevent explosion, but reward multiple hits
             doc_scores = doc_scores * (1.0 + torch.log1p(shared_counts))
        
        # Normalize scores using Softmax as requested
        # We multiply by 100 to maintain percentage-like scaling consistent with UI expectations
        if doc_scores.numel() > 0:
            doc_scores = torch.softmax(doc_scores, dim=0) * config.SCORE_NORMALIZATION_FACTOR

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
        min_score = max(0.01, top_score * 0.1)

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
