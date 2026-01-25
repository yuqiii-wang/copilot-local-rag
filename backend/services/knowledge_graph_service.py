import torch
import sys
from pathlib import Path
from typing import List, Dict
from services.knowledge_graph.graph_model import KeywordReconstructionHGNN
from services.knowledge_graph.feature_generator import generate_features
from services.knowledge_graph.tokenization import clean_and_tokenize

# Add knowledge graph directory to path
current_dir = Path(__file__).parent
kg_dir = current_dir / "knowledge_graph"

class KnowledgeGraphService:
    def __init__(self):
        self.model = None
        self.data = None
        self.dataset_path = str(kg_dir / "dummy_dataset")
        # Use valid trained model path
        self.model_path = kg_dir / "query_model.pth"
        self._initialize()

    def _initialize(self):
        print("Initializing Knowledge Graph Service...")
        # 1. Load Data & Features 
        # Prefer loading the same feature generator used in training to match vocabulary
        fg_path = kg_dir / "feature_gen.pkl"
        
        try:
            # We must use the same generator to ensure 'vocab' indices align with model weights
            from services.knowledge_graph.feature_generator import FeatureGenerator
            fg = FeatureGenerator()
            fg.load(str(fg_path))
            # Reconstruct the 'data' dict expected by the service
            coo = fg.features.tocoo()
            row = torch.from_numpy(coo.row.astype('int64'))
            col = torch.from_numpy(coo.col.astype('int64'))
            hyperedge_index = torch.stack([row, col], dim=0)
            
            self.data = {
                'vocab': fg.vectorizer.get_feature_names_out(),
                'node_features': fg.features.toarray(), # Features for nodes (docs)
                'hyperedge_index': hyperedge_index,
                'docs': [{'id': name, 'path': name, 'raw_content': "...", 'type': 'file'} for name in fg.doc_names]
            }
            # Note: raw_content is missing in pickle, simplified for this patch
            
            self.vocab = self.data['vocab']
            self.num_features = self.data['node_features'].shape[1]
            self.num_keywords = len(self.vocab)
            print(f"Loaded features from {fg_path}. Vocab size: {self.num_keywords}")

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
        if self.model_path.exists():
            try:
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading model weights: {e}")
        else:
            print("Model weights file not found. Service will run with untrained weights (useless).")

    def query_graph(self, user_query: str) -> List[Dict]:
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
        
        # 2. Match with Vocab
        matched_indices = []
        matched_words = []
        for i, word in enumerate(self.vocab):
            # Simple exact match or localized match
            if word in query_tokens:
                matched_indices.append(i)
                matched_words.append(word)
        
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
            probs = torch.sigmoid(logits) # Shape: [num_docs, num_keywords]

        # 4. Score Documents
        # Sum probabilities of the matched keywords for each document
        # We select the columns corresponding to matched keywords
        relevant_probs = probs[:, matched_indices] # Shape: [num_docs, num_matched]
        
        # Modified Logic: Score = Sum(probs) * Count(shared_keywords)
        if len(matched_indices) > 1:
            # We consider a keyword "shared" if the probability is significant (>0.3)
            base_scores = relevant_probs.sum(dim=1)
            shared_counts = (relevant_probs > 0.3).float().sum(dim=1)
            doc_scores = base_scores * shared_counts
        else:
            doc_scores = relevant_probs.sum(dim=1) # Shape: [num_docs]

        
        # Get top results
        results = []
        docs = self.data['docs']
        
        # Create (index, score) pairs and sort
        scores_list = doc_scores.tolist()
        scored_docs = list(enumerate(scores_list))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        for idx, score in scored_docs:
            if score > 0.01: # Threshold
                doc = docs[idx]
                results.append({
                    "title": f"[{doc.get('type', 'file').upper()}] {Path(doc['id']).name}",
                    "link": doc['path'], # Serving local path for demo
                    "score": score,
                    "snippet": doc['raw_content'][:200] + "...",
                    "matched_keywords": matched_words
                })
        
        return results[:5] # Top 5

kg_service = KnowledgeGraphService()
