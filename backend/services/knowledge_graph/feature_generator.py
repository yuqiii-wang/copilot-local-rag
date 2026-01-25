import os
import pickle
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
# Import both tokenization methods
from services.knowledge_graph.tokenization import clean_and_tokenize, extract_java_tokens

def whitespace_tokenizer(text):
    return text.split()

def generate_features(dataset_path: str, max_features: int = 1000):
    """
    Generates features for the knowledge graph.
    Returns a dictionary with docs, vocab, and hyperedge_index.
    """
    fg = FeatureGenerator(max_features=max_features)
    fg.load_data(dataset_path)
    features = fg.fit()
    
    # Construct return dict
    vocab = fg.vectorizer.get_feature_names_out()
    
    # Build a docs list compatible with visualize_graph
    docs = []
    for name in fg.doc_names:
        # Simple heuristic for type
        if name.endswith('.java'): 
            d_type = 'code'
        elif 'jira' in name.lower() or 'bug' in name.lower() or 'trd' in name.lower():
            d_type = 'jira'
        else:
            d_type = 'confluence'
        
        docs.append({
            'id': name,
            'type': d_type
        })
        
    # Hyperedge Index
    coo = features.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    hyperedge_index = torch.stack([row, col], dim=0)

    return {
        'docs': docs,
        'vocab': vocab,
        'hyperedge_index': hyperedge_index,
        'features': features
    }

class FeatureGenerator:
    def __init__(self, max_features=1000):
        # We process content before passing to vectorizer, so we use a whitespace tokenizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, 
            min_df=2, 
            max_df=0.85, 
            stop_words='english', 
            ngram_range=(1, 5),
            tokenizer=whitespace_tokenizer, # Input is already space-separated tokens
            token_pattern=None
        )
        self.file_map = {} # filename -> integers
        self.reverse_file_map = {} # integer -> filename
        self.doc_contents = []
        self.doc_names = []

    def load_data(self, root_dir):
        """
        Walks through the directory and loads text files.
        Applies specific tokenization logic based on file type.
        """
        print(f"Loading files from {root_dir}")
        file_list = []
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # Filter for relevant text files
                if file.endswith(('.txt', '.java', '.md', '.py', '.sql')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            raw_content = f.read()
                            
                        # Apply specialized tokenization immediately
                        if file.endswith('.java'):
                            tokens = extract_java_tokens(raw_content)
                        else:
                            tokens = clean_and_tokenize(raw_content)
                            
                        # Join tokens into a string for TfidfVectorizer
                        # This pre-processing allows us to use different logic per file type
                        processed_content = " ".join(tokens)
                        
                        file_list.append((file, processed_content))
                        
                    except Exception as e:
                        print(f"Error reading {file}: {e}")
        
        # Add QA Dataset keys as a "synthetic" document to ensure they exist in vocabulary
        # if min_df=2 is set. 
        # This increases the document frequency of these terms by 1.
        qa_file_path = os.path.join(root_dir, "qa_dataset.json")
        if os.path.exists(qa_file_path):
            try:
                import json
                with open(qa_file_path, 'r') as f:
                    qa_data = json.load(f)
                
                # Extract all tokens from keys
                all_qa_tokens = []
                for query in qa_data.keys():
                    all_qa_tokens.extend(clean_and_tokenize(query))
                
                # Combine into one document
                qa_doc_content = " ".join(all_qa_tokens)
                file_list.append(("__QA_KEYS_METADATA__", qa_doc_content))
                print(f"Loaded QA Dataset Metada as synthetic document with {len(all_qa_tokens)} tokens")
            except Exception as e:
                print(f"Error loading QA dataset metadata: {e}")

        # Sort to ensure consistent mapping
        file_list.sort(key=lambda x: x[0])
        
        self.doc_names = [x[0] for x in file_list]
        self.doc_contents = [x[1] for x in file_list]
        
        # Build mapping
        self.file_map = {name: i for i, name in enumerate(self.doc_names)}
        self.reverse_file_map = {i: name for i, name in enumerate(self.doc_names)}

    def fit(self):
        print("Fitting TfidfVectorizer...")
        self.features = self.vectorizer.fit_transform(self.doc_contents)
        print(f"Feature matrix shape: {self.features.shape}")
        return self.features

    def transform(self, texts):
        # For query transformation, we should probably stick to standard tokenization 
        # unless we want to parse the query as Java (unlikely)
        # However, since vectorizer expects space-separated tokens now:
        processed_texts = [" ".join(clean_and_tokenize(text)) for text in texts]
        return self.vectorizer.transform(processed_texts)

    def save(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'file_map': self.file_map,
                'reverse_file_map': self.reverse_file_map,
                'doc_names': self.doc_names,
                'doc_contents': self.doc_contents # Save contents to allow feature reconstruction
            }, f)
        print(f"Feature generator saved to {output_path}")

    def load(self, input_path):
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.file_map = data['file_map']
            self.reverse_file_map = data['reverse_file_map']
            self.doc_names = data['doc_names']
            # Try to load doc contents if saved, otherwise we can't regenerate features easily without reloading data
            if 'doc_contents' in data:
                self.doc_contents = data['doc_contents']
                # If we have contents, we can transform to get features
                self.features = self.vectorizer.transform(self.doc_contents)
            else:
                 # Limitation: If we didn't save contents, we can't restore 'self.features' exactly without reloading files
                 # However, usually we save after fit()
                 pass
        print(f"Feature generator loaded from {input_path}")
