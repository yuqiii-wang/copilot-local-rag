import os
import networkx as nx
from pathlib import Path
from collections import defaultdict
from services.knowledge_graph.tokenization import clean_and_tokenize, tokenize_log_message

class BaseCodeIndexer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.file_dependency_graph = nx.DiGraph() 
        self.file_literals = defaultdict(list) 

    def index_project(self):
        """
        Walks through root_dir, parses files, and builds dependency graph.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement index_project")

    def get_code_priors(self, hard_weight=10.0, decay=0.5):
        """
        Generates priors:
        - String Literals -> File (Hard Weight)
        - String Literals -> Dependent Files (Decaying Weight - Transitive/Deep)
        """
        priors = defaultdict(lambda: defaultdict(float)) 
        
        for filename, strings in self.file_literals.items():
            if not strings: continue
            
            descendants = {} 
            
            if filename in self.file_dependency_graph:
                for v in self.file_dependency_graph.nodes():
                    if v == filename: continue
                    try:
                        path_len = nx.shortest_path_length(self.file_dependency_graph, source=filename, target=v)
                        descendants[v] = path_len
                    except nx.NetworkXNoPath:
                        pass
            
            for s in strings:
                # Use tokenize_log_message to automatically handle underscore concatenation 
                # for multi-word literals
                tokens = clean_and_tokenize(s)
                
                # Check for sliding window n-grams (1 to 5)
                # We want to boost specific sequences found in the literal
                # But here we are generating priors for specific tokens/keys.
                # If the key is an n-gram, we need to add it to priors if it matches vocabulary.
                
                # Since we don't know the full vocab here easily, we generate n-grams 
                # and assign weights. The graph builder will filter those not in vocab.
                
                # Generate n-grams for this string
                ngrams = []
                # Max n-gram size 5 to match vectorizer
                for n in range(1, 6): 
                     for i in range(len(tokens) - n + 1):
                         ngram = " ".join(tokens[i:i+n])
                         ngrams.append(ngram)
                         
                for gram in ngrams:
                    # Calculate weight boost for this n-gram
                    # Longer n-grams get exponentially higher weight
                    gram_len = len(gram.split())
                    
                    # Base boost * Power of length
                    # E.g. "Starting" (1) -> 10.0
                    # "Starting HFT Strategy" (3) -> 10 * 5^3 = 1250
                    boost_factor = (5.0 ** (gram_len - 1)) 
                    current_weight = hard_weight * boost_factor
                    
                    priors[gram][filename] += current_weight
                    
                    for neighbor, distance in descendants.items():
                         weight = current_weight * (decay ** distance)
                         if weight > 0.01:
                             priors[gram][neighbor] += weight
                             
        return priors
