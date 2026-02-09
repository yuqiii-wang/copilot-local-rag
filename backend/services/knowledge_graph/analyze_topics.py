import os
import sys
import numpy as np
from sklearn.decomposition import NMF

# Add backend to path
sys.path.append(os.getcwd())

from services.knowledge_graph.feature_generator import FeatureGenerator

def analyze_topics(dataset_path, n_topics=5):
    print(f"Loading data from {dataset_path}...")
    fg = FeatureGenerator(max_features=1000)
    fg.load_data(dataset_path)
    
    print("Fitting Vectorizer...")
    # This generates the TF-IDF matrix (Docs x Terms)
    tfidf_matrix = fg.fit()
    
    print(f"Applying NMF with {n_topics} topics...")
    # Initialize NMF
    nmf = NMF(n_components=n_topics, random_state=42, init='nndsvd')
    # Fit NMF (Docs x Topics)
    W = nmf.fit_transform(tfidf_matrix)
    # H matrix (Topics x Terms)
    H = nmf.components_
    
    feature_names = fg.vectorizer.get_feature_names_out()
    
    print("\n--- Discovered Topics ---")
    for topic_idx, topic in enumerate(H):
        top_indices = topic.argsort()[:-11:-1] # Top 10 terms
        top_terms = [feature_names[i] for i in top_indices]
        print(f"Topic {topic_idx + 1}: {', '.join(top_terms)}")
        
    return W, H

if __name__ == "__main__":
    dataset_dir = os.path.join("services", "knowledge_graph", "dummy_dataset")
    analyze_topics(dataset_dir)
