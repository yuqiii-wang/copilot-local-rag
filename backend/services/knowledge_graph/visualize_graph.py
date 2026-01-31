import sys
from pathlib import Path

# Add backend directory to sys.path
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent.parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

import networkx as nx
import matplotlib.pyplot as plt
import csv
from services.knowledge_graph.feature_generator import generate_features

def visualize_knowledge_graph(dataset_path: str, output_file="knowledge_graph.png", max_keywords=50, max_edges=100):
    G = nx.Graph()
    
    # Define colors for different node types
    color_map = {
        'jira': '#FF9999',       # Red (Jira)
        'code': '#9999FF',       # Blue (Code)
        'confluence': '#99FF99', # Green (Confluence)
        'unknown': '#CCCCCC'
    }

    csv_path = current_dir / "hypergraph_results.csv"
    
    if csv_path.exists():
        print(f"Loading graph data from {csv_path}...")
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    print("CSV is empty.")
                    return

                print(f"Found {len(rows)} connections in CSV. Filtering and sorting...")
                
                # Collect edges with scores
                edges_list = []
                for row in rows:
                    keyword = row['Keyword']
                    if len(keyword.strip().split()) > 3:
                        continue
                        
                    doc_id = row['Document ID']
                    doc_type = row.get('Document Type', 'unknown')
                    try:
                        score = float(row.get('Score', 0.0))
                    except (ValueError, KeyError):
                        score = 0.0
                        
                    edges_list.append({
                        'keyword': keyword,
                        'doc_id': doc_id,
                        'doc_type': doc_type,
                        'score': score
                    })

                # Sort by score descending and limit
                edges_list.sort(key=lambda x: x['score'], reverse=True)
                edges_list = edges_list[:max_edges]
                
                print(f"Building graph with top {len(edges_list)} edges...")

                for item in edges_list:
                    keyword = item['keyword']
                    doc_id = item['doc_id']
                    doc_type = item['doc_type']

                    # Document Node
                    # Use doc_id as unique key. 
                    # Note: Original code used doc_{i} index. Here we use path or ID string as key.
                    doc_node_key = f"doc_{doc_id}"
                    
                    if not G.has_node(doc_node_key):
                        label = Path(doc_id).name
                        if len(label) > 15:
                            label = label[:12] + "..."
                        G.add_node(doc_node_key, label=label, type=doc_type, node_class='document')
                    
                    # Keyword Node
                    kw_node_key = f"kw_{keyword}"
                    if not G.has_node(kw_node_key):
                        G.add_node(kw_node_key, label=keyword, type='keyword', node_class='keyword')
                    
                    # Edge
                    if not G.has_edge(doc_node_key, kw_node_key):
                        G.add_edge(doc_node_key, kw_node_key)
                        
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
            
    else:
        print("CSV not found. Generating graph data from features...")
        # Using the feature generator to get the graph structure
        # We limit keywords to keep the visualization readable
        data = generate_features(dataset_path, max_features=max_keywords)
        
        if not data:
            print("No data found.")
            return

        # Extract data
        docs = data['docs']
        vocab = data['vocab']
        added_edges = 0
        if len(doc_indices.shape) > 0:
            for d_idx, k_idx in zip(doc_indices, kw_indices):
                if added_edges >= max_edges:
                    break
                    
                word = vocab[k_idx]
                if len(word.strip().split()) > 3:
                    continue

                G.add_edge(f"doc_{d_idx}", f"kw_{k_idx}")
                added_edges += 1
        for i, doc in enumerate(docs):
            node_id = f"doc_{i}"
            doc_type = doc.get('type', 'unknown')
            
            # Use filename as label, truncated if too long
            label = Path(doc['id']).name
            if len(label) > 15:
                label = label[:12] + "..."
                
            G.add_node(node_id, label=label, type=doc_type, node_class='document')
        
        # 2. Add Keyword Nodes
        print(f"Adding {len(vocab)} keyword nodes...")
        for i, word in enumerate(vocab):
            node_id = f"kw_{i}"
            G.add_node(node_id, label=word, type='keyword', node_class='keyword')

        # 3. Add Edges (Document <-> Keyword)
        print("Adding edges...")
        # hyperedge_index is numpy array
        doc_indices = hyperedge_index[0]
        kw_indices = hyperedge_index[1]
        
        # Zip iterating over columns of the adjacency matrix essentially
        if len(doc_indices.shape) > 0:
            for d_idx, k_idx in zip(doc_indices, kw_indices):
                G.add_edge(f"doc_{d_idx}", f"kw_{k_idx}")
            
    num_edges = G.number_of_edges()
    print(f"Graph created with {G.number_of_nodes()} nodes and {num_edges} edges.")

    if num_edges == 0:
        print("Warning: No edges found. Visualization will show disconnected nodes.")

    # Visualization Layout
    print("Computing layout (this may take a moment)...")
    # spring_layout behaves well for these connected components
    # Increased k to 0.4 for sparsity (default is usually smaller), increased iterations for stability
    pos = nx.spring_layout(G, k=0.4, iterations=100, seed=42)
    
    plt.figure(figsize=(20, 16)) # Increased figure size to help with spacing
    
    # Draw Document Nodes
    doc_nodes = [n for n, attr in G.nodes(data=True) if attr['node_class'] == 'document']
    doc_node_colors = [color_map[G.nodes[n]['type']] for n in doc_nodes]
    
    # Reduced node_size from 500 to 300 and keyword size to 150 to reduce visual overlap
    nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color=doc_node_colors, node_size=300, alpha=0.9, edgecolors='black')
    
    # Draw Keyword Nodes
    kw_nodes = [n for n, attr in G.nodes(data=True) if attr['node_class'] == 'keyword']
    nx.draw_networkx_nodes(G, pos, nodelist=kw_nodes, node_color='#FFCC00', node_size=150, alpha=0.8, node_shape='s', edgecolors='black')
    
    # Draw Edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='#666666')
    
    # Draw Labels
    doc_labels = {n: G.nodes[n]['label'] for n in doc_nodes}
    kw_labels = {n: G.nodes[n]['label'] for n in kw_nodes}
    
    nx.draw_networkx_labels(G, pos, labels=doc_labels, font_size=8, font_color='#000000')
    nx.draw_networkx_labels(G, pos, labels=kw_labels, font_size=9, font_color='#663300', font_weight='bold')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['jira'], label='Jira Ticket', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['code'], label='Source Code', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map['confluence'], label='Confluence Page', markersize=12, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFCC00', label='Keyword', markersize=12, markeredgecolor='black'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.title(f"Knowledge Graph Visualization\n(Documents linked by Shared Selective Keywords - Top {max_edges} Edges)", fontsize=16)
    plt.axis('off')
    
    output_path = current_dir / output_file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved to {output_path}")

if __name__ == "__main__":
    dataset_dir = current_dir / "dummy_dataset"
    visualize_knowledge_graph(str(dataset_dir))
