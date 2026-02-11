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
from services.knowledge_graph.keyword_expander import KeywordExpander

def visualize_knowledge_graph(dataset_path: str, output_file="knowledge_graph.png", max_keywords=50, max_edges=500):
    G = nx.Graph()
    
    # Reserve 10% capacity for latent edges (Keyword->Keyword)
    MIN_LATENT_EDGES = int(max_edges * 0.1)
    
    # Capacity Plan:
    # 1. Structural Edges (Doc->Kw): Target = max_edges - MIN_LATENT_EDGES
    # 2. Latent Edges (Kw->Kw): Target = MIN_LATENT_EDGES (or remaining unused capacity from structural? No, we reserve logic.)
    # 3. If Latent edges < MIN_LATENT_EDGES, give unused slots back to Structural.
    
    structural_capacity_target = max_edges - MIN_LATENT_EDGES
    
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
                structural_candidates = []
                latent_candidates = []
                
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
                    
                    item = {
                        'keyword': keyword,
                        'doc_id': doc_id,
                        'doc_type': doc_type,
                        'score': score
                    }
                    
                    if doc_type == 'keyword':
                        latent_candidates.append(item)
                    else:
                        structural_candidates.append(item)

                # Sort by score descending
                structural_candidates.sort(key=lambda x: x['score'], reverse=True)
                latent_candidates.sort(key=lambda x: x['score'], reverse=True)
                
                print(f"Building graph with max {max_edges} edges...")

                MAX_DEGREE_PER_ENTITY = 20
                added_structural_edges = 0
                
                def add_edge_to_graph(graph, edge_item):
                    keyword = edge_item['keyword']
                    target_id = edge_item['doc_id']
                    target_type = edge_item['doc_type']
                    
                    if target_type == 'keyword':
                        # Keyword -> Keyword (Latent)
                        u = f"kw_{keyword}"
                        v = f"kw_{target_id}"
                        
                        if not graph.has_node(u):
                            graph.add_node(u, label=keyword, type='keyword', node_class='keyword')
                        if not graph.has_node(v):
                            graph.add_node(v, label=target_id, type='keyword', node_class='keyword')
                            
                        # Degree check not strictly applied to latent edges in the same way, or maybe yes?
                        # Let's be lenient for latent edges to ensure connectivity
                        if not graph.has_edge(u, v):
                            graph.add_edge(u, v, type='latent')
                            return True
                    else:
                        # Document -> Keyword (Structural)
                        doc_node_key = f"doc_{target_id}"
                        kw_node_key = f"kw_{keyword}"

                        # Check max degree
                        if graph.has_node(doc_node_key) and graph.degree[doc_node_key] >= MAX_DEGREE_PER_ENTITY:
                            return False
                        if graph.has_node(kw_node_key) and graph.degree[kw_node_key] >= MAX_DEGREE_PER_ENTITY:
                            return False
                        
                        if not graph.has_node(doc_node_key):
                            label = Path(target_id).name
                            if len(label) > 15:
                                label = label[:12] + "..."
                            graph.add_node(doc_node_key, label=label, type=target_type, node_class='document')
                        
                        if not graph.has_node(kw_node_key):
                            graph.add_node(kw_node_key, label=keyword, type='keyword', node_class='keyword')
                        
                        if not graph.has_edge(doc_node_key, kw_node_key):
                            graph.add_edge(doc_node_key, kw_node_key, type='structural')
                            return True
                    return False

                # PHASE 1: Add Initial Budget of Structural Edges
                print(f"Phase 1: Adding up to {structural_capacity_target} structural edges...")
                structural_idx = 0
                while added_structural_edges < structural_capacity_target and structural_idx < len(structural_candidates):
                    item = structural_candidates[structural_idx]
                    if add_edge_to_graph(G, item):
                        added_structural_edges += 1
                    structural_idx += 1

                # Phase 2: Latent
                latent_added = 0
                
                # Option A: From CSV (if available)
                if latent_candidates:
                    print(f"Phase 2: Adding up to {MIN_LATENT_EDGES} latent edges from CSV...")
                    latent_idx = 0
                    while latent_added < MIN_LATENT_EDGES and latent_idx < len(latent_candidates):
                        item = latent_candidates[latent_idx]
                        if add_edge_to_graph(G, item):
                            latent_added += 1
                        latent_idx += 1
                else: 
                    # Option B: Fallback to Pickle
                    expander_path = current_dir / "keyword_expander.pkl"
                    if expander_path.exists():
                        expander = KeywordExpander()
                        try:
                            expander.load(str(expander_path))
                            lbl_to_node = {attrs['label']: n for n, attrs in G.nodes(data=True) if attrs.get('node_class') == 'keyword'}
                            for src, tgts in expander.keyword_edges.items():
                                if latent_added >= MIN_LATENT_EDGES: break
                                if src in lbl_to_node:
                                    u = lbl_to_node[src]
                                    for tgt, sc in tgts.items():
                                        if latent_added >= MIN_LATENT_EDGES: break
                                        if tgt in lbl_to_node:
                                            v = lbl_to_node[tgt]
                                            if sc > 0.5 and not G.has_edge(u, v):
                                                G.add_edge(u, v, type='latent', weight=sc)
                                                latent_added += 1
                            print(f"Added {latent_added} latent edges from expander.")
                        except: pass
                
                # Phase 3: Backfill
                unused = MIN_LATENT_EDGES - latent_added
                if unused > 0:
                    print(f"Phase 3: Backfilling {unused} slots from structural candidates...")
                    structural_target_final = structural_capacity_target + unused
                    # continue from where we left off
                    while added_structural_edges < structural_target_final and structural_idx < len(structural_candidates):
                        item = structural_candidates[structural_idx]
                        if add_edge_to_graph(G, item):
                            added_structural_edges += 1
                        structural_idx += 1
                        
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
            
    else:
        # CSV not found - Use Feature Generator
        print("CSV not found. Generating graph data from features...")
        data = generate_features(dataset_path, max_features=max_keywords)
        
        if not data:
            print("No data found.")
            return

        docs = data['docs']
        vocab = data['vocab']
        hyperedge_index = data.get('hyperedge_index')
        
        if hasattr(hyperedge_index, 'numpy'):
            hyperedge_index = hyperedge_index.numpy()
            
        doc_indices = hyperedge_index[0]
        kw_indices = hyperedge_index[1]

        # Add Nodes
        for i, doc in enumerate(docs):
            node_id = f"doc_{i}"
            label = Path(doc['id']).name
            if len(label) > 15: label = label[:12] + "..."
            G.add_node(node_id, label=label, type=doc.get('type', 'unknown'), node_class='document')
            
        for i, word in enumerate(vocab):
            node_id = f"kw_{i}"
            G.add_node(node_id, label=word, type='keyword', node_class='keyword')
            
        # Phase 1: Structural
        added_structural = 0
        cur_idx = 0
        total_avail = len(doc_indices) if doc_indices is not None else 0
        
        print(f"Phase 1: Adding structural edges from features (Target {structural_capacity_target})...")
        while added_structural < structural_capacity_target and cur_idx < total_avail:
            d = doc_indices[cur_idx]
            k = kw_indices[cur_idx]
            dn, kn = f"doc_{d}", f"kw_{k}"
            MAX_DEGREE = 20
            
            if G.has_node(dn) and G.has_node(kn):
                if G.degree[dn] < MAX_DEGREE and G.degree[kn] < MAX_DEGREE:
                     if not G.has_edge(dn, kn):
                        G.add_edge(dn, kn, type='structural')
                        added_structural += 1
            cur_idx += 1
            
        # Phase 2: Latent
        latent_added = 0
        expander_path = current_dir / "keyword_expander.pkl"
        if expander_path.exists():
            expander = KeywordExpander()
            try:
                expander.load(str(expander_path))
                lbl_to_node = {attrs['label']: n for n, attrs in G.nodes(data=True) if attrs.get('node_class') == 'keyword'}
                for src, tgts in expander.keyword_edges.items():
                    if latent_added >= MIN_LATENT_EDGES: break
                    if src in lbl_to_node:
                        u = lbl_to_node[src]
                        for tgt, sc in tgts.items():
                            if latent_added >= MIN_LATENT_EDGES: break
                            if tgt in lbl_to_node:
                                v = lbl_to_node[tgt]
                                if sc > 0.5 and not G.has_edge(u, v):
                                    G.add_edge(u, v, type='latent', weight=sc)
                                    latent_added += 1
                print(f"Added {latent_added} latent edges from expander.")
            except: pass
            
        # Phase 3: Backfill
        unused = MIN_LATENT_EDGES - latent_added
        if unused > 0:
            print(f"Phase 3: Backfilling {unused} slots...")
            tgt = added_structural + unused
            while added_structural < tgt and cur_idx < total_avail:
                 d = doc_indices[cur_idx]
                 k = kw_indices[cur_idx]
                 dn, kn = f"doc_{d}", f"kw_{k}"
                 MAX_DEGREE = 20
                 if G.has_node(dn) and G.has_node(kn):
                    if G.degree[dn] < MAX_DEGREE and G.degree[kn] < MAX_DEGREE:
                        if not G.has_edge(dn, kn):
                             G.add_edge(dn, kn, type='structural')
                             added_structural += 1
                 cur_idx += 1

    if G.number_of_edges() == 0:
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
    # Separate structural and latent edges
    structural_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') != 'latent']
    latent_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'latent']

    nx.draw_networkx_edges(G, pos, edgelist=structural_edges, alpha=0.2, edge_color='#666666')
    if latent_edges:
        nx.draw_networkx_edges(G, pos, edgelist=latent_edges, alpha=0.6, edge_color='#888888', style='dashed', width=1.0)
    
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
        Line2D([0], [0], color='#888888', linestyle='--', label='Latent Connection', linewidth=1),
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
