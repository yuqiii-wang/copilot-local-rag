# Knowledge Graph Service

This service builds a Hypergraph-based Knowledge Graph to link code, documentation (Confluence), and issue tracking (Jira) entities based on semantic similarity and shared keywords.

## Architecture

### 1. Tokenization & Indexing
The system uses specialized tokenization strategies for different file types:

*   **Java Code (`java_code_indexer.py`, `tokenization.py`)**:
    *   Uses `javalang` to parse Abstract Syntax Trees (AST).
    *   Extracts structural tokens: Class names, Method names, Variable types.
    *   Extracts String Literals (e.g., error messages, business keys like "USOPTIONS4").
    *   **Dependency Graph**: Builds a directed graph of class dependencies to propagate context (e.g., if A uses B, keywords in B are relevant to A).
*   **Text/Docs (`tokenization.py`)**:
    *   Regex-based pattern matching (UUIDs, ISINs, CUSIPs, Dates).
    *   CamelCase and snake_case splitting.
    *   Standard stop-word removal.

### 2. Feature Generation (`feature_generator.py`)
*   **Vectorization**: Uses `TfidfVectorizer` to convert documents into sparse vectors.
*   **Settings**: 
    *   N-grams: 1 to 5 (captures phrases).
    *   Min DF: 2 (filters noise).
    *   Max DF: 0.85 (filters stopwords).
*   **Hypergraph Structure**: 
    *   **Nodes**: Documents (Files).
    *   **Hyperedges**: Keywords (Tokens).
    *   An index is built where a Document is connected to a Keyword if the TF-IDF score is non-zero.

### 3. Model: Keyword Reconstruction HGNN (`graph_model.py`)
A Hypergraph Neural Network (HGNN) is trained to reconstruct the keyword features of documents, learning latent relationships between files that share similar terminology.

*   **Architecture**:
    *   Layer 1: `HypergraphConv` (Input vocab -> Hidden dim 128) + ReLU + Dropout.
    *   Layer 2: `HypergraphConv` (Hidden -> Hidden) + ReLU.
    *   Output Head: Linear (Hidden -> Output vocab).
*   **Purpose**: To smooth representations. If File A and File B share many keywords (neighbors), the GNN makes their embeddings similar. This helps retrieve "related" files even if they don't share the *exact* search term but share semantic context.

### 4. Training & Prior Injection (`train_model.py`)
The training process is "Self-Supervised Reconstruction" with "Hard Priors".

1.  **Input Features ($X$)**: TF-IDF Matrix.
2.  **Prior Injection**:
    *   **QA Dataset**: Known QA pairs (Query -> File) are injected into $X$ with high weight, teaching the model that specific tokens imply specific files.
    *   **Code Dependencies**: If File A relies on File B, keywords from B are injected into A's feature vector (Softened by distance decay).
3.  **Loss Function**: `SmoothL1Loss` (Huber Loss) to handle outliers/bursty terms.
4.  **Optimization**: AdamW with Cosine Annealing Learning Rate scheduler.

## Design Rationale & Evolution

This architecture evolved through specific challenges encountered during development:

### Why Hypergraph Neural Network?
Initial keyword search (TF-IDF) is brittle. It misses "conceptually related" files.
*   **Observation**: A test file (`OptionTest.java`) and a base class (`DummyTestBase.java`) are strictly related.
*   **Solution**: The HGNN performs "Message Passing". Signals propagate through shared keywords. If `OptionTest` has a unique key, and it shares 90% of its structure (imports, setup methods) with `EquityTest`, the GNN learns they are related.

### The "Oversmoothing" Problem
During testing, we observed a **Signal Leakage** issue:
*   **Scenario**: The token `USOPTIONS4` existed *only* in `OptionTest.java`.
*   **Issue**: The GNN gave `EquityTest.java` a high score (~0.63) for `USOPTIONS4`, nearly identical to `OptionTest` (~0.65).
*   **Cause**: `EquityTest` and `OptionTest` are structurally almost identical (siblings). The GNN propagated the signal from `OptionTest` -> Shared Keywords -> `EquityTest`. The unique signal of `USOPTIONS4` was drowned out by the hundreds of shared structural tokens in the 128-dim bottleneck.

### The Fix: Hybrid Scoring (Residual Connection)
To resolve the leakage while keeping the benefits of the graph:
*   **Inference Change**:
    $$ \text{Final Score} = \text{GNN Output}(X) + \text{Input Features}(X) $$
*   **Logic**:
    *   **GNN Output**: Provides the "Relatedness" / "Smoothed" score.
    *   **Input Features ($X$)**: Contains the raw, sharp signal (TF-IDF + Injected Priors).
*   **Result**:
    *   `OptionTest` (Source): 2.67 (Original High Signal + Graph Support).
    *   `EquityTest` (Neighbor): 0.57 (Graph Support only).
    *   This successfully distinguishes exact matches from related context.
