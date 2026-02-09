# Knowledge Graph Service

This service builds a Hypergraph-based Knowledge Graph to link code, documentation (Confluence), and issue tracking (Jira) entities based on semantic similarity and shared keywords.

The model is learning the weighted connection between a Document and a Keyword.

Input: A sparse matrix $X$ where $X_{ij}$ represents the initial "strength" of Keyword $j$ in Document $i$.

This strength comes from:

* TF-IDF: Actual word count in the file.
* Code Priors: Imported dependencies or variable usage.
* QA History: Previous human queries associated with the file.

Output: A reconstructed matrix where the model predicts what the strength of the connection should be, based on the context of the entire graph.

## Architecture

### 1. Tokenization & Indexing
The system uses specialized tokenization strategies for different file types:

*   **Code Indexers (`code_indexer/`)**:
    *   **Java**: `javalang` AST parsing for classes, methods, and *String Literals*.
    *   **C++**: Regex/Clang-based extraction for includes and definitions.
    *   **SQL**: Table and column extraction.
    *   **Bash**: Command and argument parsing.
    *   **Dependency Graph**: Builds a directed graph of class dependencies to propagate context (e.g., if A uses B, keywords in B are relevant to A).
*   **Text/Docs (`tokenization.py`)**:
    *   Regex-based pattern matching (UUIDs, ISINs, CUSIPs, Dates).
    *   CamelCase and snake_case splitting.
    *   Standard stop-word removal.

### 2. Feature Generation (`feature_generator.py`)
*   **Vectorization**: Uses `TfidfVectorizer` to convert documents into sparse vectors.
*   **Settings**: 
    *   N-grams: 1 to 5 (captures phrases). Uses space delimiters to match vectorizer format.
    *   **Min DF: 1** (Allows unique code literals).
    *   Max DF: 0.85 (filters stopwords).
*   **Augmentations**:
    *   **Topic Modeling (NMF)**: Extracts 10 latent topics (e.g., "Risk", "Order Management") to group documents thematically.
    *   **Recency Weighting**: Recently modified files receive a small weight boost (up to 5%).
    *   **QA Text Injection**: Text from known QA pairs is appended to document content to artificially increase term frequency for relevant keywords.

### 3. Model: Keyword Reconstruction HGNN (`graph_model.py`)
A Hypergraph Neural Network (HGNN) is trained to reconstruct the keyword features of documents, learning latent relationships between files that share similar terminology.

*   **Architecture**:
    *   Layer 1: `HypergraphConv` (Input vocab -> Hidden dim 128) + ReLU + Dropout.
    *   Layer 2: `HypergraphConv` (Hidden -> Hidden) + ReLU.
    *   Output Head: Linear (Hidden -> Output vocab).
*   **Purpose**: To smooth representations. If File A and File B share many keywords (neighbors), the GNN makes their embeddings similar. This helps retrieve "related" files even if they don't share the *exact* search term but share semantic context.

### 4. Training & Prior Injection (`train_model.py`)
The training process is "Self-Supervised Reconstruction" with "Hard Priors".

1.  **Input Features ($X$)**: Log-transformed TF-IDF Matrix (`log1p`).
2.  **Prior Injection**:
    *   **QA Dataset**: Known QA pairs are injected.
    *   **Conversation Records**: Past chat interactions (Human/AI) are treated as signals.
    *   **N-Gram Boosting**: Longer phrases (e.g., "margin call procedure") are boosted exponentially ($3.5^{length}$) over single words.
    *   **Topic Injection**: Latent topic scores (NMF) are appended to the node feature matrix ($X$) and create new "Topic Hyperedges" connecting conceptually related documents.
    *   **Code Literals**: Hardcoded strings in code are treated as "pending queries" mapping to their source file.
    *   **Structural Patterns**: Files sharing regex patterns (ISINs, UUIDs) receive a connection boost.
3.  **Loss Function**: **Weighted MSE Loss**.
    *   Standard MSE is dominated by zeros (sparsity > 99%).
    *   We apply a **20x weight** to non-zero targets to force the model to respect known signals (avoiding "zero-collapse").
4.  **Optimization**: AdamW with Cosine Annealing Learning Rate scheduler. (500 Epochs).

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