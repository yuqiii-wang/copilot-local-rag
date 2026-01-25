import os
import javalang
from pathlib import Path
from collections import defaultdict
import networkx as nx
from services.knowledge_graph.tokenization import clean_and_tokenize

class JavaCodeIndexer:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        # Graph stores relationships: File -> File (Dependencies)
        self.file_dependency_graph = nx.DiGraph() 
        self.class_to_file = {} # ClassName -> Filename
        self.file_literals = defaultdict(list) # Filename -> List[String content]
        self.pending_deps = defaultdict(set) # Filename -> Set[Dep ClassNames]

    def index_project(self):
        print(f"Indexing Java project in {self.root_dir}...")
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".java"):
                    self.parse_file(Path(root) / file)
        
        # Resolve Dependencies
        self._resolve_dependencies()
        
        print(f"Indexing complete. Built dependency graph with {self.file_dependency_graph.number_of_nodes()} files and {self.file_dependency_graph.number_of_edges()} edges.")

    def parse_file(self, file_path):
        filename = file_path.name
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = javalang.parse.parse(content)
            
            # 1. Identify Class Definition (Map Class -> File) AND Inheritance
            for path, node in tree:
                if isinstance(node, javalang.tree.ClassDeclaration):
                    self.class_to_file[node.name] = filename
                    
                    # Handle Inheritance Dependencies here (merged)
                    if node.extends:
                         if isinstance(node.extends, list):
                             for ext in node.extends:
                                 if hasattr(ext, 'name'): self.pending_deps[filename].add(ext.name)
                         elif hasattr(node.extends, 'name'):
                             self.pending_deps[filename].add(node.extends.name)

                # 2. Identify String Literals (Logs, Errors)
                elif isinstance(node, javalang.tree.Literal):
                    val = str(node.value)
                    if val.startswith('"') and val.endswith('"'):
                        # Strip quotes
                        self.file_literals[filename].append(val[1:-1])

                # 3. Identify Dependencies (Fields/Variables)
                elif isinstance(node, javalang.tree.FieldDeclaration):
                    if node.type and hasattr(node.type, 'name'):
                        self.pending_deps[filename].add(node.type.name)
                
                # 4. Identify Dependencies (Inheritance) - REMOVED (Merged into Step 1)


        except Exception as e:
            if 'Test' in filename:
                print(f"Failed to parse {filename}: {e}")
            pass

    def _resolve_dependencies(self):
        """Links files based on Class usage"""
        for filename, deps in self.pending_deps.items():
            for dep_class in deps:
                if dep_class in self.class_to_file:
                    target_file = self.class_to_file[dep_class]
                    if target_file != filename:
                        self.file_dependency_graph.add_edge(filename, target_file)

    def get_code_priors(self, hard_weight=10.0, decay=0.5):
        """
        Generates priors:
        - String Literals -> File (Hard Weight)
        - String Literals -> Dependent Files (Decaying Weight - Transitive/Deep)
        """
        priors = defaultdict(lambda: defaultdict(float)) # token -> {filename: weight}
        
        # We want to propagate from Source (contains string) -> Downstream Dependencies
        # If OptionTest contains "USOPTIONS4", and OptionTest -> DummyTestBase -> ExecutionEngine
        # We want "USOPTIONS4" to have weight in DummyTestBase and ExecutionEngine.
        
        # Precompute all-pairs shortest paths or just do BFS for each source
        # using networkx.
        
        for filename, strings in self.file_literals.items():
            if not strings: continue
            
            # Find all reachable nodes from 'filename' in the dependency graph
            # We want to go "down" the dependency chain.
            # If A depends on B (A->B), and A has string "S".
            # "S" is contextually relevant to B (because A uses B in the context of S).
            
            # BFS to find all descendants
            descendants = {} # filename -> distance
            
            if filename in self.file_dependency_graph:
                # Use BFS to get distances
                for v in self.file_dependency_graph.nodes():
                    if v == filename: continue
                    try:
                        path_len = nx.shortest_path_length(self.file_dependency_graph, source=filename, target=v)
                        descendants[v] = path_len
                    except nx.NetworkXNoPath:
                        pass
            
            for s in strings:
                tokens = clean_and_tokenize(s)
                for t in tokens:
                    # 1. Hard Bind to current file (Source)
                    priors[t][filename] += hard_weight
                    
                    # 2. Propagate to dependencies with decay based on distance
                    for neighbor, distance in descendants.items():
                         # Weight = Hard * (Decay ^ Distance)
                         # e.g., dist=1 (DummyTestBase), weight = 10 * 0.5 = 5
                         # dist=2 (ExecutionEngine), weight = 10 * 0.25 = 2.5
                         weight = hard_weight * (decay ** distance)
                         if weight > 0.01: # Cutoff
                             priors[t][neighbor] += weight
                             
        return priors


if __name__ == "__main__":
    # Test execution
    indexer = JavaCodeIndexer("services/knowledge_graph/dummy_dataset/dummy_trading_sys_java/src")
    indexer.index_project()
    priors = indexer.get_code_priors()
    print(f"Generated {len(priors)} code priors.")
    # Example
    for k, v in list(priors.items())[:5]:
        print(f"Token '{k}': {dict(v)}")