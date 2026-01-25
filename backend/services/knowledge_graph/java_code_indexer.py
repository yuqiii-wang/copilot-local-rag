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
            
            # 1. Identify Class Definition (Map Class -> File)
            for path, node in tree:
                if isinstance(node, javalang.tree.ClassDeclaration):
                    self.class_to_file[node.name] = filename
                    
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
                
                # 4. Identify Dependencies (Inheritance)
                elif isinstance(node, javalang.tree.ClassDeclaration):
                    if node.extends:
                         if isinstance(node.extends, list):
                             for ext in node.extends:
                                 if hasattr(ext, 'name'): self.pending_deps[filename].add(ext.name)
                         elif hasattr(node.extends, 'name'):
                             self.pending_deps[filename].add(node.extends.name)

        except Exception as e:
            # print(f"Failed to parse {file_path}: {e}")
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
        - String Literals -> Dependent Files (Decaying Weight)
        """
        priors = defaultdict(lambda: defaultdict(float)) # token -> {filename: weight}
        
        for filename, strings in self.file_literals.items():
            for s in strings:
                # Tokenize string
                tokens = clean_and_tokenize(s)
                for t in tokens:
                    # Hard Bind to current file
                    priors[t][filename] += hard_weight
                    
                    # Propagate to dependencies (Diminishing weight)
                    # If File A depends on File B, and File A has a log string, 
                    # File B (the context) is somewhat relevant.
                    if filename in self.file_dependency_graph:
                        neighbors = list(self.file_dependency_graph.successors(filename))
                        for neighbor in neighbors:
                             priors[t][neighbor] += (hard_weight * decay)
                             
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