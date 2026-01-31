import os
import javalang
from pathlib import Path
from collections import defaultdict
from services.knowledge_graph.code_indexer.code_indexer import BaseCodeIndexer

class JavaCodeIndexer(BaseCodeIndexer):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.class_to_file = {} # ClassName -> Filename
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




if __name__ == "__main__":
    # Test execution
    indexer = JavaCodeIndexer("services/knowledge_graph/dummy_dataset/dummy_trading_sys_java/src")
    indexer.index_project()
    priors = indexer.get_code_priors()
    print(f"Generated {len(priors)} code priors.")
    # Example
    for k, v in list(priors.items())[:5]:
        print(f"Token '{k}': {dict(v)}")