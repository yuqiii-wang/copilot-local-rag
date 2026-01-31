import os
import re
from pathlib import Path
from collections import defaultdict
from services.knowledge_graph.code_indexer.code_indexer import BaseCodeIndexer

class CppCodeIndexer(BaseCodeIndexer):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.includes = defaultdict(set) # Filename -> Set[Included filenames]

    def index_project(self):
        print(f"Indexing C++ project in {self.root_dir}...")
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.cpp', '.cc', '.cxx', '.c', '.h', '.hpp')):
                    self.parse_file(Path(root) / file)
        
        self._resolve_dependencies()
        
        print(f"Indexing complete. Built dependency graph with {self.file_dependency_graph.number_of_nodes()} files and {self.file_dependency_graph.number_of_edges()} edges.")

    def parse_file(self, file_path):
        filename = file_path.name
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 1. Identify Includes (Dependencies)
            # Matches #include "header.h"
            local_includes = re.findall(r'#include\s*"([^"]+)"', content)
            for inc in local_includes:
                self.includes[filename].add(os.path.basename(inc))

            # 2. Identify String Literals (Logs, Errors)
            # Matches "string literal"
            literals = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content)
            for val in literals:
                self.file_literals[filename].append(val)

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

    def _resolve_dependencies(self):
        """Links files based on Includes"""
        all_files = set()
        for root, _, files in os.walk(self.root_dir):
             for f in files: all_files.add(f)

        for filename, dependencies in self.includes.items():
            for dep_file in dependencies:
                # Simple matching by filename functionality
                if dep_file in all_files:
                    if dep_file != filename:
                        self.file_dependency_graph.add_edge(filename, dep_file)



if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    indexer = CppCodeIndexer(path)
    indexer.index_project()
    priors = indexer.get_code_priors()
    print(f"Generated {len(priors)} code priors.")
