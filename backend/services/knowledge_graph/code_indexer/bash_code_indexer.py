import os
import re
from pathlib import Path
from collections import defaultdict
from services.knowledge_graph.code_indexer.code_indexer import BaseCodeIndexer

class BashCodeIndexer(BaseCodeIndexer):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.sourced_files = defaultdict(set) # Filename -> Set[Sourced filenames]

    def index_project(self):
        print(f"Indexing Bash project in {self.root_dir}...")
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.sh', '.bash')):
                    self.parse_file(Path(root) / file)
        
        self._resolve_dependencies()
        
        print(f"Indexing complete. Built dependency graph with {self.file_dependency_graph.number_of_nodes()} files and {self.file_dependency_graph.number_of_edges()} edges.")

    def parse_file(self, file_path):
        filename = file_path.name
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 1. Identify Dependencies (source or .)
            # matches: source path/to/file.sh or . path/to/file.sh
            sourced = re.findall(r'(?:source|\.)\s+([^\s;]+)', content)
            for s in sourced:
                # We often only get relative paths or vars. We'll try to match filename.
                self.sourced_files[filename].add(os.path.basename(s))

            # 2. Identify String Literals (Logs, Errors)
            # Matches " string " or ' string '
            literals_double = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', content)
            literals_single = re.findall(r"'([^']*)'", content)
            
            for val in literals_double + literals_single:
                self.file_literals[filename].append(val)

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

    def _resolve_dependencies(self):
        """Links files based on sourced files"""
        all_files = set()
        for root, _, files in os.walk(self.root_dir):
             for f in files: all_files.add(f)

        for filename, deps in self.sourced_files.items():
            for dep_name in deps:
                # Heuristic: if basename matches a file in project
                if dep_name in all_files:
                    if dep_name != filename:
                        self.file_dependency_graph.add_edge(filename, dep_name)



if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    indexer = BashCodeIndexer(path)
    indexer.index_project()
    priors = indexer.get_code_priors()
    print(f"Generated {len(priors)} code priors.")
