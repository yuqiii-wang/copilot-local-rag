import os
import re
from pathlib import Path
from collections import defaultdict
from services.knowledge_graph.code_indexer.code_indexer import BaseCodeIndexer

class SqlCodeIndexer(BaseCodeIndexer):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        # Map TableName/ViewName -> Filename (where it is defined)
        self.table_to_file = {} 
        # Filename -> Set[TableNames it uses]
        self.pending_deps = defaultdict(set)  

    def index_project(self):
        print(f"Indexing SQL project in {self.root_dir}...")
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.sql'):
                    self.parse_file(Path(root) / file)
        
        self._resolve_dependencies()
        
        print(f"Indexing complete. Built dependency graph with {self.file_dependency_graph.number_of_nodes()} files and {self.file_dependency_graph.number_of_edges()} edges.")

    def parse_file(self, file_path):
        filename = file_path.name
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 1. Identify Table Definitions (CREATE TABLE table_name)
            create_stmts = re.findall(r'CREATE\s+TABLE\s+(\w+)', content, re.IGNORECASE)
            for table in create_stmts:
                self.table_to_file[table.lower()] = filename

            # 2. Identify View/Procedure Definitions (CREATE VIEW view_name)
            create_views = re.findall(r'CREATE\s+(?:OR\s+REPLACE\s+)?(?:VIEW|PROCEDURE)\s+(\w+)', content, re.IGNORECASE)
            for view in create_views:
                self.table_to_file[view.lower()] = filename

            # 3. Identify Dependencies (REFERENCES, JOIN, FROM, INSERT INTO, UPDATE)
            # REFERENCES x (Foreign Keys)
            refs = re.findall(r'REFERENCES\s+(\w+)', content, re.IGNORECASE)
            for r in refs:
                self.pending_deps[filename].add(r.lower())
                
            # FROM x (Selects)
            froms = re.findall(r'FROM\s+(\w+)', content, re.IGNORECASE)
            for f_name in froms:
                self.pending_deps[filename].add(f_name.lower())
            
            # JOIN x (Joins)
            joins = re.findall(r'JOIN\s+(\w+)', content, re.IGNORECASE)
            for j in joins:
                self.pending_deps[filename].add(j.lower())
                
            # INSERT INTO x
            inserts = re.findall(r'INSERT\s+INTO\s+(\w+)', content, re.IGNORECASE)
            for i in inserts:
                self.pending_deps[filename].add(i.lower())
                
            # UPDATE x
            updates = re.findall(r'UPDATE\s+(\w+)', content, re.IGNORECASE)
            for u in updates:
                self.pending_deps[filename].add(u.lower())

            # 4. Identify String Literals (Values in inserts, or logic)
            # Matches 'string'
            literals = re.findall(r"'([^']*)'", content)
            for val in literals:
                self.file_literals[filename].append(val)

        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

    def _resolve_dependencies(self):
        """Links files based on Table/View usage"""
        for filename, deps in self.pending_deps.items():
            for dep_table in deps:
                if dep_table in self.table_to_file:
                    target_file = self.table_to_file[dep_table]
                    # Avoid self-loops
                    if target_file != filename:
                        self.file_dependency_graph.add_edge(filename, target_file)



if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    indexer = SqlCodeIndexer(path)
    indexer.index_project()
    priors = indexer.get_code_priors()
    print(f"Generated {len(priors)} code priors.")
